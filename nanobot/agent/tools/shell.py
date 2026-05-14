"""Shell execution tool."""

import asyncio
import os
import re
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.sandbox import wrap_command
from nanobot.agent.tools.schema import IntegerSchema, StringSchema, tool_parameters_schema
from nanobot.config.paths import get_media_dir

_IS_WINDOWS = sys.platform == "win32"

# Pseudo-devices that are safe even when the workspace boundary is enforced.
# They appear in shell pipelines as redirect targets (`2>/dev/null`) or stream
# placeholders, never as actual data sinks. Treating them like ordinary paths
# was the most common cause of false-positive workspace blocks.
_SAFE_DEVICE_PATHS = frozenset({
    "/dev/null",
    "/dev/stdin",
    "/dev/stdout",
    "/dev/stderr",
    "/dev/tty",
    "/dev/zero",
    "/dev/random",
    "/dev/urandom",
    "/dev/full",
})

# Path prefixes the workspace-boundary check treats as safe-to-touch even
# when the resolved path lies outside the agent's workspace root. Two tiers:
#
# - "virtual": kernel-managed pseudo-filesystems and user-scoped scratch.
#   These are either read-only by nature (/proc, /sys) or owned by the
#   running user and meant for ephemeral I/O (/tmp, /var/tmp, /run/user,
#   /dev/shm, /dev/fd). Whitelisted for both reads and writes — the OS
#   permission model already enforces the limits that matter.
#
# - "system_ro": canonical system locations that hold configuration and
#   binaries. Whitelisted for *reads* only. A path under one of these
#   prefixes is allowed when it appears as an argument or input, but
#   blocked when it appears as the target of a shell redirect (`> /etc/x`),
#   so the LLM can still cat/grep/find/exec system files while remaining
#   prevented from writing into them via the guard.
_SAFE_READ_PREFIXES_VIRTUAL = (
    "/proc",
    "/sys",
    "/tmp",
    "/var/tmp",
    "/run/user",
    "/dev/shm",
    "/dev/fd",
)
_SAFE_READ_PREFIXES_SYSTEM_RO = (
    "/etc",
    "/usr",
    "/opt",
    "/var/log",
    "/bin",
    "/sbin",
    "/lib",
    "/lib32",
    "/lib64",
)


def _path_under(prefix: str, p: Path) -> bool:
    """True if `p` is exactly `prefix` or sits under it (no string prefix bug)."""
    s = str(p)
    return s == prefix or s.startswith(prefix + "/")


@tool_parameters(
    tool_parameters_schema(
        command=StringSchema("The shell command to execute"),
        working_dir=StringSchema("Optional working directory for the command"),
        timeout=IntegerSchema(
            60,
            description=(
                "Timeout in seconds. Increase for long-running commands "
                "like compilation or installation (default 60, max 600)."
            ),
            minimum=1,
            maximum=600,
        ),
        required=["command"],
    )
)
class ExecTool(Tool):
    """Tool to execute shell commands."""

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
        sandbox: str = "",
        path_append: str = "",
        allowed_env_keys: list[str] | None = None,
        block_internal_urls: bool = True,
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        self.sandbox = sandbox
        self.block_internal_urls = block_internal_urls
        # System-critical roots: rm -rf on these is almost always catastrophic.
        # Note: /tmp, /proc, /run, /mnt, /media intentionally excluded — legit
        # cleanup targets. /dev included because removing device nodes breaks
        # the system without obvious recovery.
        _system_roots = r"(?:bin|boot|etc|root|home|usr|var|opt|sys|lib|sbin|dev|lib32|lib64|libx32)"
        self.deny_patterns = deny_patterns or [
            # rm -rf targeting system roots, $HOME, or ~ (allows rm -rf node_modules,
            # rm -rf /tmp/foo, rm -rf squashfs-root, etc.)
            rf"\brm\s+-[rf]+\s+(?:/\s|/$|/{_system_roots}(?:/|\s|$)|~(?:/?\s|/?$)|\$home\b)",
            # Windows destructive deletes (anchored to command start)
            r"(?:^|[;&|]\s*)\bdel\s+/[fq]\b",
            r"(?:^|[;&|]\s*)\brmdir\s+/s\b",
            # format / mkfs / diskpart (anchored — avoids matching inside grep/echo)
            r"(?:^|[;&|]\s*)\b(?:sudo\s+)?format\b",
            r"(?:^|[;&|]\s*)\b(?:sudo\s+)?(?:mkfs(?:\.\w+)?|diskpart)\b",
            # System power (anchored — avoids matching `grep shutdown /var/log/...`)
            r"(?:^|[;&|]\s*)\b(?:sudo\s+)?(?:shutdown|reboot|poweroff|halt|init\s+[06])\b",
            # dd writing to a block device (the dangerous direction)
            r"\bdd\b[^|;&<>]*\bof=\s*/dev/(?:sd|nvme|hd|mmcblk|xvd|loop|vd)",
            # Redirect to block device
            r">\s*/dev/(?:sd|nvme|hd|mmcblk|xvd|loop|vd)",
            # Fork bomb
            r":\(\)\s*\{.*\};\s*:",
            # Block writes to nanobot internal state files (#2989).
            # history.jsonl / .dream_cursor are managed by append_history();
            # direct writes corrupt the cursor format and crash /dream.
            r">>?\s*\S*(?:history\.jsonl|\.dream_cursor)",            # > / >> redirect
            r"\btee\b[^|;&<>]*(?:history\.jsonl|\.dream_cursor)",     # tee / tee -a
            r"\b(?:cp|mv)\b(?:\s+[^\s|;&<>]+)+\s+\S*(?:history\.jsonl|\.dream_cursor)",  # cp/mv target
            r"\bdd\b[^|;&<>]*\bof=\S*(?:history\.jsonl|\.dream_cursor)",  # dd of=
            r"\bsed\s+-i[^|;&<>]*(?:history\.jsonl|\.dream_cursor)",  # sed -i
        ]
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace
        self.path_append = path_append
        self.allowed_env_keys = allowed_env_keys or []

    @property
    def name(self) -> str:
        return "exec"

    _MAX_TIMEOUT = 600
    _MAX_OUTPUT = 10_000

    @property
    def description(self) -> str:
        return (
            "Execute a shell command and return its output. "
            "Prefer read_file/write_file/edit_file over cat/echo/sed, "
            "and grep/glob over shell find/grep. "
            "Use -y or --yes flags to avoid interactive prompts. "
            "Output is truncated at 10 000 chars; timeout defaults to 60s."
        )

    @property
    def exclusive(self) -> bool:
        return True

    async def execute(
        self, command: str, working_dir: str | None = None,
        timeout: int | None = None, **kwargs: Any,
    ) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()

        # Prevent an LLM-supplied working_dir from escaping the configured
        # workspace when restrict_to_workspace is enabled (#2826). Without
        # this, a caller can pass working_dir="/etc" and then all absolute
        # paths under /etc would pass the _guard_command check that anchors
        # on cwd.
        if self.restrict_to_workspace and self.working_dir:
            try:
                requested = Path(cwd).expanduser().resolve()
                workspace_root = Path(self.working_dir).expanduser().resolve()
            except Exception:
                return "Error: working_dir could not be resolved"
            if requested != workspace_root and workspace_root not in requested.parents:
                return "Error: working_dir is outside the configured workspace"

        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error

        if self.sandbox:
            if _IS_WINDOWS:
                logger.warning(
                    "Sandbox '{}' is not supported on Windows; running unsandboxed",
                    self.sandbox,
                )
            else:
                workspace = self.working_dir or cwd
                command = wrap_command(self.sandbox, command, workspace, cwd)
                cwd = str(Path(workspace).resolve())

        effective_timeout = min(timeout or self.timeout, self._MAX_TIMEOUT)
        env = self._build_env()

        if self.path_append:
            if _IS_WINDOWS:
                env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append
            else:
                env["NANOBOT_PATH_APPEND"] = self.path_append
                command = f'export PATH="$PATH{os.pathsep}$NANOBOT_PATH_APPEND"; {command}'

        try:
            process = await self._spawn(command, cwd, env)

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                await self._kill_process(process)
                return f"Error: Command timed out after {effective_timeout} seconds"
            except asyncio.CancelledError:
                await self._kill_process(process)
                raise

            output_parts = []

            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))

            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")

            output_parts.append(f"\nExit code: {process.returncode}")

            result = "\n".join(output_parts) if output_parts else "(no output)"

            max_len = self._MAX_OUTPUT
            if len(result) > max_len:
                half = max_len // 2
                result = (
                    result[:half]
                    + f"\n\n... ({len(result) - max_len:,} chars truncated) ...\n\n"
                    + result[-half:]
                )

            return result

        except Exception as e:
            return f"Error executing command: {str(e)}"

    @staticmethod
    async def _spawn(
        command: str, cwd: str, env: dict[str, str],
    ) -> asyncio.subprocess.Process:
        """Launch *command* in a platform-appropriate shell."""
        if _IS_WINDOWS:
            comspec = env.get("COMSPEC", os.environ.get("COMSPEC", "cmd.exe"))
            return await asyncio.create_subprocess_exec(
                comspec, "/c", command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
        bash = shutil.which("bash") or "/bin/bash"
        return await asyncio.create_subprocess_exec(
            bash, "-l", "-c", command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

    @staticmethod
    async def _kill_process(process: asyncio.subprocess.Process) -> None:
        """Kill a subprocess and reap it to prevent zombies."""
        process.kill()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        finally:
            if not _IS_WINDOWS:
                try:
                    os.waitpid(process.pid, os.WNOHANG)
                except (ProcessLookupError, ChildProcessError) as e:
                    logger.debug("Process already reaped or not found: {}", e)

    def _build_env(self) -> dict[str, str]:
        """Build a minimal environment for subprocess execution.

        On Unix, only HOME/LANG/TERM are passed; ``bash -l`` sources the
        user's profile which sets PATH and other essentials.

        On Windows, ``cmd.exe`` has no login-profile mechanism, so a curated
        set of system variables (including PATH) is forwarded.  API keys and
        other secrets are still excluded.
        """
        if _IS_WINDOWS:
            sr = os.environ.get("SYSTEMROOT", r"C:\Windows")
            env = {
                "SYSTEMROOT": sr,
                "COMSPEC": os.environ.get("COMSPEC", f"{sr}\\system32\\cmd.exe"),
                "USERPROFILE": os.environ.get("USERPROFILE", ""),
                "HOMEDRIVE": os.environ.get("HOMEDRIVE", "C:"),
                "HOMEPATH": os.environ.get("HOMEPATH", "\\"),
                "TEMP": os.environ.get("TEMP", f"{sr}\\Temp"),
                "TMP": os.environ.get("TMP", f"{sr}\\Temp"),
                "PATHEXT": os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD"),
                "PATH": os.environ.get("PATH", f"{sr}\\system32;{sr}"),
                "APPDATA": os.environ.get("APPDATA", ""),
                "LOCALAPPDATA": os.environ.get("LOCALAPPDATA", ""),
                "ProgramData": os.environ.get("ProgramData", ""),
                "ProgramFiles": os.environ.get("ProgramFiles", ""),
                "ProgramFiles(x86)": os.environ.get("ProgramFiles(x86)", ""),
                "ProgramW6432": os.environ.get("ProgramW6432", ""),
            }
            for key in self.allowed_env_keys:
                val = os.environ.get(key)
                if val is not None:
                    env[key] = val
            return env
        home = os.environ.get("HOME", "/tmp")
        env = {
            "HOME": home,
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "TERM": os.environ.get("TERM", "dumb"),
        }
        for key in self.allowed_env_keys:
            val = os.environ.get(key)
            if val is not None:
                env[key] = val
        return env

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Best-effort safety guard for potentially destructive commands.

        Two error prefixes are used intentionally:
        - "rejected by policy" for soft denials (deny_patterns, allowlist,
          internal URL, path traversal): the LLM sees the error as a tool
          result and may retry with a corrected command.
        - "blocked by safety guard" for absolute-path workspace-boundary
          violations: runner classifies these as workspace_violation and
          aborts the turn (final, last-resort safety net).

        Path traversal lives in the soft tier because the check is a
        regex-level heuristic on a single argument that the LLM can almost
        always fix (use absolute paths, `cd` first, drop the `..`). It is
        not a sign of a real escape attempt — `ln -sf ../foo /ws/bar`
        builds a symlink whose target the kernel resolves relative to the
        symlink's directory, not cwd.
        """
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command rejected by policy (dangerous pattern detected)"

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command rejected by policy (not in allowlist)"

        if self.block_internal_urls:
            from nanobot.security.network import contains_internal_url
            if contains_internal_url(cmd):
                return "Error: Command rejected by policy (internal/private URL detected)"

        if self.restrict_to_workspace:
            cwd_path = Path(cwd).resolve()

            traversal_error = self._check_path_traversal(cmd, cwd_path)
            if traversal_error:
                return traversal_error

            media_path = get_media_dir().resolve()

            for prefix_char, raw in self._extract_absolute_paths(cmd):
                try:
                    expanded = os.path.expandvars(raw.strip())
                    # Standard device pseudo-files used as redirect sinks/sources
                    # (`2>/dev/null`, `</dev/stdin`, `>/dev/stderr`, …). They
                    # are not data targets and were a frequent false-positive
                    # on otherwise-safe shell pipelines. Checked pre-resolve
                    # because /dev/stderr et al. are symlinks into /proc/self
                    # that would otherwise resolve outside the workspace.
                    if expanded in _SAFE_DEVICE_PATHS:
                        continue
                    p = Path(expanded).expanduser().resolve()
                except Exception:
                    continue

                if not p.is_absolute():
                    continue
                if cwd_path == p or cwd_path in p.parents:
                    continue
                if media_path == p or media_path in p.parents:
                    continue

                # Virtual / scratch prefixes: always permitted. /proc & /sys are
                # kernel pseudo-fs, /tmp et al. are user-owned scratch.
                if any(_path_under(prefix, p) for prefix in _SAFE_READ_PREFIXES_VIRTUAL):
                    continue

                # System read-only prefixes: permitted only when the path is
                # being read, not when it is the destination of a redirect.
                # `>` and `<` precede write/read redirects; `>` is the dangerous
                # one (writes), `<` is fine. We block on `>` to keep the guard
                # honest while still letting `cat /etc/resolv.conf` pass.
                is_redirect_target = prefix_char == ">"
                if not is_redirect_target and any(
                    _path_under(prefix, p) for prefix in _SAFE_READ_PREFIXES_SYSTEM_RO
                ):
                    continue

                return "Error: Command blocked by safety guard (path outside working dir)"

        return None

    @staticmethod
    def _strip_quoted_strings(s: str) -> str:
        """Replace contents of `'…'` and `"…"` with empty strings.

        Used to ignore `../` (and similar tokens) that live inside string
        literals — they are not shell tokens the shell will resolve, they are
        data being passed to the inner program.

        Note: this is intentionally NOT used for absolute-path extraction.
        `bash -c '…'` and `sh -c "…"` re-enter the shell on the inner
        contents, so absolute paths inside those quotes must still be
        checked. The traversal check is OK to relax because `../` in a
        re-entered shell command would be just as suspicious as in the
        outer one, and the workspace boundary check still catches the
        absolute case.
        """
        # Single-quoted: literal, no escapes
        s = re.sub(r"'[^']*'", "''", s)
        # Double-quoted: allow backslash escapes
        s = re.sub(r'"(?:\\.|[^"\\])*"', '""', s)
        return s

    def _check_path_traversal(self, command: str, cwd_path: Path) -> str | None:
        """Detect `../` traversal that escapes the workspace.

        Substring-matching `"../" in command` was a major false-positive
        source: it fired on `pytest tests/../tests/specific/` (resolves
        in-tree), `grep '../' file.txt` (the slash-dot-dot is the search
        pattern), `git log a..b` (no `../` but historically near misses).

        New heuristic:
        1. If the command contains no `..\\` or `../` at all → pass.
        2. If `../` only appears inside quoted strings → pass.
        3. Otherwise shell-tokenize the command; for each token containing
           `../`, resolve it relative to cwd. If every such token resolves
           inside the workspace, pass. If any escapes, block.
        """
        if "..\\" not in command and "../" not in command:
            return None

        stripped = self._strip_quoted_strings(command)
        if "..\\" not in stripped and "../" not in stripped:
            return None

        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            # Malformed quoting: fall back to the conservative block. A
            # mismatched quote could let `../` leak through that the
            # stripping pass didn't catch.
            return (
                "Error: Command rejected by policy (path traversal detected). "
                "Try absolute paths inside the workspace or `cd` first."
            )

        for tok in tokens:
            if "../" not in tok and "..\\" not in tok:
                continue
            # Absolute & home paths are handled by the absolute-path check.
            if tok.startswith("/") or tok.startswith("~"):
                continue
            try:
                resolved = (cwd_path / tok).resolve()
            except Exception:
                return (
                    "Error: Command rejected by policy (path traversal detected). "
                    "Try absolute paths inside the workspace or `cd` first."
                )
            if resolved != cwd_path and cwd_path not in resolved.parents:
                return (
                    "Error: Command rejected by policy (path traversal detected). "
                    "Try absolute paths inside the workspace or `cd` first."
                )

        return None

    @staticmethod
    def _extract_absolute_paths(command: str) -> list[tuple[str, str]]:
        """Return [(prefix_char, path), …] for each absolute path in `command`.

        `prefix_char` is the meaningful (non-whitespace) character immediately
        preceding the path: one of `>` (redirect-target), `|`, `'`, `"`, or
        `""` for start-of-string / preceded only by whitespace.

        Callers use `prefix_char == ">"` to tell read positions from write
        redirects so the read-safe prefix whitelist doesn't accidentally
        allow writes into /etc, /usr, etc.
        """
        results: list[tuple[str, str]] = []

        # Windows: drive-root paths like `C:\…`.  Redirect syntax with drive
        # paths is uncommon; default to "" prefix.
        for m in re.finditer(r"[A-Za-z]:\\[^\s\"'|><;]*", command):
            results.append(("", m.group(0)))

        # POSIX absolute and `~`-relative paths.
        #
        # `/(?!/)` avoids capturing `//…` (Python integer division, C/JS
        # comments, scheme-less URL fragments) as the bogus path `/<rest>`.
        #
        # `[^\s\"'>;|<]+` requires at least one non-separator char after the
        # leading `/`. A bare `/` (e.g. inside `chunk_size / (n * 4)` regular
        # Python division) would otherwise be captured as the filesystem-root
        # path, immediately fail the workspace-boundary check, and abort the
        # turn on commands that only happened to contain a slash operator.
        path_re = re.compile(r"(?:^|[\s|>'\"])(/(?!/)[^\s\"'>;|<]+|~[^\s\"'>;|<]*)")
        for m in path_re.finditer(command):
            # Walk left from the match start past any whitespace to find the
            # meaningful prefix char. `echo data > /etc/x` matches the space
            # before `/etc/x`, but the redirect-determining char is the `>`
            # that sits before the space.
            i = m.start(1) - 1
            while i >= 0 and command[i].isspace():
                i -= 1
            prefix_char = command[i] if i >= 0 else ""
            results.append((prefix_char, m.group(1)))

        return results
