"""Bronzo safetyguard false-positive port onto v0.2.0 (migration task #3).

Covers the custom shell-guard hardening that v0.2.0 upstream does NOT
have: system-root-scoped rm -rf, anchored destructive-command patterns,
the read-safe `/proc /etc /usr …` prefix whitelist (read-only for the
system tier), the `//` and bare-`/` absolute-path fixes, and the
quote/resolve-aware `../` traversal check.

Design note: v0.2.0 added `_WORKSPACE_BOUNDARY_NOTE` and classifies
violations by *content marker* (`runner._WORKSPACE_VIOLATION_MARKERS`
matches "path outside working dir" / "path traversal detected"), not by
a "rejected by policy" vs "blocked by safety guard" prefix. The custom
"rejected by policy" rename and the runner softening were therefore
dropped as redundant; the boundary uses upstream's single policy voice.
The marker substrings are preserved so the peer error-loop break
(migration task #6) still classifies these turns correctly.
"""

from __future__ import annotations

import pytest

from nanobot.agent.tools.shell import ExecTool


# --- deny_patterns: system-root-scoped rm -rf + anchored destructive ----

@pytest.mark.parametrize(
    "command",
    [
        "rm -rf node_modules",
        "rm -rf ./build",
        "rm -rf build/cache",
        "rm -r dist",
        "rm -fr cache",
    ],
)
def test_exec_allows_rm_rf_on_local_paths(command):
    # Only relative targets are allowed; any absolute path / ~ / $HOME is
    # denied (reconciled policy — see test_exec_blocks_rm_rf_on_system_roots).
    tool = ExecTool()
    assert tool._guard_command(command, "/tmp") is None, command


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -rf / ",
        "rm -rf /etc",
        "rm -rf /etc/passwd",
        "rm -rf /usr/bin",
        "rm -rf /home",
        "rm -rf /var/log",
        "rm -rf /lib",
        # Reconciled policy: ANY absolute path is denied, not just system
        # roots (v0.2.0 made blocks soft, so the looser /tmp carve-out is
        # no longer worth the security cost).
        "rm -rf /tmp/build",
        "rm -rf /var/lib/private",
        "rm -rf ~",
        "rm -rf ~/",
        "rm -rf $HOME",
    ],
)
def test_exec_blocks_rm_rf_on_system_roots(command):
    tool = ExecTool()
    assert tool._guard_command(command, "/tmp") is not None, command


@pytest.mark.parametrize(
    "command",
    [
        "grep shutdown /var/log/syslog",
        'echo "system will shutdown soon"',
        "cat /var/log/reboot.log",
        "ls /etc/format-rules.d/",
        "cd ~/projects/diskpart-tool",
    ],
)
def test_exec_allows_keywords_inside_other_commands(command):
    tool = ExecTool()
    assert tool._guard_command(command, "/tmp") is None, command


@pytest.mark.parametrize(
    "command",
    [
        "shutdown -h now",
        "sudo reboot",
        "poweroff",
        "halt",
        "init 0",
        "init 6",
        "mkfs.ext4 /dev/sda1",
        "sudo mkfs.xfs /dev/nvme0n1p1",
        "diskpart",
        "dd if=/dev/zero of=/dev/sda",
        "dd if=disk.img of=/dev/nvme0n1",
        "echo data > /dev/sda1",
        ":(){ :|:& };:",
    ],
)
def test_exec_blocks_truly_dangerous_commands(command):
    tool = ExecTool()
    assert tool._guard_command(command, "/tmp") is not None, command


# --- benign device redirect targets --------------------------------------

@pytest.mark.parametrize(
    "command",
    [
        'find . -name "x" -type f 2>/dev/null',
        "ls -la 2>/dev/null && echo ok",
        "grep foo bar.txt >/dev/null",
        "cat </dev/stdin",
        "echo hi >/dev/stderr",
        "cmd >/dev/null 2>&1",
    ],
)
def test_redirect_to_safe_device_does_not_trigger_workspace_guard(tmp_path, command):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True)
    result = tool._guard_command(command, str(workspace))
    assert result is None or "blocked by safety guard" not in result.lower(), command


def test_real_path_outside_workspace_still_blocks(tmp_path):
    """Whitelist doesn't cover arbitrary out-of-tree paths.

    /var/lib is intentionally NOT in `_SAFE_READ_PREFIXES_*` (it holds
    per-application state we don't want the agent inspecting by default).
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True)
    result = tool._guard_command(
        "find /var/lib/private -name secret 2>/dev/null", str(workspace)
    )
    assert result is not None
    assert "blocked by safety guard" in result.lower()


def test_redirect_to_arbitrary_outside_path_still_blocks(tmp_path):
    """Redirecting to a non-device path outside workspace stays blocked."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True)
    result = tool._guard_command("echo data >/etc/exfil", str(workspace))
    assert result is not None
    assert "blocked by safety guard" in result.lower()


# --- `//` and bare `/` are not absolute paths ----------------------------

@pytest.mark.parametrize(
    "command",
    [
        'python3 -c "x = len(blob) // 4"',
        'python3 -c "print(10 // 3)"',
        "echo $((10 // 2))",
        'node -e "// noop\nconsole.log(1)"',
        "awk '{ print $1 // $2 }' file.txt",
        # Regular division `a / b` used to capture a bare `/` as fs root.
        'python3 -c "x = a / b"',
        "python3 -c \"vec_dim = chunk_size / (n_vecs * 4)\"",
        "awk '{ print $1 / $2 }' file.txt",
        "echo $((10 / 2))",
        # `find /` standalone: bare-root path no longer captured at all.
        "find / -name foo",
    ],
)
def test_double_slash_is_not_treated_as_absolute_path(tmp_path, command):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True)
    result = tool._guard_command(command, str(workspace))
    assert result is None or "blocked by safety guard" not in result.lower(), command


# --- read-safe system prefix whitelist (read-only for system tier) -------

@pytest.mark.parametrize(
    "command",
    [
        # Virtual / scratch (read & write both fine)
        "cat /proc/self/status",
        "cat /sys/class/net/eth0/address",
        "ls /tmp",
        "cat /tmp/scratch.txt",
        "echo data > /tmp/output.txt",
        "ls /var/tmp/cache",
        "ls /run/user/1000/",
        "ls /dev/shm",
        "ls /dev/fd/",
        # System read-only (read paths)
        "cat /etc/resolv.conf",
        "grep nanobot /etc/hosts",
        "find /etc -name passwd",
        "tail -n 50 /var/log/syslog",
        "/usr/bin/env python3 -c 'print(1)'",
        "python3 /usr/local/bin/myscript.py",
        "ls /opt/some-vendor/bin",
        "file /bin/bash",
    ],
)
def test_read_safe_system_paths_are_allowed(tmp_path, command):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True)
    result = tool._guard_command(command, str(workspace))
    assert result is None or "blocked by safety guard" not in result.lower(), command


@pytest.mark.parametrize(
    "command",
    [
        "echo evil > /etc/cron.d/x",
        "echo evil >> /usr/local/bin/whoops.sh",
        "echo evil > /opt/marker",
        "echo evil > /var/log/forged.log",
        "echo evil > /bin/forged",
    ],
)
def test_redirect_into_system_ro_prefix_still_blocks(tmp_path, command):
    """A read-whitelisted prefix doesn't allow writes via shell redirect."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True)
    result = tool._guard_command(command, str(workspace))
    assert result is not None, command


# --- quote-aware + resolve-aware `../` traversal -------------------------

@pytest.mark.parametrize(
    "command",
    [
        # `../` inside quoted strings — not navigation, just data
        "grep '../' file.txt",
        'python3 -c "print(\\"../foo\\")"',
        'find . -path "*/tests/../legacy/*"',
        # in-tree convoluted paths (resolve back inside the workspace)
        "pytest tests/../tests/specific/",
        "ls a/b/../b/c",
        # commands with `..` but no `../`
        "git log HEAD~5..HEAD",
        "git diff HEAD~3..HEAD",
    ],
)
def test_path_traversal_allows_legitimate_uses(tmp_path, command):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True)
    result = tool._guard_command(command, str(workspace))
    assert result is None or "path traversal" not in result.lower(), command


@pytest.mark.parametrize(
    "command",
    [
        "cat ../../etc/passwd",
        "rm -rf ../sibling",
        "cp ../outside/file .",
    ],
)
def test_path_traversal_blocks_real_escape(tmp_path, command):
    """Relative `../` that actually escapes the workspace still blocks.

    On v0.2.0 this uses upstream's single policy voice: the message
    carries "path traversal detected" (so runner classifies it as a
    workspace_violation, keeping the peer error-loop break working) and
    the "blocked by safety guard" marker. The custom "rejected by policy"
    rename was dropped — v0.2.0 already makes every violation soft, so
    the rename had no functional effect.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True)
    result = tool._guard_command(command, str(workspace))
    assert result is not None, command
    assert "path traversal" in result.lower(), command
