"""Tests for exec tool internal URL blocking."""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

from nanobot.agent.tools.shell import ExecTool


def _fake_resolve_private(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.169.254", 0))]


def _fake_resolve_localhost(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]


def _fake_resolve_public(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]


@pytest.mark.asyncio
async def test_exec_blocks_curl_metadata():
    tool = ExecTool()
    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_private):
        result = await tool.execute(
            command='curl -s -H "Metadata-Flavor: Google" http://169.254.169.254/computeMetadata/v1/'
        )
    assert "Error" in result
    assert "internal" in result.lower() or "private" in result.lower()


@pytest.mark.asyncio
async def test_exec_blocks_wget_localhost():
    tool = ExecTool()
    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_localhost):
        result = await tool.execute(command="wget http://localhost:8080/secret -O /tmp/out")
    assert "Error" in result


@pytest.mark.asyncio
async def test_exec_allows_normal_commands():
    tool = ExecTool(timeout=5)
    result = await tool.execute(command="echo hello")
    assert "hello" in result
    assert "Error" not in result.split("\n")[0]


@pytest.mark.asyncio
async def test_exec_allows_curl_to_public_url():
    """Commands with public URLs should not be blocked by the internal URL check."""
    tool = ExecTool()
    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_public):
        guard_result = tool._guard_command("curl https://example.com/api", "/tmp")
    assert guard_result is None


@pytest.mark.asyncio
async def test_exec_blocks_chained_internal_url():
    """Internal URLs buried in chained commands should still be caught."""
    tool = ExecTool()
    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_private):
        result = await tool.execute(
            command="echo start && curl http://169.254.169.254/latest/meta-data/ && echo done"
        )
    assert "Error" in result


# --- #2989: block writes to nanobot internal state files -----------------


@pytest.mark.parametrize(
    "command",
    [
        "cat foo >> history.jsonl",
        "echo '{}' > history.jsonl",
        "echo '{}' > memory/history.jsonl",
        "echo '{}' > ./workspace/memory/history.jsonl",
        "tee -a history.jsonl < foo",
        "tee history.jsonl",
        "cp /tmp/fake.jsonl history.jsonl",
        "mv backup.jsonl memory/history.jsonl",
        "dd if=/dev/zero of=memory/history.jsonl",
        "sed -i 's/old/new/' history.jsonl",
        "echo x > .dream_cursor",
        "cp /tmp/x memory/.dream_cursor",
    ],
)
def test_exec_blocks_writes_to_history_jsonl(command):
    """Direct writes to history.jsonl / .dream_cursor must be blocked (#2989)."""
    tool = ExecTool()
    result = tool._guard_command(command, "/tmp")
    assert result is not None
    assert "dangerous pattern" in result.lower()


@pytest.mark.parametrize(
    "command",
    [
        "cat history.jsonl",
        "wc -l history.jsonl",
        "tail -n 5 history.jsonl",
        "grep foo history.jsonl",
        "cp history.jsonl /tmp/history.backup",
        "ls memory/",
        "echo history.jsonl",
    ],
)
def test_exec_allows_reads_of_history_jsonl(command):
    """Read-only access to history.jsonl must still be allowed."""
    tool = ExecTool()
    result = tool._guard_command(command, "/tmp")
    assert result is None


# --- #2826: working_dir must not escape the configured workspace ---------


@pytest.mark.asyncio
async def test_exec_blocks_working_dir_outside_workspace(tmp_path):
    """An LLM-supplied working_dir outside the workspace must be rejected."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True)
    result = await tool.execute(command="rm calendar.ics", working_dir="/etc")
    assert "outside the configured workspace" in result


@pytest.mark.asyncio
async def test_exec_blocks_absolute_rm_via_hijacked_working_dir(tmp_path):
    """Regression for #2826: `rm /abs/path` via working_dir hijack."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    victim_dir = tmp_path / "outside"
    victim_dir.mkdir()
    victim = victim_dir / "file.ics"
    victim.write_text("data")

    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True)
    result = await tool.execute(
        command=f"rm {victim}",
        working_dir=str(victim_dir),
    )
    assert "outside the configured workspace" in result
    assert victim.exists(), "victim file must not have been deleted"


@pytest.mark.asyncio
async def test_exec_allows_working_dir_within_workspace(tmp_path):
    """A working_dir that is a subdirectory of the workspace is fine."""
    workspace = tmp_path / "workspace"
    subdir = workspace / "project"
    subdir.mkdir(parents=True)
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True, timeout=5)
    result = await tool.execute(command="echo ok", working_dir=str(subdir))
    assert "ok" in result
    assert "outside the configured workspace" not in result


@pytest.mark.asyncio
async def test_exec_allows_working_dir_equal_to_workspace(tmp_path):
    """Passing working_dir equal to the workspace root must be allowed."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True, timeout=5)
    result = await tool.execute(command="echo ok", working_dir=str(workspace))
    assert "ok" in result
    assert "outside the configured workspace" not in result


@pytest.mark.asyncio
async def test_exec_ignores_workspace_check_when_not_restricted(tmp_path):
    """Without restrict_to_workspace, the LLM may still choose any working_dir."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=False, timeout=5)
    result = await tool.execute(command="echo ok", working_dir=str(other))
    assert "ok" in result
    assert "outside the configured workspace" not in result


# --- Relaxed deny_patterns: avoid false positives ------------------------
#
# Background: the previous regex `\brm\s+-[rf]{1,2}\b` blocked every `rm -rf`
# regardless of target. Confirmed production block: `rm -rf squashfs-root`
# during AppImage extraction in /tmp aborted the agent turn.

@pytest.mark.parametrize(
    "command",
    [
        "rm -rf squashfs-root",                  # production case
        "rm -rf node_modules",
        "rm -rf ./build",
        "rm -rf /tmp/foo",
        "rm -rf /tmp/extract-dir/",
        "rm -r dist",
        "rm -fr cache",
    ],
)
def test_exec_allows_rm_rf_on_local_paths(command):
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
        # shutdown/reboot should NOT match when used as text in other commands
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


# --- Marker decoupling --------------------------------------------------
#
# Soft policy errors (deny_patterns, allow_patterns, internal URL) should
# return "rejected by policy" so runner._is_workspace_violation does NOT
# abort the turn. Workspace boundary errors (path traversal, path outside
# workdir) keep "blocked by safety guard" so the runner aborts.

def test_deny_pattern_uses_rejected_by_policy_prefix():
    tool = ExecTool()
    result = tool._guard_command("rm -rf /etc", "/tmp")
    assert result is not None
    assert "rejected by policy" in result.lower()
    assert "blocked by safety guard" not in result.lower()


def test_allowlist_uses_rejected_by_policy_prefix():
    tool = ExecTool(allow_patterns=[r"^echo "])
    result = tool._guard_command("ls", "/tmp")
    assert result is not None
    assert "rejected by policy" in result.lower()
    assert "blocked by safety guard" not in result.lower()


def test_internal_url_uses_rejected_by_policy_prefix():
    tool = ExecTool()
    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_localhost):
        result = tool._guard_command("curl http://localhost:8080/x", "/tmp")
    assert result is not None
    assert "rejected by policy" in result.lower()
    assert "blocked by safety guard" not in result.lower()


def test_path_outside_workdir_keeps_safety_guard_marker(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True)
    result = tool._guard_command("cat /etc/passwd", str(workspace))
    assert result is not None
    assert "blocked by safety guard" in result.lower()


# --- Opt-in internal URL block (block_internal_urls) ---------------------

def test_block_internal_urls_default_true_blocks():
    tool = ExecTool()  # default block_internal_urls=True
    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_localhost):
        result = tool._guard_command("curl http://localhost/api", "/tmp")
    assert result is not None
    assert "internal" in result.lower() or "private" in result.lower()


def test_block_internal_urls_false_allows():
    """With block_internal_urls=False the LLM can curl LAN/Docker services."""
    tool = ExecTool(block_internal_urls=False)
    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_localhost):
        result = tool._guard_command("curl http://nanobot-arnaldo:8080/health", "/tmp")
    assert result is None


def test_block_internal_urls_false_still_blocks_dangerous_patterns():
    """Disabling internal URL block doesn't disable other guards."""
    tool = ExecTool(block_internal_urls=False)
    result = tool._guard_command("rm -rf /etc", "/tmp")
    assert result is not None
    assert "rejected by policy" in result.lower()


# --- Safe device pseudo-files in redirects --------------------------------
#
# Common shell idioms like `2>/dev/null` were misclassified as
# "path outside working dir" because the regex extracts `/dev/null`
# as a path argument. /dev/null and friends are not data targets and
# must not trigger the boundary guard.

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
    """Whitelist only covers /dev/* pseudo-files, not arbitrary paths."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True)
    result = tool._guard_command("find /etc -name passwd 2>/dev/null", str(workspace))
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


# Double-slash false positives. `a // b` (Python integer division), `// foo`
# (C/JS line comment), and scheme-less URL fragments all contain ` //` which
# the absolute-path regex used to capture as the bogus path `/<rest>`. Once
# resolved via Path.resolve(), the doubled slash collapses to a single one
# pointing somewhere outside the workspace, and the boundary guard fires on a
# perfectly innocent inline snippet. Real cases observed: `len(blob) // 4`
# inside `python3 -c "..."` killed multiple grocco turns.
@pytest.mark.parametrize(
    "command",
    [
        'python3 -c "x = len(blob) // 4"',
        'python3 -c "print(10 // 3)"',
        "echo $((10 // 2))",
        'node -e "// noop\nconsole.log(1)"',
        "awk '{ print $1 // $2 }' file.txt",
    ],
)
def test_double_slash_is_not_treated_as_absolute_path(tmp_path, command):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True)
    result = tool._guard_command(command, str(workspace))
    assert result is None or "blocked by safety guard" not in result.lower(), command
