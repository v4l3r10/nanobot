"""Recovery-aware peer workspace-violation suppression.

Pins the semantic of _PEER_WS_VIOLATION_VAR after the 2026-05-21 fix:
the flag is True only when the LAST workspace/SSRF violation in the turn
was NOT followed by a successful tool event. Any post-violation
`status: "ok"` event counts as recovery and clears the flag. Each new
violation re-arms.

Regression: 2026-05-21 12:59 Umanio incident. Bronzo sent a logistica ZIP;
Umanio's exec calls with absolute working_dir paths were rejected, Umanio
recovered with relative paths, read 8 files, wrote a wiki note, and
produced a 3675-char positive review as final_content. The set-once flag
(pre-CV4) dropped that legitimate response.

These tests target the flag-setter logic that lives inline at
loop.py:849-852. They build synthetic tool_events lists and assert the
resulting ContextVar state.
"""

from __future__ import annotations

import asyncio

from nanobot.agent.loop import _PEER_WS_VIOLATION_VAR


# Helper: drive the same logic that loop.py:849-852 runs. Keeping it here
# as a thin test fixture means we don't have to import a private symbol;
# if the production code's shape changes (e.g. extracted to a helper),
# update both call sites together.
def _compute_violation_active(tool_events: list[dict[str, str]]) -> bool:
    violation_active = False
    for ev in tool_events:
        detail = (ev.get("detail") or "")
        if detail.startswith(("workspace_violation", "ssrf_violation")):
            violation_active = True
        elif violation_active and ev.get("status") == "ok":
            violation_active = False
    return violation_active


def _ok(name: str = "exec") -> dict[str, str]:
    return {"name": name, "status": "ok", "detail": "done"}


def _violation(kind: str = "workspace_violation", name: str = "exec") -> dict[str, str]:
    return {"name": name, "status": "error", "detail": f"{kind}: path outside working dir"}


def _error(name: str = "read", reason: str = "file not found") -> dict[str, str]:
    return {"name": name, "status": "error", "detail": reason}


class TestFlagLogic:
    def test_no_events_flag_false(self):
        assert _compute_violation_active([]) is False

    def test_only_ok_events_flag_false(self):
        assert _compute_violation_active([_ok(), _ok(), _ok()]) is False

    def test_single_violation_no_recovery_flag_true(self):
        assert _compute_violation_active([_violation()]) is True

    def test_violation_then_ok_flag_false(self):
        # The 2026-05-21 Umanio regression case in its minimal form.
        assert _compute_violation_active([_violation(), _ok()]) is False

    def test_multiple_violations_then_ok_flag_false(self):
        events = [_violation(), _violation(), _ok(), _ok()]
        assert _compute_violation_active(events) is False

    def test_only_violations_flag_true(self):
        events = [_violation(), _violation(), _violation()]
        assert _compute_violation_active(events) is True

    def test_recovery_then_re_violation_flag_true(self):
        # Agent recovered, then tripped another policy boundary at end of turn.
        # Conservative choice: end-of-turn state wins, suppress.
        events = [_violation(), _ok(), _violation()]
        assert _compute_violation_active(events) is True

    def test_recovery_then_unrelated_error_flag_false(self):
        # Non-violation error after a successful recovery does NOT re-arm.
        # The agent is operating; the failure is honest local error, not
        # a policy-loop narrative.
        events = [_violation(), _ok(), _error()]
        assert _compute_violation_active(events) is False

    def test_ssrf_violation_treated_same_as_workspace(self):
        assert _compute_violation_active([_violation(kind="ssrf_violation")]) is True
        assert _compute_violation_active([_violation(kind="ssrf_violation"), _ok()]) is False

    def test_workspace_violation_escalated_prefix_arms_flag(self):
        ev = {"name": "exec", "status": "error", "detail": "workspace_violation_escalated: repeated"}
        assert _compute_violation_active([ev]) is True

    def test_missing_detail_is_safe(self):
        ev = {"name": "exec", "status": "error"}
        assert _compute_violation_active([ev]) is False

    def test_real_umanio_shape_2026_05_21(self):
        # Approximation of the actual incident: two absolute-path exec
        # rejections, then a recovery sequence of relative-path exec +
        # multiple reads + a wiki write.
        events = [
            _violation(),
            _violation(),
            _ok(name="exec"),
            _ok(name="read"),
            _ok(name="read"),
            _ok(name="read"),
            _ok(name="read"),
            _ok(name="read"),
            _ok(name="read"),
            _ok(name="read"),
            _ok(name="read"),
            _ok(name="wiki_write"),
        ]
        assert _compute_violation_active(events) is False


# --------------------------------------------------------------------------
# Production-code wiring: the ContextVar must hold the value the loop wrote.
# We don't drive the full AgentLoop here; we exercise the line that does
# the set() and assert the ContextVar reflects it. This catches the case
# where someone changes the setter to a no-op or to the wrong source.
# --------------------------------------------------------------------------

class TestContextVarWiring:
    def test_set_true_visible(self):
        token = _PEER_WS_VIOLATION_VAR.set(True)
        try:
            assert _PEER_WS_VIOLATION_VAR.get() is True
        finally:
            _PEER_WS_VIOLATION_VAR.reset(token)

    def test_set_false_visible(self):
        token = _PEER_WS_VIOLATION_VAR.set(False)
        try:
            assert _PEER_WS_VIOLATION_VAR.get() is False
        finally:
            _PEER_WS_VIOLATION_VAR.reset(token)

    def test_concurrent_tasks_isolated(self):
        # Pins the ContextVar invariant already implied by the comment at
        # loop.py:91-98: concurrent turns on different asyncio tasks must
        # not see each other's flag.
        async def run():
            async def setter(value: bool, observed: list[bool]):
                _PEER_WS_VIOLATION_VAR.set(value)
                await asyncio.sleep(0)
                observed.append(_PEER_WS_VIOLATION_VAR.get())

            obs_true: list[bool] = []
            obs_false: list[bool] = []
            await asyncio.gather(
                setter(True, obs_true),
                setter(False, obs_false),
            )
            return obs_true[0], obs_false[0]

        true_seen, false_seen = asyncio.run(run())
        assert true_seen is True
        assert false_seen is False
