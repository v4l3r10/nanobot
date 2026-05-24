"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
from contextlib import suppress
from pathlib import Path
from typing import Any, Mapping, Sequence

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader
from nanobot.agent.tools import mcp as mcp_tools
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.wiki.paths import vault_dir
from nanobot.agent.wiki.vault import Vault
from nanobot.apps.cli import utils as cli_app_utils
from nanobot.bus.events import InboundMessage
from nanobot.session.goal_state import goal_state_runtime_lines
from nanobot.utils.helpers import (
    current_time_str,
    detect_image_mime,
    load_bundled_template,
    truncate_text,
)
from nanobot.utils.prompt_templates import (
    is_bundled_template_content,
    render_template,
)


def session_extra(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return persisted kwargs for turn-attached capabilities."""
    return cli_app_utils.session_extra(metadata) | mcp_tools.session_extra(metadata)


def runtime_lines(state: Any, msg: Any, workspace: Path, *, skip: bool = False) -> list[str]:
    """Return model-visible runtime annotations for turn-attached capabilities."""
    return [
        *cli_app_utils.runtime_lines(msg, workspace, skip=skip),
        *mcp_tools.runtime_lines(
            msg,
            configured_server_names=set(state._mcp_servers),
            connected_server_names=set(state._mcp_stacks),
            skip=skip,
        ),
    ]


async def connect_mcp(state: Any, tools: ToolRegistry) -> None:
    await mcp_tools.connect_missing_servers(state, tools)


async def handle_runtime_control(state: Any, msg: InboundMessage, tools: ToolRegistry) -> bool:
    return await mcp_tools.handle_runtime_control(state, msg, tools)


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"
    _MAX_RECENT_HISTORY = 50
    # When the wiki is enabled, durable knowledge lives in the per-user vault
    # (navigable via the wiki_note tool); the replayed history tail is just a
    # short recency window, so it is capped much smaller. _MAX_HISTORY_CHARS
    # and the wiki-off _MAX_RECENT_HISTORY=50 are deliberately unchanged.
    _MAX_RECENT_HISTORY_WIKI = 10
    _MAX_HISTORY_CHARS = 32_000  # hard cap on recent history section size
    # Hard cap on the per-user vault MOC / USER.md injected on the wiki-on hot
    # path (every turn). Same 32_000 value as _MAX_HISTORY_CHARS — both bound a
    # single durable-knowledge section that Lint normally keeps small, so one
    # shared ceiling is the least surprising choice and keeps the bound
    # uniform; a separate name documents intent and lets the two diverge later
    # without touching call sites. This is what makes I1's token probe
    # guaranteed-conservative: the probe and the real turn read the same
    # bounded text. The wiki-OFF path (global get_memory_context) is unchanged.
    _MAX_MEMORY_CHARS = 32_000
    _RUNTIME_CONTEXT_END = "[/Runtime Context]"
    # Hard cap on the per-interlocutor "Sender:" line injected in the runtime
    # tail (Layer 2). Bounds a single summary so a bloated people-page
    # frontmatter can't grow the per-message tail.
    _SENDER_CARD_MAX = 200

    def __init__(
        self,
        workspace: Path,
        timezone: str | None = None,
        disabled_skills: list[str] | None = None,
        wiki_enabled: bool = False,
    ):
        self.workspace = workspace
        self.timezone = timezone
        # Master gate for the wiki-tree memory read path (Task 3.1 knob,
        # plumbed from config.agents.defaults.dream.wiki_enabled). Default
        # False so the system prompt is byte-identical to a stock v0.2.0
        # install — the single most safety-critical invariant.
        self.wiki_enabled = wiki_enabled
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace, disabled_skills=set(disabled_skills) if disabled_skills else None)

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        channel: str | None = None,
        session_summary: str | None = None,
        workspace: Path | None = None,
        session_key: str | None = None,
        memory_key: str | None = None,
    ) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills.

        ``session_key`` is the effective session key, passed explicitly by the
        caller (the plan's chosen approach — an explicit argument rather than a
        ContextVar — so the per-user vault is resolved purely from arguments and
        every call site stays auditable). It is only consulted when
        ``self.wiki_enabled`` is True. When the wiki is off, or no
        ``session_key`` is available, the original (pre-6.1) code path runs
        verbatim so the prompt is byte-identical to a stock install.

        ``memory_key`` (CV2): explicit vault-key for wiki reads. When provided,
        overrides ``session_key`` for vault resolution. When ``None``, falls
        back to ``session_key`` (pre-CV2 behaviour). Used by callers that
        want per-user chat sessions with a shared memory vault
        (``unified_memory=true``).
        """
        root = workspace or self.workspace
        # CV2: vault_key resolution — memory_key wins, session_key fallback.
        # Pre-CV2 callers (memory_key=None) keep byte-identical behaviour.
        vault_key = memory_key or session_key
        # Single source of truth for "the wiki read path is fully activated
        # for this call" (M3). The read block below and every wiki branch are
        # gated on this ONE predicate so they can never diverge: a falsy but
        # non-None key (e.g. "") would otherwise skip the read yet take the
        # wiki branch, leaving the user with NEITHER vault memory/profile NOR
        # the global fallback. bool(vault_key) is False for both None and
        # "" -> safe wiki-off fallback in both cases. The wiki-off path stays
        # byte-identical (this predicate only ever suppresses the wiki block).
        wiki_active = self.wiki_enabled and bool(vault_key)

        # Resolve the per-user vault MOC iff the wiki read path is fully
        # activated for this call. Anything missing -> wiki_moc stays None and
        # every branch below falls back to the verbatim original behaviour.
        wiki_moc: str | None = None
        vault_user: str | None = None
        if wiki_active:
            vroot = vault_dir(root, vault_key)
            with suppress(OSError):
                moc_path = vroot / "MEMORY.md"
                if moc_path.is_file():
                    # Intentionally lock-free and torn-read-safe: Lint ALWAYS
                    # rewrites this MOC via atomic_write_text (tmp file +
                    # os.replace), never an in-place open(...,'w'), so a
                    # concurrent Lint can only swap the whole file, never
                    # expose a partial write. Do NOT change Lint to write the
                    # MOC non-atomically. (M2)
                    text = moc_path.read_text(encoding="utf-8")
                    if text.strip():
                        # Deterministic hot-path bound (M1): a corrupted /
                        # pre-Lint / hand-edited MOC must not blow up the
                        # prompt every turn. Same truncate_text helper the
                        # history path uses.
                        wiki_moc = truncate_text(text, self._MAX_MEMORY_CHARS)
            with suppress(OSError):
                user_path = vroot / "USER.md"
                if user_path.is_file():
                    vault_user = truncate_text(
                        user_path.read_text(encoding="utf-8"),
                        self._MAX_MEMORY_CHARS,
                    )

        parts = [self._get_identity(channel=channel, workspace=root)]

        # USER.md de-duplication: when the wiki read path is active the vault
        # owns the user profile, so the global workspace-root USER.md must not
        # also be injected by the bootstrap block (no double / stale profile).
        # SOUL.md / AGENTS.md / TOOLS.md handling is untouched.
        if wiki_active:
            bootstrap = self._load_bootstrap_files(root, skip={"USER.md"})
            if bootstrap:
                parts.append(bootstrap)
            if vault_user and vault_user.strip():
                parts.append(f"## USER.md\n\n{vault_user}")
        else:
            bootstrap = self._load_bootstrap_files(root)
            if bootstrap:
                parts.append(bootstrap)

        parts.append(render_template("agent/tool_contract.md"))

        if wiki_active:
            # Wiki source of truth: inject the Lint-regenerated per-user MOC
            # (same wrapper/heading; only the content source changes). If the
            # vault is empty / the MOC is absent or blank, skip the section
            # entirely — do NOT fall back to the global MEMORY.md.
            if wiki_moc and wiki_moc.strip():
                parts.append(f"# Memory\n\n{wiki_moc}")
        else:
            memory = self.memory.get_memory_context()
            if memory and not self._is_template_content(self.memory.read_memory(), "memory/MEMORY.md"):
                parts.append(f"# Memory\n\n{memory}")

        always_skills = self.skills.get_always_skills()
        if always_skills:
            if wiki_active and "memory" in always_skills:
                # Wiki active: the legacy `memory` skill (MEMORY.md / grep
                # history.jsonl) is stale and misleading — the per-user vault
                # MOC + wiki_note are the real mechanism. Substitute ONLY the
                # `memory` body with the wiki-aware guidance, preserving the
                # exact `### Skill: <name>` / `\n\n---\n\n` shape
                # load_skills_for_context produces and the original ordering
                # (each skill emitted in get_always_skills() order, the wiki
                # body in `memory`'s slot). Other always-skills are loaded
                # verbatim. This branch is unreachable when wiki is off, so
                # the wiki-OFF system prompt is byte-identical to before.
                wiki_mem = render_template("agent/memory_skill_wiki.md").strip()
                rendered = []
                # `### Skill: <name>` wrapper + `\n\n---\n\n` join below MUST mirror the source of
                # truth nanobot/agent/skills.py:load_skills_for_context (L104-109); keep in sync.
                for name in always_skills:
                    if name == "memory":
                        rendered.append(f"### Skill: memory\n\n{wiki_mem}")
                    else:
                        body = self.skills.load_skills_for_context([name])
                        if body:
                            rendered.append(body)
                always_content = "\n\n---\n\n".join(p for p in rendered if p)
            else:
                always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        skills_summary = self.skills.build_skills_summary(exclude=set(always_skills))
        if skills_summary:
            parts.append(render_template("agent/skills_section.md", skills_summary=skills_summary))

        entries = self.memory.read_unprocessed_history(since_cursor=self.memory.get_last_dream_cursor())
        if entries:
            history_cap = (
                self._MAX_RECENT_HISTORY_WIKI if wiki_active else self._MAX_RECENT_HISTORY
            )
            capped = entries[-history_cap:]
            history_text = "\n".join(
                f"- [{e['timestamp']}] {e['content']}" for e in capped
            )
            history_text = truncate_text(history_text, self._MAX_HISTORY_CHARS)
            parts.append("# Recent History\n\n" + history_text)

        if session_summary:
            parts.append(f"[Archived Context Summary]\n\n{session_summary}")

        return "\n\n---\n\n".join(parts)

    def _get_identity(self, channel: str | None = None, workspace: Path | None = None) -> str:
        """Get the core identity section."""
        root = workspace or self.workspace
        workspace_path = str(root.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return render_template(
            "agent/identity.md",
            workspace_path=workspace_path,
            runtime=runtime,
            platform_policy=render_template("agent/platform_policy.md", system=system),
            channel=channel or "",
        )

    @staticmethod
    def _build_runtime_context(
        channel: str | None,
        chat_id: str | None,
        timezone: str | None = None,
        sender_id: str | None = None,
        supplemental_lines: Sequence[str] | None = None,
    ) -> str:
        """Build untrusted runtime metadata block appended after user content."""
        lines = [f"Current Time: {current_time_str(timezone)}"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        if sender_id:
            lines += [f"Sender ID: {sender_id}"]
        if supplemental_lines:
            lines.extend(supplemental_lines)
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines) + "\n" + ContextBuilder._RUNTIME_CONTEXT_END

    def resolve_sender_card(
        self, sender_id: str | None, channel: str | None, vault_key: str | None,
    ) -> str | None:
        """Resolve the current interlocutor to a people-page one-liner.

        Returns ``"<title> (<summary>)"`` (bounded) for the first hot people
        page whose ``sender_ids`` frontmatter contains a channel-qualified
        match for this sender, else ``None``. Matches the full ``sender_id``
        first, then its numeric prefix before ``|`` (telegram's id is
        ``"<numeric>|<username>"``, so a username change does not break the
        binding). Wiki-gated AND data-gated: with the wiki off, no vault key,
        or no binding, returns ``None`` so the runtime tail is byte-identical
        (opt-in by data presence, no new flag). Hot path — never raises: a
        parse/IO failure for any page is suppressed and yields ``None``.
        """
        if not (self.wiki_enabled and vault_key and sender_id and channel):
            return None
        candidates = {
            f"{channel}:{sender_id}",
            f"{channel}:{sender_id.split('|', 1)[0]}",
        }
        with suppress(Exception):
            for _rel, page in Vault(vault_dir(self.workspace, vault_key)).iter_pages():
                if page.type != "people" or not page.sender_ids or not page.summary:
                    continue
                if candidates.intersection(page.sender_ids):
                    return truncate_text(
                        f"{page.title} ({page.summary})", self._SENDER_CARD_MAX,
                    )
        return None

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            return f"{left}\n\n{right}" if left else right

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [item if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in value]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)

    def _load_bootstrap_files(
        self,
        workspace: Path | None = None,
        skip: set[str] | None = None,
    ) -> str:
        """Load all bootstrap files from workspace.

        ``skip`` (default None) names bootstrap files to omit; used by the
        wiki read path to suppress the global USER.md (the vault owns it).
        With ``skip`` None/empty the iteration is the verbatim original, so
        the wiki-off prompt is byte-identical.
        """
        parts = []
        root = workspace or self.workspace

        for filename in self.BOOTSTRAP_FILES:
            if skip and filename in skip:
                continue
            file_path = root / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _is_template_content(content: str, template_path: str) -> bool:
        """Check if *content* is identical to the bundled template (user hasn't customized it).

        Delegates to the shared leaf helper so this check and the Task 7.1
        legacy-migration template guard can never diverge.
        """
        return is_bundled_template_content(content, template_path)

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        current_role: str = "user",
        sender_id: str | None = None,
        session_summary: str | None = None,
        session_metadata: Mapping[str, Any] | None = None,
        current_runtime_lines: Sequence[str] | None = None,
        workspace: Path | None = None,
        runtime_state: Any | None = None,
        inbound_message: Any | None = None,
        skip_runtime_lines: bool = False,
        session_key: str | None = None,
        memory_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call.

        ``session_key`` is threaded straight to :meth:`build_system_prompt`
        so the wiki read path can resolve the caller's per-user vault. None
        (the default, and the consolidator token-probe case) keeps the
        verbatim wiki-off behaviour.

        ``memory_key`` (CV2): when provided, overrides ``session_key`` for
        vault resolution inside :meth:`build_system_prompt`. Threaded
        unchanged; back-compat by default (``None`` → use session_key).
        """
        root = workspace or self.workspace
        extra = list(goal_state_runtime_lines(session_metadata) or [])
        if runtime_state is not None and inbound_message is not None:
            extra.extend(runtime_lines(runtime_state, inbound_message, root, skip=skip_runtime_lines))
        if current_runtime_lines:
            extra.extend(line for line in current_runtime_lines if line)
        # Layer 2: per-interlocutor sender card. Resolved from people-page
        # frontmatter and appended to the VOLATILE runtime tail (never the
        # cacheable system-prompt prefix), so it is group-safe and adds zero
        # cache cost. None -> no line -> tail byte-identical (opt-in by data).
        card = self.resolve_sender_card(sender_id, channel, memory_key or session_key)
        if card:
            extra.append(f"Sender: {card}")
        runtime_ctx = self._build_runtime_context(
            channel,
            chat_id,
            self.timezone,
            sender_id=sender_id,
            supplemental_lines=extra or None,
        )
        user_content = self._build_user_content(current_message, media)

        # Merge runtime context and user content into a single user message
        # to avoid consecutive same-role messages that some providers reject.
        # Runtime context is appended to keep the user-content prefix stable
        # for prompt-cache hits (the context changes every turn due to time).
        if isinstance(user_content, str):
            merged = f"{user_content}\n\n{runtime_ctx}"
        else:
            merged = user_content + [{"type": "text", "text": runtime_ctx}]
        messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(
                    skill_names,
                    channel=channel,
                    session_summary=session_summary,
                    workspace=root,
                    session_key=session_key,
                    memory_key=memory_key,
                ),
            },
            *history,
        ]
        if messages[-1].get("role") == current_role:
            last = dict(messages[-1])
            last["content"] = self._merge_message_content(last.get("content"), merged)
            messages[-1] = last
            return messages
        messages.append({"role": current_role, "content": merged})
        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            if not p.is_file():
                continue
            raw = p.read_bytes()
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(raw).decode()
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
                "_meta": {"path": str(p)},
            })

        if not images:
            return text
        return images + [{"type": "text", "text": text}]
