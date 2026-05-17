"""Inter-agent peer messaging tools.

Three tools that surface the :class:`nanobot.channels.peer.PeerChannel` to
the LLM:

* :class:`PeerSayTool` (``peer_say``) — send a message (with optional file
  attachments) to a named peer agent. The peer receives it as a normal chat
  turn on chat_id ``<sender>`` (i.e. just the sender's agent_id; the
  ``channel="peer"`` field on the bus is what disambiguates it from chats
  on other channels) and responds with their own persona.
* :class:`PeerListTool` (``peer_list``) — return the roster of peers
  currently online, as last reported by the router via presence push.
* :class:`PeerThreadShowTool` (``peer_thread_show``) — read back the most
  recent N messages exchanged with a specific peer. Lets the agent recap
  "what did I say to / hear from <peer> recently?" without parsing the raw
  session jsonl.

Wiring: all three tools are constructed by the gateway bootstrap with
callbacks that bind to the running PeerChannel instance, then registered
into the agent's :class:`ToolRegistry`. Registration happens only when the
``peer`` channel is enabled in config.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, Awaitable

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import (
    ArraySchema,
    BooleanSchema,
    IntegerSchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.bus.events import OutboundMessage

PEER_CHANNEL_NAME = "peer"


@tool_parameters(
    tool_parameters_schema(
        to=StringSchema(
            "Agent id of the recipient peer (e.g. 'grocco', 'naldo'). "
            "Use peer_list to discover available peers; never hardcode."
        ),
        text=StringSchema("Plain-text message body to send to the peer."),
        thread_id=StringSchema(
            "Optional thread id to continue an existing conversation. "
            "Omit to start a new thread."
        ),
        in_reply_to=StringSchema(
            "Optional message id this peer_say is replying to."
        ),
        media=ArraySchema(
            StringSchema("Absolute file path"),
            description=(
                "Optional list of ABSOLUTE local file paths to send along "
                "with the message (e.g. '/home/nanobot/.nanobot/workspace/"
                "progetti/foo.zip'). Relative paths are rejected — the peer "
                "channel does not assume any working directory. Files are "
                "uploaded once to the mailbox blob store and the recipient "
                "downloads them on receipt; both sides see them as ordinary "
                "media attachments."
            ),
        ),
        closing=BooleanSchema(
            description=(
                "Set true ONLY for purely conversational closure ('grazie, "
                "alla prossima', a wave emoji) where you expect no reply and "
                "the recipient has nothing left to act on. The recipient's "
                "agent loop is NOT awakened on a closing message — this is "
                "the system-level mechanism that breaks pleasantry loops. "
                "Do NOT use closing=true when sending instructions, files, "
                "or any payload that requires the peer to take action: the "
                "agent will never see it. Default: false."
            ),
        ),
        required=["to", "text"],
    )
)
class PeerSayTool(Tool):
    """Send a message to another nanobot peer agent.

    The peer receives the message on chat_id ``<sender>`` and answers
    with their own persona; the conversation accumulates as a long-running
    DM thread, exactly like a Telegram private chat between two people.
    """

    # Manually wired in cli.commands with the live PeerChannel's callbacks;
    # opt out of ToolLoader auto-discovery. Without this the loader calls
    # cls() with no args -> TypeError (caught, but logged at every startup);
    # that noise once masked a real signal and triggered a false-alarm
    # rollback. The tool is still fully registered via the manual glue.
    _plugin_discoverable = False

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]],
        own_agent_id: str,
        peer_validator: Callable[[str], bool] | None = None,
    ):
        self._send_callback = send_callback
        self._own_agent_id = own_agent_id
        self._peer_validator = peer_validator
        # Per-turn flag: True after a successful peer_say in the current turn,
        # mirrors MessageTool._sent_in_turn so the agent loop can suppress its
        # own final_content emission (otherwise the LLM's narrative wrap-up
        # — e.g. "Fatto. Messaggio inviato e chiuso." — is forwarded as a
        # second peer message without closing=true, defeating the closing
        # protocol the bot just used).
        self._sent_in_turn_var: ContextVar[bool] = ContextVar(
            "peer_say_sent_in_turn", default=False,
        )

    def start_turn(self) -> None:
        """Reset per-turn tracking. Called by the agent loop at the start
        of every dispatch."""
        self._sent_in_turn_var.set(False)

    @property
    def _sent_in_turn(self) -> bool:
        return self._sent_in_turn_var.get()

    @property
    def name(self) -> str:
        return "peer_say"

    @property
    def description(self) -> str:
        return (
            "Initiate or continue a 1:1 conversation with another peer "
            "nanobot agent. The recipient receives your message as a chat "
            "turn and replies with their own persona; the dialogue is "
            "persistent across turns. Use peer_list first to find available "
            "peers — never hardcode names. Attach files via the 'media' "
            "parameter (any size; uploaded transparently)."
        )

    async def execute(
        self,
        to: str,
        text: str,
        thread_id: str | None = None,
        in_reply_to: str | None = None,
        media: list[str] | None = None,
        closing: bool = False,
        **_kwargs: Any,
    ) -> str:
        to = (to or "").strip()
        if not to:
            return "Error: 'to' is required"
        if to == self._own_agent_id:
            return "Error: cannot peer_say to self"
        if self._peer_validator is not None and not self._peer_validator(to):
            return (
                f"Error: invalid peer agent id {to!r}. "
                "Agent ids must match ^[a-z][a-z0-9_-]{0,31}$."
            )
        if not isinstance(text, str) or not text:
            return "Error: 'text' is required"

        meta: dict[str, Any] = {}
        if thread_id:
            meta["peer_thread_id"] = thread_id
        if in_reply_to:
            meta["peer_in_reply_to"] = in_reply_to
        if closing:
            meta["peer_closing"] = True

        msg = OutboundMessage(
            channel=PEER_CHANNEL_NAME,
            chat_id=to,
            content=text,
            media=list(media) if media else [],
            metadata=meta,
        )
        try:
            await self._send_callback(msg)
        except Exception as exc:  # noqa: BLE001
            return f"Error sending peer message: {exc}"
        # Mark turn so the agent loop suppresses its narrative final_content
        # — see PeerSayTool docstring and loop.py post-turn dispatch.
        self._sent_in_turn_var.set(True)
        media_info = f", {len(media)} attachment(s)" if media else ""
        closing_info = " [closing]" if closing else ""
        return f"Sent to peer {to}{media_info}{closing_info}"


@tool_parameters(tool_parameters_schema())
class PeerListTool(Tool):
    """Return the roster of peers currently online.

    Backed by presence pushes from the router: every time a peer connects or
    disconnects, the router broadcasts the new online roster to all live
    connections. This tool returns the most recent snapshot.
    """

    # See PeerSayTool: manually wired, opt out of ToolLoader auto-discovery.
    _plugin_discoverable = False

    def __init__(
        self,
        list_peers_callback: Callable[[], list[str]],
        own_agent_id: str,
    ):
        self._list_peers_callback = list_peers_callback
        self._own_agent_id = own_agent_id

    @property
    def name(self) -> str:
        return "peer_list"

    @property
    def description(self) -> str:
        return (
            "List peer agents currently online (reachable via the inter-agent "
            "channel). Returns a JSON array of objects {id, online}. "
            "'online: false' is not currently emitted — peers absent from the "
            "list are simply not connected, but you can still call peer_say "
            "to them: the router queues messages and delivers on reconnect."
        )

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **_kwargs: Any) -> str:
        peers = self._list_peers_callback() or []
        payload = [
            {"id": p, "online": True}
            for p in peers
            if p != self._own_agent_id
        ]
        return json.dumps(payload, ensure_ascii=False)


def _format_thread(messages: list[dict[str, Any]], own_agent_id: str) -> str:
    """Render a list of message dicts as a human-readable transcript.

    The format favours LLM readability over machine parseability: short
    timestamps (HH:MM:SS), speaker prefix, body text. Reverse-chronological
    or chronological order is decided by the caller; we just iterate.
    """
    lines: list[str] = []
    for msg in messages:
        ts = (msg.get("ts") or "")[11:19]  # HH:MM:SS slice from ISO 8601
        sender = msg.get("from") or "?"
        recipient = msg.get("to") or "?"
        prefix = (
            f"tu → {recipient}" if sender == own_agent_id
            else f"{sender} → tu"
        )
        text = (msg.get("text") or "").strip()
        if not text:
            text = "(messaggio vuoto)"
        lines.append(f"[{ts}] {prefix}: {text}")
    return "\n".join(lines)


@tool_parameters(
    tool_parameters_schema(
        peer=StringSchema(
            "Agent id del peer di cui leggere lo storico (es. 'grocco')."
        ),
        last_n=IntegerSchema(
            description=(
                "Numero massimo di messaggi recenti da restituire "
                "(default 20, max 100)."
            ),
            minimum=1,
            maximum=100,
        ),
        required=["peer"],
    )
)
class PeerThreadShowTool(Tool):
    """Show the recent thread between this agent and a named peer.

    Reads from the mailbox router's authoritative DB (not from the local
    session jsonl), so it survives container restart and is the
    single-source-of-truth for "what did I just say to / hear from
    <peer>?" questions. Returns a chronologically-ordered transcript ready
    to be pasted into an answer to the user.
    """

    # See PeerSayTool: manually wired, opt out of ToolLoader auto-discovery.
    _plugin_discoverable = False

    def __init__(
        self,
        fetch_thread_callback: Callable[..., Awaitable[list[dict[str, Any]]]],
        own_agent_id: str,
        peer_validator: Callable[[str], bool] | None = None,
    ):
        self._fetch_thread_callback = fetch_thread_callback
        self._own_agent_id = own_agent_id
        self._peer_validator = peer_validator

    @property
    def name(self) -> str:
        return "peer_thread_show"

    @property
    def description(self) -> str:
        return (
            "Read back the most recent messages exchanged with a peer agent. "
            "Use this when the user asks what a peer said, or when you need "
            "to recall context before continuing a peer conversation. "
            "Returns a clean chronological transcript (oldest→newest) sourced "
            "from the mailbox router's authoritative DB, not the local "
            "session — works even after a restart."
        )

    @property
    def read_only(self) -> bool:
        return True

    async def execute(
        self, peer: str, last_n: int = 20, **_kwargs: Any
    ) -> str:
        peer = (peer or "").strip().lower()
        if not peer:
            return "Error: 'peer' is required"
        if peer == self._own_agent_id:
            return "Error: peer must differ from yourself"
        if self._peer_validator is not None and not self._peer_validator(peer):
            return (
                f"Error: invalid peer agent id {peer!r}. "
                "Agent ids must match ^[a-z][a-z0-9_-]{0,31}$."
            )
        try:
            last_n = max(1, min(int(last_n), 100))
        except (TypeError, ValueError):
            last_n = 20
        try:
            messages = await self._fetch_thread_callback(peer, last_n=last_n)
        except Exception as exc:  # noqa: BLE001
            return f"Error fetching thread with {peer}: {exc}"
        if not messages:
            return f"Nessun messaggio scambiato con {peer} (DB del router vuoto per questa coppia)."
        transcript = _format_thread(messages, self._own_agent_id)
        header = (
            f"Conversazione con {peer} "
            f"(ultimi {len(messages)} messaggi, oldest→newest):\n"
        )
        return header + transcript
