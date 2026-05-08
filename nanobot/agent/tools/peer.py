"""Inter-agent peer messaging tools.

Two tools that surface the :class:`nanobot.channels.peer.PeerChannel` to the
LLM:

* :class:`PeerSayTool` (``peer_say``) — send a message (with optional file
  attachments) to a named peer agent. The peer receives it as a normal chat
  turn on chat_id ``peer:<sender>`` and responds with their own persona.
* :class:`PeerListTool` (``peer_list``) — return the roster of peers
  currently online, as last reported by the router via presence push.

Wiring: both tools are constructed by the gateway bootstrap with callbacks
that bind to the running PeerChannel instance, then registered into the
agent's :class:`ToolRegistry`. Registration happens only when the ``peer``
channel is enabled in config.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Awaitable

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import (
    ArraySchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.bus.events import OutboundMessage

PEER_CHANNEL_NAME = "peer"
PEER_CHAT_PREFIX = "peer:"


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
            StringSchema("File path"),
            description=(
                "Optional list of local file paths to send along with the "
                "message. Files are uploaded once to the mailbox blob store "
                "and the recipient downloads them on receipt; both sides see "
                "them as ordinary media attachments."
            ),
        ),
        required=["to", "text"],
    )
)
class PeerSayTool(Tool):
    """Send a message to another nanobot peer agent.

    The peer receives the message on chat_id ``peer:<sender>`` and answers
    with their own persona; the conversation accumulates as a long-running
    DM thread, exactly like a Telegram private chat between two people.
    """

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]],
        own_agent_id: str,
        peer_validator: Callable[[str], bool] | None = None,
    ):
        self._send_callback = send_callback
        self._own_agent_id = own_agent_id
        self._peer_validator = peer_validator

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

        msg = OutboundMessage(
            channel=PEER_CHANNEL_NAME,
            chat_id=f"{PEER_CHAT_PREFIX}{to}",
            content=text,
            media=list(media) if media else [],
            metadata=meta,
        )
        try:
            await self._send_callback(msg)
        except Exception as exc:  # noqa: BLE001
            return f"Error sending peer message: {exc}"
        media_info = f", {len(media)} attachment(s)" if media else ""
        return f"Sent to peer {to}{media_info}"


@tool_parameters(tool_parameters_schema())
class PeerListTool(Tool):
    """Return the roster of peers currently online.

    Backed by presence pushes from the router: every time a peer connects or
    disconnects, the router broadcasts the new online roster to all live
    connections. This tool returns the most recent snapshot.
    """

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
