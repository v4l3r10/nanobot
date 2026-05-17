"""Message envelope models, validators, ULID generation."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, Field, StringConstraints
from ulid import ULID

PROTOCOL_VERSION = 1
SUBJECT_MAX = 200
BODY_MAX_BYTES = 16384

# Agent ids are arbitrary lowercase tokens. The actual roster of agents is
# learned dynamically at startup from MAILBOX_AGENT_TOKENS — this regex only
# constrains the format so values are safe in URLs, paths, and log fields.
AGENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

AgentId = str
MessageType = Literal["notification", "request", "response", "broadcast"]
Priority = Literal["low", "normal", "high"]


def is_valid_agent_id(value: str) -> bool:
    return isinstance(value, str) and AGENT_ID_PATTERN.match(value) is not None


def new_message_id() -> str:
    return f"msg_{ULID()}"


def new_thread_id() -> str:
    return f"thr_{ULID()}"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class AttachmentMeta(BaseModel):
    id: int
    name: str
    mime: str
    size_bytes: int


class Envelope(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    id: str
    thread_id: str
    in_reply_to: str | None = None
    from_agent: AgentId = Field(alias="from")
    to_agent: AgentId = Field(alias="to")
    type: MessageType
    subject: Annotated[str, StringConstraints(max_length=SUBJECT_MAX)]
    body: str
    priority: Priority = "normal"
    attachments: list[AttachmentMeta] = Field(default_factory=list)
    created_at: str
    read_at: str | None = None

    model_config = {"populate_by_name": True}


class InlineAttachmentInput(BaseModel):
    name: Annotated[str, StringConstraints(min_length=1, max_length=255)] = Field(
        validation_alias=AliasChoices("name", "filename"),
    )
    mime: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    content_b64: str

    model_config = {"populate_by_name": True}
