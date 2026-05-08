"""Bearer token to agent identity mapping. The X-Agent-Id header is ignored;
identity is derived from the token alone."""

from __future__ import annotations

import hmac
import json
import os
from dataclasses import dataclass

from .protocol import is_valid_agent_id


@dataclass(frozen=True)
class AuthContext:
    agent_id: str


class TokenRegistry:
    def __init__(self, token_to_agent: dict[str, str]):
        self._pairs: list[tuple[str, str]] = list(token_to_agent.items())
        self._agents: frozenset[str] = frozenset(token_to_agent.values())

    @classmethod
    def from_env(cls, env_var: str = "MAILBOX_AGENT_TOKENS") -> "TokenRegistry":
        raw = os.environ.get(env_var)
        if not raw:
            raise RuntimeError(f"{env_var} not set")
        agents_to_tokens = json.loads(raw)
        if not isinstance(agents_to_tokens, dict):
            raise RuntimeError(f"{env_var} must be a JSON object")
        empty = [a for a, t in agents_to_tokens.items() if not t]
        if empty:
            raise RuntimeError(f"empty token for agents: {empty}")
        invalid = [a for a in agents_to_tokens if not is_valid_agent_id(a)]
        if invalid:
            raise RuntimeError(
                f"invalid agent ids in {env_var}: {invalid} "
                f"(must match ^[a-z][a-z0-9_-]{{0,31}}$)"
            )
        return cls({token: agent for agent, token in agents_to_tokens.items()})

    @property
    def known_agents(self) -> frozenset[str]:
        return self._agents

    def resolve(self, bearer: str | None) -> AuthContext | None:
        if not bearer:
            return None
        for token, agent_id in self._pairs:
            if hmac.compare_digest(bearer, token):
                return AuthContext(agent_id=agent_id)
        return None


def extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None
