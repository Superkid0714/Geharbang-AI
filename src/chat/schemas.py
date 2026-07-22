from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ChatDomain(StrEnum):
    GUESTHOUSE = "guesthouse"
    STAFF_STEP = "staff_step"
    JEJU_TRAVEL = "jeju_travel"
    GEHARBANG_SERVICE = "geharbang_service"
    GREETING = "greeting"
    UNCLEAR = "unclear"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True)
class RouteDecision:
    domain: ChatDomain
    confidence: float
    reason: str


@dataclass
class ConversationState:
    """Server-side state needed to understand short follow-up questions."""

    active_domain: ChatDomain | None = None
    domain_states: dict[ChatDomain, Any] = field(default_factory=dict)


def serialize_conversation_state(state: ConversationState) -> dict[str, Any]:
    """Convert in-process state to a JSON-safe payload for durable storage."""
    return {
        "activeDomain": state.active_domain.value if state.active_domain else None,
        "domainStates": {
            domain.value: value for domain, value in state.domain_states.items()
        },
    }


def deserialize_conversation_state(payload: dict[str, Any] | None) -> ConversationState:
    """Restore saved state while ignoring fields from malformed/older payloads."""
    if not isinstance(payload, dict):
        return ConversationState()

    active_domain = None
    raw_active_domain = payload.get("activeDomain")
    if isinstance(raw_active_domain, str):
        try:
            active_domain = ChatDomain(raw_active_domain)
        except ValueError:
            active_domain = None

    domain_states: dict[ChatDomain, Any] = {}
    raw_domain_states = payload.get("domainStates")
    if isinstance(raw_domain_states, dict):
        for raw_domain, value in raw_domain_states.items():
            try:
                domain_states[ChatDomain(raw_domain)] = value
            except (TypeError, ValueError):
                continue

    return ConversationState(
        active_domain=active_domain,
        domain_states=domain_states,
    )


@dataclass(frozen=True)
class ChatResponse:
    answer: str
    domain: ChatDomain
    confidence: float
