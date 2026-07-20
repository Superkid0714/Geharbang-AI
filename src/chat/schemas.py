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


@dataclass(frozen=True)
class ChatResponse:
    answer: str
    domain: ChatDomain
    confidence: float
