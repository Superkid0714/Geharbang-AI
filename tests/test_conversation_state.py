from src.chat.schemas import (
    ChatDomain,
    ConversationState,
    deserialize_conversation_state,
    serialize_conversation_state,
)


def test_conversation_state_round_trip() -> None:
    state = ConversationState(
        active_domain=ChatDomain.GUESTHOUSE,
        domain_states={
            ChatDomain.GUESTHOUSE: {
                "guestHouseId": 17,
                "guestHouseName": "테스트 게스트하우스",
            }
        },
    )

    restored = deserialize_conversation_state(serialize_conversation_state(state))

    assert restored.active_domain == ChatDomain.GUESTHOUSE
    assert restored.domain_states[ChatDomain.GUESTHOUSE]["guestHouseId"] == 17


def test_conversation_state_ignores_unknown_domains() -> None:
    restored = deserialize_conversation_state(
        {
            "activeDomain": "removed-domain",
            "domainStates": {
                "removed-domain": {"value": 1},
                "staff_step": {"staffRecruitmentId": 3},
            },
        }
    )

    assert restored.active_domain is None
    assert restored.domain_states == {
        ChatDomain.STAFF_STEP: {"staffRecruitmentId": 3}
    }
