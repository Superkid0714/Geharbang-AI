from __future__ import annotations

from src.chat.router import route_chat_query
from src.chat.schemas import ChatDomain, ChatResponse, ConversationState
from src.guesthouses.recommender import answer_guesthouse_chat
from src.guesthouses.vector_store import has_vector_store_documents as has_guesthouse_index
from src.staff_steps.recommender import answer_staff_chat
from src.staff_steps.vector_store import has_vector_store_documents as has_staff_index
from src.service_guide.recommender import answer_service_guide
from src.service_guide.vector_store import has_vector_store_documents as has_service_guide_index


PENDING_DOMAIN_MESSAGES = {
    ChatDomain.JEJU_TRAVEL: "제주 관광 정보 기능은 아직 준비 중입니다.",
}


def answer_chat(
    query: str,
    state: ConversationState | None = None,
) -> tuple[ChatResponse, ConversationState]:
    current_state = state or ConversationState()
    decision = route_chat_query(query, active_domain=current_state.active_domain)

    if decision.domain == ChatDomain.GUESTHOUSE:
        if not has_guesthouse_index():
            current_state.active_domain = ChatDomain.GUESTHOUSE
            answer = "게스트하우스 검색 인덱스가 아직 준비되지 않았습니다. 관리자에게 인덱스 갱신을 요청해 주세요."
            return ChatResponse(answer, decision.domain, decision.confidence), current_state
        previous_result = current_state.domain_states.get(ChatDomain.GUESTHOUSE)
        answer, next_result = answer_guesthouse_chat(query, previous_result=previous_result)
        current_state.active_domain = ChatDomain.GUESTHOUSE
        if next_result is not None:
            current_state.domain_states[ChatDomain.GUESTHOUSE] = next_result
        return ChatResponse(answer, decision.domain, decision.confidence), current_state

    if decision.domain == ChatDomain.STAFF_STEP:
        current_state.active_domain = ChatDomain.STAFF_STEP
        if not has_staff_index():
            answer = "스텝 공고 검색 인덱스가 아직 준비되지 않았습니다. 관리자에게 인덱스 갱신을 요청해 주세요."
            return ChatResponse(answer, decision.domain, decision.confidence), current_state
        previous_result = current_state.domain_states.get(ChatDomain.STAFF_STEP)
        answer, next_result = answer_staff_chat(query, previous_result)
        if next_result is not None:
            current_state.domain_states[ChatDomain.STAFF_STEP] = next_result
        return ChatResponse(answer, decision.domain, decision.confidence), current_state

    if decision.domain == ChatDomain.GEHARBANG_SERVICE:
        current_state.active_domain = ChatDomain.GEHARBANG_SERVICE
        if not has_service_guide_index():
            answer = "게하르방 이용 안내 문서가 아직 준비되지 않았습니다."
        else:
            answer = answer_service_guide(query)
        return ChatResponse(answer, decision.domain, decision.confidence), current_state

    if decision.domain in PENDING_DOMAIN_MESSAGES:
        current_state.active_domain = decision.domain
        return (
            ChatResponse(PENDING_DOMAIN_MESSAGES[decision.domain], decision.domain, decision.confidence),
            current_state,
        )

    if decision.domain == ChatDomain.GREETING:
        answer = "안녕하세요. 게스트하우스와 스텝 공고에 관해 물어보세요."
    else:
        answer = "질문을 이해하기 어려워요. 게스트하우스 또는 스텝 공고에 관해 조금 더 구체적으로 알려주세요."

    return ChatResponse(answer, decision.domain, decision.confidence), current_state
