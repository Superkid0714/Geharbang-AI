from __future__ import annotations

from src.chat.general_responder import answer_general_chat
from src.chat.persona import GREETING_MESSAGE, UNCLEAR_QUESTION_MESSAGE
from src.chat.router import route_chat_query
from src.chat.schemas import ChatDomain, ChatResponse, ConversationState
from src.guesthouses.recommender import answer_guesthouse_chat
from src.guesthouses.vector_store import is_vector_store_initialized as is_guesthouse_index_ready
from src.jeju_travel.recommender import answer_jeju_travel
from src.jeju_travel.vector_store import has_vector_store_documents as has_jeju_travel_index
from src.staff_steps.recommender import answer_staff_chat
from src.staff_steps.vector_store import is_vector_store_initialized as is_staff_index_ready
from src.service_guide.recommender import answer_service_guide
from src.service_guide.vector_store import has_vector_store_documents as has_service_guide_index


def answer_chat(
    query: str,
    state: ConversationState | None = None,
) -> tuple[ChatResponse, ConversationState]:
    current_state = state or ConversationState()
    decision = route_chat_query(query, active_domain=current_state.active_domain)

    if decision.domain == ChatDomain.GUESTHOUSE:
        if not is_guesthouse_index_ready():
            current_state.active_domain = ChatDomain.GUESTHOUSE
            answer = "지금은 게스트하우스 정보를 불러오지 못했어요. 잠시 뒤에 다시 물어봐 주세요."
            return ChatResponse(answer, decision.domain, decision.confidence), current_state
        previous_result = current_state.domain_states.get(ChatDomain.GUESTHOUSE)
        answer, next_result = answer_guesthouse_chat(query, previous_result=previous_result)
        current_state.active_domain = ChatDomain.GUESTHOUSE
        if next_result is not None:
            current_state.domain_states[ChatDomain.GUESTHOUSE] = next_result
        return ChatResponse(answer, decision.domain, decision.confidence), current_state

    if decision.domain == ChatDomain.STAFF_STEP:
        current_state.active_domain = ChatDomain.STAFF_STEP
        if not is_staff_index_ready():
            answer = "지금은 스텝 공고를 불러오지 못했어요. 잠시 뒤에 다시 확인해볼까요?"
            return ChatResponse(answer, decision.domain, decision.confidence), current_state
        previous_result = current_state.domain_states.get(ChatDomain.STAFF_STEP)
        answer, next_result = answer_staff_chat(query, previous_result)
        if next_result is not None:
            current_state.domain_states[ChatDomain.STAFF_STEP] = next_result
        return ChatResponse(answer, decision.domain, decision.confidence), current_state

    if decision.domain == ChatDomain.GEHARBANG_SERVICE:
        current_state.active_domain = ChatDomain.GEHARBANG_SERVICE
        if not has_service_guide_index():
            answer = "지금은 이용 안내를 확인하기 어려워요. 잠시 뒤에 다시 물어봐 주세요."
        else:
            answer = answer_service_guide(query)
        return ChatResponse(answer, decision.domain, decision.confidence), current_state

    if decision.domain == ChatDomain.JEJU_TRAVEL:
        current_state.active_domain = ChatDomain.JEJU_TRAVEL
        if not has_jeju_travel_index():
            answer = "지금은 제주 여행 정보를 불러오지 못했어요. 잠시 뒤에 다시 이야기해봐요."
        else:
            answer = answer_jeju_travel(query)
        return ChatResponse(answer, decision.domain, decision.confidence), current_state

    if decision.domain == ChatDomain.GREETING:
        answer = GREETING_MESSAGE
    elif decision.domain == ChatDomain.UNCLEAR:
        answer = UNCLEAR_QUESTION_MESSAGE
    else:
        answer = answer_general_chat(query)

    return ChatResponse(answer, decision.domain, decision.confidence), current_state
