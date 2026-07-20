from __future__ import annotations

import json
import os
import re

from src.chat.schemas import ChatDomain, RouteDecision


GUESTHOUSE_KEYWORDS = {
    "게하", "게스트하우스", "숙소", "도미토리", "객실", "체크인", "체크아웃",
    "숙박", "파티 게하", "파티 숙소",
}
STAFF_STEP_KEYWORDS = {
    "스텝", "스태프", "구인", "공고", "근무", "알바", "일자리", "지원 자격",
    "근무 기간", "근무시간", "근무 시간", "휴무", "복지", "급여", "잡일",
}
JEJU_TRAVEL_KEYWORDS = {
    "관광지", "여행지", "맛집", "음식", "카페", "날씨", "오름", "해수욕장",
    "흑돼지", "갈치", "여행 코스",
}
SERVICE_KEYWORDS = {
    "게하르방", "회원가입", "로그인", "내정보", "내 정보", "찜", "채팅",
    "신청 취소", "지원서", "지원 내역", "지원 현황", "사장님 인증", "게시글 등록",
    "앱 사용", "알림 설정",
}
GREETING_WORDS = {"안녕", "안녕하세요", "하이", "헬로", "hello", "hi"}
FOLLOW_UP_KEYWORDS = {
    "거기", "그곳", "그 곳", "그 숙소", "그 게하", "그 공고", "그 일자리",
    "가격은", "위치는", "조건은", "근무는", "기간은", "파티는",
}


def route_chat_query(
    query: str,
    active_domain: ChatDomain | None = None,
    use_llm_fallback: bool = True,
) -> RouteDecision:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")

    normalized = _normalize(query)
    if normalized in GREETING_WORDS:
        return RouteDecision(ChatDomain.GREETING, 1.0, "greeting")

    scores = {
        ChatDomain.GUESTHOUSE: _keyword_score(normalized, GUESTHOUSE_KEYWORDS),
        ChatDomain.STAFF_STEP: _keyword_score(normalized, STAFF_STEP_KEYWORDS),
        ChatDomain.JEJU_TRAVEL: _keyword_score(normalized, JEJU_TRAVEL_KEYWORDS),
        ChatDomain.GEHARBANG_SERVICE: _keyword_score(normalized, SERVICE_KEYWORDS),
    }

    if _looks_like_service_how_to(normalized):
        scores[ChatDomain.GEHARBANG_SERVICE] += 4

    # "게하 스텝", "게스트하우스에서 일"은 숙소가 아니라 채용 공고 질문입니다.
    scores[ChatDomain.STAFF_STEP] += _staff_context_bonus(normalized)

    if active_domain is not None and _looks_like_follow_up(normalized):
        scores[active_domain] = scores.get(active_domain, 0) + 3

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_domain, best_score = ranked[0]
    second_score = ranked[1][1]

    if best_score >= 2 and best_score > second_score:
        confidence = min(0.98, 0.72 + best_score * 0.06)
        return RouteDecision(best_domain, confidence, "keyword")

    if use_llm_fallback and os.getenv("GEMINI_API_KEY"):
        return _route_with_gemini(query, active_domain)

    if best_score > 0 and best_score > second_score:
        return RouteDecision(best_domain, 0.62, "weak_keyword")

    return RouteDecision(ChatDomain.OUT_OF_SCOPE, 0.4, "insufficient_context")


def _route_with_gemini(query: str, active_domain: ChatDomain | None) -> RouteDecision:
    try:
        from google import genai
    except ImportError:
        return RouteDecision(ChatDomain.OUT_OF_SCOPE, 0.4, "llm_unavailable")

    prompt = f"""게하르방 AI 챗봇의 도메인 라우터입니다.
질문을 정확히 하나의 domain으로 분류하세요.

- guesthouse: 게스트하우스 추천, 숙박, 객실, 가격, 시설, 파티 및 특정 숙소 정보
- staff_step: 게스트하우스 스텝 구인 공고, 업무, 근무 기간·시간·조건·복지 및 지원할 공고 검색
- jeju_travel: 제주 관광지, 음식점, 카페, 날씨와 여행 정보
- geharbang_service: 게하르방 앱 기능, 계정, 찜, 지원 방법, 채팅과 이용 안내
- greeting: 인사
- out_of_scope: 위 분야와 무관하거나 판단할 정보가 부족함

"스텝을 제공하는 게하", "게하에서 일하고 싶어"는 staff_step입니다.
직전 대화 domain이 있고 대상을 생략한 후속 질문이면 그 domain을 유지합니다.
직전 domain: {active_domain.value if active_domain else "없음"}
질문: {query}

JSON만 출력하세요: {{"domain":"guesthouse","confidence":0.95,"reason":"짧은 이유"}}
"""
    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
            contents=prompt,
        )
        payload = _parse_json_object(response.text or "")
        domain = ChatDomain(payload["domain"])
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.7))))
        return RouteDecision(domain, confidence, str(payload.get("reason", "llm")))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, RuntimeError):
        return RouteDecision(ChatDomain.OUT_OF_SCOPE, 0.4, "llm_routing_failed")


def _parse_json_object(value: str) -> dict:
    cleaned = value.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise json.JSONDecodeError("JSON object not found", cleaned, 0)
    return json.loads(cleaned[start : end + 1])


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().strip().rstrip("?!.。！？"))


def _keyword_score(query: str, keywords: set[str]) -> int:
    return sum(1 for keyword in keywords if keyword in query)


def _staff_context_bonus(query: str) -> int:
    matches = sum(
        1
        for phrase in ["일하고", "일자리", "구해", "구인", "지원", "근무", "스텝"]
        if phrase in query
    )
    # A work-seeking verb is stronger evidence than a generic accommodation noun.
    return matches + 1 if matches else 0


def _looks_like_follow_up(query: str) -> bool:
    if any(keyword in query for keyword in FOLLOW_UP_KEYWORDS):
        return True
    return len(query.split()) <= 3 and any(
        marker in query for marker in ["어때", "알려줘", "얼마야", "뭐야"]
    )


def _looks_like_service_how_to(query: str) -> bool:
    how_to_markers = ["어떻게", "방법", "어디서", "어디에서", "안 돼", "안돼", "오류", "사용법"]
    feature_markers = [
        "게하르방", "로그인", "회원가입", "지원서", "지원 내역", "지원 현황", "찜",
        "채팅", "알림", "사장님 인증", "게스트하우스 등록", "공고 등록",
    ]
    search_markers = ["추천", "찾아줘", "찾아 줘", "공고 있어", "공고 보여"]
    return (
        any(marker in query for marker in how_to_markers)
        and any(marker in query for marker in feature_markers)
        and not any(marker in query for marker in search_markers)
    )
