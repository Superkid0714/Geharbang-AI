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
    if _looks_like_gibberish(normalized):
        return RouteDecision(ChatDomain.UNCLEAR, 1.0, "gibberish")
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
        decision = _route_with_gemini(query, active_domain)
        if (
            decision.domain == ChatDomain.STAFF_STEP
            and not _has_staff_domain_evidence(normalized, active_domain)
        ):
            return RouteDecision(
                ChatDomain.OUT_OF_SCOPE,
                0.72,
                "generic_work_or_personal_question",
            )
        return decision

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
- out_of_scope: 위 네 분야와 무관하지만 의미가 분명한 일반 지식, 일상 대화 또는 요청
- unclear: 키보드 난타, 무작위 문자, 의미 없는 말, 맥락이 없어 무엇을 묻는지 전혀 알 수 없는 입력

"스텝을 제공하는 게하", "게하에서 일하고 싶어"는 staff_step입니다.
일반 회사의 면접·직장 고민·진로 상담은 staff_step이 아니라 out_of_scope입니다. 질문에 게스트하우스 스텝, 공고 또는 제주 지역의 구체적인 구인 조건이 드러날 때만 staff_step으로 분류하세요.
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
    has_guesthouse_context = any(
        marker in query for marker in ["게하", "게스트하우스"]
    )
    has_work_context = any(
        marker in query
        for marker in ["일하고", "일자리", "구해", "구인", "지원", "근무", "스텝"]
    )
    # Generic work worries belong to general chat. A work expression only wins
    # over accommodation keywords when guesthouse context is also present.
    return 3 if has_guesthouse_context and has_work_context else 0


def _has_staff_domain_evidence(
    query: str,
    active_domain: ChatDomain | None,
) -> bool:
    if active_domain == ChatDomain.STAFF_STEP and _looks_like_follow_up(query):
        return True
    if any(marker in query for marker in ["스텝", "스태프", "구인", "스텝 공고"]):
        return True
    if any(place in query for place in ["게하", "게스트하우스"]) and any(
        marker in query
        for marker in ["일하고", "일자리", "알바", "근무", "지원", "공고", "구해"]
    ):
        return True
    recruitment_terms = {
        "공고", "근무", "알바", "일자리", "지원 자격", "근무 기간",
        "근무시간", "근무 시간", "휴무", "복지", "급여",
    }
    return _keyword_score(query, recruitment_terms) >= 2


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


def _looks_like_gibberish(query: str) -> bool:
    compact = re.sub(r"\s+", "", query)
    if not compact:
        return True
    if not re.search(r"[0-9a-z가-힣ㄱ-ㅎㅏ-ㅣ]", compact):
        return True
    if re.fullmatch(r"[0-9]+", compact):
        return True
    if re.fullmatch(r"[ㄱ-ㅎㅏ-ㅣ]+", compact) and len(compact) >= 3:
        return True
    if len(compact) >= 4 and len(set(compact)) <= 2:
        return True
    # 자모/한글이 섞이지 않은 순수 영문+숫자 문자열이 앞뒤로 똑같이 반복될 때만
    # 키보드를 무의미하게 두 번 친 것("asdfasdf")으로 간주한다. 이 조건이 없으면
    # "감사합니다감사합니다"처럼 실제 문장을 강조하려고 반복한 경우까지 걸러진다.
    if (
        len(compact) >= 6
        and len(compact) % 2 == 0
        and re.fullmatch(r"[0-9a-z]+", compact)
    ):
        midpoint = len(compact) // 2
        if compact[:midpoint] == compact[midpoint:]:
            return True
    if re.fullmatch(r"[a-z]+", compact) and len(compact) >= 4:
        if not re.search(r"[aeiouy]", compact):
            return True
    return False
