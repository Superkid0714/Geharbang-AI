from __future__ import annotations

from src.guesthouses.recommender import _generate_gemini_answer
from src.staff_steps.vector_store import search_staff_recruitments
from src.staff_steps.structured_filter import extract_staff_conditions


FOLLOW_UP_MARKERS = ["그 공고", "거기", "그곳", "근무는", "기간은", "조건은", "복지는", "업무는"]


def answer_staff_chat(query: str, previous_result: dict | None = None) -> tuple[str, dict | None]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")

    if previous_result is not None and any(marker in query for marker in FOLLOW_UP_MARKERS):
        result = previous_result
    else:
        conditions = extract_staff_conditions(query)
        results = search_staff_recruitments(query, top_k=1, conditions=conditions)
        if not results:
            return _no_result_message(conditions), previous_result
        result = results[0]

    return _generate_gemini_answer(_build_prompt(query, result)), result


def _no_result_message(conditions: dict[str, str]) -> str:
    labels = []
    if "region" in conditions:
        labels.append(conditions["region"].replace("_", "·"))
    if "workingPeriod" in conditions:
        labels.append(conditions["workingPeriod"])
    if "gender" in conditions:
        labels.append(f"성별 {conditions['gender']}")
    condition_text = " ".join(labels)
    if condition_text:
        return f"현재 {condition_text} 조건에 맞는 활성 스텝 공고를 찾지 못했습니다. 조건을 조금 넓혀 다시 질문해 주세요."
    return "현재 확인할 수 있는 활성 스텝 공고가 없습니다."


def _build_prompt(query: str, result: dict) -> str:
    return f"""당신은 게하르방의 제주 게스트하우스 스텝 공고 안내 챗봇입니다.
사용자 질문과 검색된 활성 공고 하나만 참고해 한국어로 답하세요.

규칙:
1. 검색된 공고에 있는 정보만 사용하고 추측하지 않습니다.
2. 없는 정보는 확인하기 어렵다고 명확히 말합니다.
3. 공고의 근무 시작일, 기간, 업무, 근무·휴무일, 성별 조건과 복지를 구분합니다.
4. 사용자의 조건과 명확히 맞지 않으면 억지로 추천하지 않습니다.
5. 공고 내용은 변경될 수 있으므로 실제 지원 전에 상세 화면에서 다시 확인하도록 안내합니다.
6. 내부 벡터 점수, 임베딩, 검색 시스템은 언급하지 않습니다.
7. Markdown 기호 없이 간결하게 답합니다.
8. 새로운 공고 추천 질문에는 공고 제목과 게스트하우스 이름을 함께 말합니다.

사용자 질문:
{query}

검색된 공고:
제목: {result.get('title', '')}
게스트하우스: {result.get('guestHouseName', '')}
지역: {result.get('region', '')}
정보:
{str(result.get('content', ''))[:3500]}
"""
