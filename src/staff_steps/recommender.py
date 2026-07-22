from __future__ import annotations

from src.chat.persona import GEHARBANG_TONE_RULES
from src.guesthouses.recommender import _generate_gemini_answer
from src.staff_steps.data_loader import load_staff_recruitment, load_staff_recruitments
from src.staff_steps.document_builder import build_staff_recruitment_document, build_staff_recruitment_documents
from src.staff_steps.vector_store import search_staff_recruitments
from src.staff_steps.structured_filter import extract_staff_conditions, filter_staff_recruitments


FOLLOW_UP_MARKERS = ["그 공고", "거기", "그곳", "근무는", "기간은", "조건은", "복지는", "업무는"]


def answer_staff_chat(query: str, previous_result: dict | None = None) -> tuple[str, dict | None]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")

    if previous_result is not None and any(marker in query for marker in FOLLOW_UP_MARKERS):
        result = _load_fresh_staff_result(previous_result)
    else:
        conditions = extract_staff_conditions(query)
        current_items = load_staff_recruitments()
        filtered_items = filter_staff_recruitments(current_items, conditions)
        if not filtered_items:
            return _no_result_message(conditions), previous_result
        candidate_ids = [int(item["id"]) for item in filtered_items]
        results = search_staff_recruitments(
            query,
            top_k=min(5, len(candidate_ids)),
            conditions=None,
            candidate_ids=candidate_ids,
        )
        current_documents = {
            int(document["staffRecruitmentId"]): document
            for document in build_staff_recruitment_documents(filtered_items)
        }
        if not results:
            result = build_staff_recruitment_document(filtered_items[0])
        else:
            result = build_staff_recruitment_document(filtered_items[0])
            for indexed_result in results:
                recruitment_id = indexed_result.get("staffRecruitmentId")
                if isinstance(recruitment_id, int) and recruitment_id in current_documents:
                    result = {
                        **current_documents[recruitment_id],
                        "distance": indexed_result.get("distance"),
                    }
                    break

        if any(keyword in query.replace(" ", "") for keyword in ["평점", "리뷰좋은", "후기좋은", "리뷰많은", "후기많은"]):
            result = _select_by_reviews(query, list(current_documents.values()))

    return _generate_gemini_answer(_build_prompt(query, result)), result


def _select_by_reviews(query: str, documents: list[dict]) -> dict:
    normalized = query.replace(" ", "")
    reviewed = [document for document in documents if document.get("reviewCount", 0) > 0]
    if not reviewed:
        return documents[0]
    if "리뷰많은" in normalized or "후기많은" in normalized:
        return max(reviewed, key=lambda item: (item.get("reviewCount", 0), item.get("averageRating", 0.0)))
    return max(reviewed, key=lambda item: (item.get("averageRating", 0.0), item.get("reviewCount", 0)))


def _load_fresh_staff_result(previous_result: dict) -> dict:
    recruitment_id = previous_result.get("staffRecruitmentId")
    if not isinstance(recruitment_id, int):
        return previous_result
    return build_staff_recruitment_document(load_staff_recruitment(recruitment_id))


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
        return f"지금은 {condition_text} 조건에 딱 맞는 스텝 공고가 없어요. 가장 중요한 조건을 남기고 범위를 조금 넓혀 같이 찾아볼까요?"
    return "지금 확인할 수 있는 스텝 공고가 없어요. 새 공고가 올라온 뒤 다시 같이 살펴봐요."


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
7. 질문받은 조건만 우선 답하고 기본 답변은 2~3문장, 약 250자 안팎으로 끝냅니다. 넓은 질문이어도 핵심 조건 두 개만 말하고 세부 정보를 나열하지 않습니다.
8. 새로운 공고 추천 질문에는 공고 제목과 게스트하우스 이름을 함께 말합니다.

{GEHARBANG_TONE_RULES}

사용자 질문:
{query}

검색된 공고:
제목: {result.get('title', '')}
게스트하우스: {result.get('guestHouseName', '')}
지역: {result.get('region', '')}
정보:
{str(result.get('content', ''))[:3500]}
"""
