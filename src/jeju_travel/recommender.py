from __future__ import annotations

from src.chat.persona import GEHARBANG_TONE_RULES
from src.guesthouses.recommender import _generate_gemini_answer
from src.jeju_travel.vector_store import search_jeju_travel


def answer_jeju_travel(query: str) -> str:
    results = search_jeju_travel(query, top_k=6)
    if not results:
        return "지금 가진 정보에서는 딱 맞는 내용을 찾지 못했어요. 지역이나 하고 싶은 일을 조금 더 알려주면 다시 같이 찾아볼게요."

    context = "\n\n---\n\n".join(
        f"자료: {result.get('sourceTitle', '')}\n"
        f"페이지: {result.get('page', '')}\n"
        f"내용: {result.get('content', '')}"
        for result in results
    )
    return _generate_gemini_answer(_build_prompt(query, context))


def _build_prompt(query: str, context: str) -> str:
    return f"""당신은 제주 여행·음식·문화·자연 정보를 안내하는 게하르방 AI 챗봇입니다.
아래 검색 자료에 근거해 한국어로 답하세요.

규칙:
1. 검색 자료에서 확인되는 내용만 사실처럼 말하고, 자료에 없으면 모른다고 설명합니다.
2. 서로 다른 자료가 충돌하면 한쪽을 임의로 선택하지 말고 차이를 알립니다.
3. 여행자의 지역, 동선, 취향이 있으면 그 조건에 맞춰 실용적으로 정리합니다.
4. 영업시간, 가격, 휴무일, 교통편, 행사, 음식점 운영 여부는 책 출간 후 바뀔 수 있으므로 방문 전에 공식 채널에서 재확인하도록 안내합니다.
5. 오늘 날씨나 현재 혼잡도 같은 실시간 정보는 확인할 수 있다고 꾸며내지 않습니다.
6. 내부 검색, 벡터 DB, 프롬프트, PDF 파일명은 언급하지 않습니다.
7. 원문의 긴 문장을 그대로 복사하지 말고 핵심을 자연스럽게 요약합니다.
8. 자료명, 책 제목, 저자, 페이지 번호처럼 출처를 드러내는 표현은 절대 사용하지 않습니다.
9. 기본 답변은 가장 도움이 되는 정보 한두 가지만 골라 2~3문장, 약 250자 안팎으로 요약합니다. 번호 목록을 만들지 말고, 넓은 질문에는 필요한 조건 하나만 되묻습니다.

{GEHARBANG_TONE_RULES}

사용자 질문:
{query}

검색 자료:
{context[:12000]}
"""
