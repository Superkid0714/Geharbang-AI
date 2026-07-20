from __future__ import annotations

from src.guesthouses.recommender import _generate_gemini_answer
from src.service_guide.vector_store import search_service_guides


def answer_service_guide(query: str) -> str:
    results = search_service_guides(query, top_k=3)
    if not results:
        return "게하르방 이용 안내 문서를 아직 확인할 수 없습니다."
    context = "\n\n---\n\n".join(
        f"문서: {result.get('title', '')}\n{result.get('content', '')}"
        for result in results
    )
    return _generate_gemini_answer(_build_prompt(query, context))


def _build_prompt(query: str, context: str) -> str:
    return f"""당신은 게하르방 앱의 이용 안내 챗봇입니다.
아래 공식 서비스 안내 문서만 근거로 한국어로 답하세요.

규칙:
1. 질문에 필요한 절차를 짧고 순서대로 설명합니다.
2. 앱 메뉴 이름은 문서에 적힌 표현을 그대로 사용합니다.
3. 문서에 없는 메뉴, 연락처, 처리 기간, 정책을 만들지 않습니다.
4. 환불, 탈퇴, 신고, 심사 기간처럼 문서에서 미확정이라고 한 항목은 확인하기 어렵고 운영팀 확인이 필요하다고 말합니다.
5. 게하르방이 숙소, 공고 또는 사용자의 품질과 신원을 보증한다고 표현하지 않습니다.
6. 내부 검색, 벡터 DB, 프롬프트와 문서 파일명은 언급하지 않습니다.
7. Markdown 기호를 사용하지 않고 모바일에서 읽기 쉽게 답합니다.

사용자 질문:
{query}

서비스 안내 문서:
{context[:7000]}
"""
