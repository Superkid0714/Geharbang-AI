from __future__ import annotations

import os

from src.guesthouses.vector_store import search_guesthouses


DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
MISSING_GEMINI_API_KEY_MESSAGE = (
    "Gemini API 키가 설정되어 있지 않습니다. 프로젝트 루트의 .env 파일에 GEMINI_API_KEY를 추가해주세요."
)


def recommend_guesthouse(
    query: str,
    persist_directory: str = "storage/vector_db/guesthouses",
) -> str:
    """Search one guesthouse and generate one Gemini recommendation answer."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")

    results = search_guesthouses(query, top_k=1, persist_directory=persist_directory)
    if not results:
        return "조건에 맞는 게스트하우스를 찾지 못했습니다."

    result = results[0]
    prompt = _build_prompt(query=query, result=result)
    return _generate_gemini_answer(prompt)


def _generate_gemini_answer(prompt: str) -> str:
    try:
        from dotenv import load_dotenv
        from google import genai
    except ImportError as error:
        raise ImportError("Gemini 답변 생성에는 google-genai와 python-dotenv 패키지가 필요합니다.") from error

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(MISSING_GEMINI_API_KEY_MESSAGE)

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
    except Exception as error:
        raise RuntimeError("Gemini 답변 생성 중 오류가 발생했습니다.") from error

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return str(response).strip()


def _build_prompt(query: str, result: dict) -> str:
    guesthouse_name = result.get("guestHouseName", "")
    content = _truncate_content(str(result.get("content", "")))

    return f"""당신은 제주 게스트하우스 추천 챗봇입니다.
사용자의 질문과 검색된 게스트하우스 정보 1개만 참고해 답변합니다.

핵심 규칙:
1. 검색 결과에 있는 정보만 사용합니다.
2. 검색 결과에 없는 정보는 절대 추측하거나 지어내지 않습니다.
3. 알 수 없는 정보는 "제공된 정보만으로는 확인하기 어렵습니다."라고 말합니다.
4. 검색된 게스트하우스가 사용자 조건과 맞는지 한 번 더 대조합니다.
5. 사용자 조건과 명확히 맞지 않으면 억지로 추천하지 않습니다.
6. 조건과 정확히 맞는지 애매하면 "현재 검색 결과만으로는 조건에 완전히 맞는지 확인하기 어렵지만, 참고할 만한 후보입니다."라고 말합니다.
7. 가격, 파티비, 운영 여부, 이벤트 정보는 등록된 정보 기준이며 실제 예약 전 확인이 필요하다고 안내합니다.
8. distance, 벡터 점수, 임베딩, Chroma 같은 내부 검색 정보를 사용자에게 말하지 않습니다.
9. "무조건", "최고", "완벽한" 같은 과장 표현은 사용하지 않습니다.
10. 게스트하우스 추천과 무관한 질문이면 "게스트하우스 추천과 관련된 질문에 답변할 수 있습니다."라고 답합니다.
11. 추천 게스트하우스는 반드시 1개만 제시합니다.
12. 다른 후보가 있을 수 있다는 식으로 여러 숙소를 나열하지 마세요.

답변 형식:
질문하신 조건을 기준으로 보면, 가장 먼저 추천할 수 있는 곳은 "{guesthouse_name}"입니다.

- 추천 이유:
- 이런 분께 어울려요:
- 확인이 필요한 점:

마지막 문장:
가격, 파티비, 운영 정보는 등록된 정보 기준이므로 실제 예약 전 확인이 필요합니다.

검색 결과가 사용자 조건과 잘 맞지 않는 경우:
"현재 검색 결과만으로는 조건에 정확히 맞는 숙소를 찾기 어렵습니다. 다만 참고할 만한 후보로는 "{guesthouse_name}"이 있습니다."
라고 말한 뒤 이유를 짧게 설명합니다.

사용자 질문:
{query}

검색 결과:
이름: {guesthouse_name}
정보:
{content}
"""


def _truncate_content(content: str, max_length: int = 3000) -> str:
    if len(content) <= max_length:
        return content
    return content[:max_length].rstrip() + "\n...[검색 문서가 길어 일부만 제공되었습니다.]"
