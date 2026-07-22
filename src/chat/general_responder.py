from __future__ import annotations

import os

from src.chat.persona import GEHARBANG_TONE_RULES


DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"


def answer_general_chat(query: str) -> str:
    """Answer a question that does not belong to one of the four service domains."""
    try:
        from google import genai
    except ImportError as error:
        raise ImportError("일반 AI 답변에는 google-genai 패키지가 필요합니다.") from error

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("일반 AI 답변을 위한 Gemini API 키가 설정되어 있지 않습니다.")

    prompt = f"""당신은 게하르방 앱의 친절한 AI 챗봇입니다.
아래 질문은 게스트하우스, 스텝 공고, 제주 여행, 게하르방 이용 안내 중 어느 분야에도 해당하지 않는 일반 질문입니다.

답변 원칙:
1. 사용자의 언어에 맞춰 자연스럽고 정확하게 답변합니다.
2. 질문에 바로 답하고 불필요하게 게하르방 주제로 돌리지 않습니다.
3. 게하르방 DB나 서비스 내부 정보를 조회한 것처럼 말하지 않습니다.
4. 확실하지 않은 내용은 단정하지 않습니다.
5. 기본 답변은 핵심만 2~3문장, 약 250자 안팎으로 작성합니다. 사용자가 자세한 설명을 요청한 경우에만 최대 5문장까지 답합니다.
6. 질문이 의미 없는 문자나 말로 되어 있으면 억지로 해석하지 말고 무엇이 궁금한지 다시 물어봅니다.

{GEHARBANG_TONE_RULES}

사용자 질문:
{query}
"""

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            contents=prompt,
        )
    except Exception as error:
        raise RuntimeError("Gemini 일반 답변 생성 중 오류가 발생했습니다.") from error

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip().replace("**", "")
    raise RuntimeError("Gemini가 일반 질문에 대한 답변을 반환하지 않았습니다.")
