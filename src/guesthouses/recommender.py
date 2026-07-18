from __future__ import annotations

import json
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
    """Answer one guesthouse chat turn without prior conversation context."""
    answer, _ = answer_guesthouse_chat(query=query, persist_directory=persist_directory)
    return answer


def answer_guesthouse_chat(
    query: str,
    previous_result: dict | None = None,
    persist_directory: str = "storage/vector_db/guesthouses",
) -> tuple[str, dict | None]:
    """Route a user query, then answer with recommendation or detail style."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")

    intent = _route_query_deterministically(
        query=query,
        has_previous_result=previous_result is not None,
    )
    if intent is None:
        intent = _route_query_with_llm(
            query=query,
            has_previous_result=previous_result is not None,
        )
    if intent == "greeting":
        return (
            "안녕하세요. 제주 게스트하우스 추천을 도와드릴게요. 원하는 지역, 분위기, 파티 여부, 가격대 등을 말해 주세요.",
            previous_result,
        )
    if intent == "help":
        return (
            "제주 게스트하우스 추천과 관련된 질문에 답변할 수 있습니다. 예를 들면 \"애월 쪽 조용한 게하 추천해줘\"처럼 물어보면 됩니다.",
            previous_result,
        )
    if intent == "unrelated":
        return "게스트하우스 추천과 관련된 질문에 답변할 수 있습니다.", previous_result
    if intent == "follow_up":
        if previous_result is None:
            return (
                "어떤 게스트하우스에 대한 질문인지 먼저 알려주세요. 예를 들면 \"메르블루 게하 어때?\"처럼 숙소 이름을 포함해서 물어볼 수 있습니다.",
                previous_result,
            )
        prompt = _build_detail_prompt(query=query, result=previous_result)
        return _generate_gemini_answer(prompt), previous_result

    results = search_guesthouses(query, top_k=1, persist_directory=persist_directory)
    if not results:
        return "조건에 맞는 게스트하우스를 찾지 못했습니다.", previous_result

    result = results[0]
    if intent == "detail_question":
        prompt = _build_detail_prompt(query=query, result=result)
    else:
        prompt = _build_recommendation_prompt(query=query, result=result)
    return _generate_gemini_answer(prompt), result


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
        return _clean_generated_answer(text)
    return _clean_generated_answer(str(response))


def _route_query_with_llm(query: str, has_previous_result: bool) -> str:
    prompt = f"""당신은 제주 게스트하우스 추천 챗봇의 라우터입니다.
사용자 질문을 보고 아래 intent 중 정확히 하나만 고르세요.

intent 목록:
- greeting: 인사. 예: 안녕, 안녕하세요, 하이
- help: 사용법이나 가능한 질문을 묻는 경우
- recommendation: 새로운 게스트하우스 추천을 요청하는 경우. 예: 애월 조용한 게하 추천해줘, 바닷가 근처 숙소 찾아줘
- detail_question: 특정 게스트하우스 이름을 언급하거나, 특정 숙소 자체가 어떤지 묻는 경우. 예: 메르블루 게하 어때, 동행in협재 가격 알려줘
- follow_up: 직전에 추천받은 숙소에 대해 이어서 묻는 경우. 예: 가격은?, 분위기는 뭐야?, 파티는 어때?
- unrelated: 게스트하우스 추천과 관련 없는 질문

판단 규칙:
1. recommendation은 "추천해줘", "찾아줘", "골라줘", "어디가 좋아"처럼 새 숙소를 골라달라는 요청이 명확할 때만 선택합니다.
2. "메르블루 게스트하우스에 대해서 설명해줘", "동행in협재 어때?", "제주게토 가격 알려줘"처럼 특정 숙소에 대해 설명, 가격, 분위기, 파티, 위치를 묻는 경우는 detail_question입니다.
3. 게스트하우스라는 단어가 있어도 설명/정보/가격/분위기/파티를 묻는 질문이면 recommendation이 아니라 detail_question입니다.
4. 이전 추천 숙소가 있고, "가격은?", "분위기는?", "거긴 어때?", "파티는?"처럼 짧게 이어 묻는 경우 follow_up입니다.
5. 이전 추천 숙소가 없으면 짧은 후속 질문은 detail_question이 아니라 help로 분류합니다.
6. 사람 평가, 맛집, 날씨, 일반 지식 등은 unrelated입니다.

예시:
- "메르블루 게스트하우스에 대해서 설명해줄 수 있어?" -> detail_question
- "메르블루 게하 어때?" -> detail_question
- "메르블루 가격 알려줘" -> detail_question
- "바닷가 근처 소규모 파티 게하 추천해줘" -> recommendation
- "애월 쪽 조용한 숙소 찾아줘" -> recommendation
- "가격은 어떻게 돼?" 그리고 이전 추천 숙소가 있음 -> follow_up
- "신준현은 어때?" -> unrelated

이전 추천 숙소 존재 여부: {has_previous_result}
사용자 질문: {query}

반드시 JSON만 출력하세요.
예:
{{"intent": "recommendation"}}
"""
    raw_answer = _generate_gemini_answer(prompt)
    try:
        parsed = json.loads(raw_answer)
    except json.JSONDecodeError:
        start = raw_answer.find("{")
        end = raw_answer.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return "unrelated"
        parsed = json.loads(raw_answer[start : end + 1])

    intent = parsed.get("intent")
    allowed_intents = {"greeting", "help", "recommendation", "detail_question", "follow_up", "unrelated"}
    if intent not in allowed_intents:
        return "unrelated"
    return _correct_routed_intent(intent, query=query, has_previous_result=has_previous_result)


def _route_query_deterministically(query: str, has_previous_result: bool) -> str | None:
    normalized_query = _normalize_query(query)

    if _looks_like_greeting(normalized_query):
        return "greeting"
    if _looks_like_help(normalized_query):
        return "help"

    # Follow-up questions must stay pinned to the last recommended guesthouse.
    if has_previous_result and _looks_like_follow_up(normalized_query):
        return "follow_up"

    # Named guesthouse questions should use the detail prompt, not recommendation.
    if _looks_like_named_guesthouse_detail_question(normalized_query):
        return "detail_question"

    # Explicit recommendation verbs are the only deterministic path to the
    # recommendation template.
    if _looks_like_explicit_recommendation(normalized_query):
        return "recommendation"

    if _looks_like_guesthouse_detail_question(normalized_query):
        return "detail_question"

    return None


def _correct_routed_intent(intent: str, query: str, has_previous_result: bool) -> str:
    normalized_query = _normalize_query(query)

    if _looks_like_greeting(normalized_query):
        return "greeting"
    if _looks_like_help(normalized_query):
        return "help"
    if has_previous_result and _looks_like_follow_up(normalized_query):
        return "follow_up"
    if _looks_like_named_guesthouse_detail_question(normalized_query):
        return "detail_question"
    if _looks_like_explicit_recommendation(normalized_query):
        return "recommendation"
    if _looks_like_guesthouse_detail_question(normalized_query):
        return "detail_question"
    if intent == "follow_up" and not has_previous_result:
        return "help"
    if intent == "detail_question":
        return "unrelated"
    if intent == "recommendation":
        return "unrelated"
    return intent


def _normalize_query(query: str) -> str:
    return " ".join(query.casefold().strip().rstrip("?!.。！？").split())


def _looks_like_greeting(query: str) -> bool:
    return query in {"안녕", "안녕하세요", "하이", "헬로", "hello", "hi"}


def _looks_like_help(query: str) -> bool:
    return any(keyword in query for keyword in ["뭐 할 수", "무엇을 할 수", "사용법", "도움말", "어떻게 물어"])


def _looks_like_explicit_recommendation(query: str) -> bool:
    return any(keyword in query for keyword in ["추천", "찾아줘", "찾아 줘", "골라줘", "골라 줘", "어디가 좋아", "어디 좋아"])


def _looks_like_guesthouse_detail_question(query: str) -> bool:
    guesthouse_markers = ["게하", "게스트하우스", "숙소", "도미토리"]
    detail_markers = [
        "설명",
        "알려",
        "정보",
        "대해서",
        "대해",
        "어때",
        "어떄",
        "가격",
        "얼마",
        "비용",
        "분위기",
        "파티",
        "위치",
        "주소",
        "객실",
        "방",
        "체크인",
        "체크아웃",
        "편의시설",
    ]
    known_name_fragments = ["메르블루", "동행", "협재", "제주게토"]

    has_guesthouse_marker = any(marker in query for marker in guesthouse_markers)
    has_detail_marker = any(marker in query for marker in detail_markers)
    return has_guesthouse_marker and has_detail_marker


def _looks_like_named_guesthouse_detail_question(query: str) -> bool:
    detail_markers = [
        "설명",
        "알려",
        "정보",
        "대해서",
        "대해",
        "어때",
        "어떄",
        "가격",
        "얼마",
        "비용",
        "분위기",
        "파티",
        "위치",
        "주소",
        "객실",
        "방",
        "체크인",
        "체크아웃",
        "편의시설",
        "추천 이유",
    ]
    known_name_fragments = ["메르블루", "동행", "협재", "제주게토"]

    has_known_name = any(fragment in query for fragment in known_name_fragments)
    has_detail_marker = any(marker in query for marker in detail_markers)
    return has_known_name and has_detail_marker


def _looks_like_follow_up(query: str) -> bool:
    follow_up_markers = [
        "가격",
        "얼마",
        "비용",
        "분위기",
        "파티",
        "위치",
        "주소",
        "객실",
        "방",
        "체크인",
        "체크아웃",
        "거기",
        "그 숙소",
        "그 게하",
        "그 게스트하우스",
    ]
    return any(marker in query for marker in follow_up_markers)


def _build_recommendation_prompt(query: str, result: dict) -> str:
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
10. 이 프롬프트에 들어온 질문은 게스트하우스 추천 질문입니다. 추천 형식으로 자연스럽게 답변합니다.
11. 추천 게스트하우스는 반드시 1개만 제시합니다.
12. 다른 후보가 있을 수 있다는 식으로 여러 숙소를 나열하지 마세요.
13. Markdown 문법을 사용하지 않습니다. 특히 **, ##, ``` 같은 서식 기호를 출력하지 않습니다.

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


def _build_detail_prompt(query: str, result: dict) -> str:
    guesthouse_name = result.get("guestHouseName", "")
    content = _truncate_content(str(result.get("content", "")))

    return f"""당신은 제주 게스트하우스 추천 챗봇입니다.
사용자는 "{guesthouse_name}"에 대해 질문하고 있습니다.
아래 게스트하우스 정보 1개만 참고해서 사용자의 질문에 자연스럽게 답변하세요.

핵심 규칙:
1. 아래 정보에 있는 내용만 사용합니다.
2. 정보에 없는 내용은 절대 추측하거나 지어내지 않습니다.
3. 알 수 없는 정보는 "제공된 정보만으로는 확인하기 어렵습니다."라고 말합니다.
4. 추천 답변 형식을 사용하지 않습니다.
5. "검색 결과에 따르면", "제공된 정보에 따르면", "문서를 보면", "정보를 보면", "content에 따르면", "위 내용을 바탕으로"처럼 내부 자료를 참고했다는 표현을 쓰지 않습니다.
6. distance, 벡터 점수, 임베딩, Chroma 같은 내부 검색 정보를 사용자에게 말하지 않습니다.
7. 가격, 파티비, 운영 여부, 이벤트 정보는 등록된 정보 기준이며 실제 예약 전 확인이 필요하다고 안내합니다.
8. 다른 게스트하우스를 새로 추천하거나 여러 숙소를 나열하지 않습니다.
9. 질문에 필요한 내용만 간결하게 답합니다.
10. Markdown 문법을 사용하지 않습니다. 특히 **, ##, ``` 같은 서식 기호를 출력하지 않습니다.

답변 방식:
- 가격 질문이면 객실 가격과 파티 참가비를 구분합니다.
- 분위기 질문이면 분위기 키워드와 소개/파티 내용을 바탕으로 설명합니다.
- 파티 질문이면 파티 유형, 시간, 외부인 참여 가능 여부, 참가비를 확인 가능한 범위에서 말합니다.
- 위치 질문이면 지역과 지번 주소만 말합니다.
- "어때?"처럼 넓게 물으면 분위기, 객실, 파티, 확인할 점을 짧게 요약합니다.

사용자 질문:
{query}

게스트하우스:
{guesthouse_name}

정보:
{content}
"""


def _truncate_content(content: str, max_length: int = 3000) -> str:
    if len(content) <= max_length:
        return content
    return content[:max_length].rstrip() + "\n...[검색 문서가 길어 일부만 제공되었습니다.]"


def _clean_generated_answer(answer: str) -> str:
    return answer.strip().replace("**", "")
