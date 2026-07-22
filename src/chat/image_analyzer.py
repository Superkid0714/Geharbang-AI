from __future__ import annotations

import os


DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"

_HEIF_BRANDS = {
    b"heic", b"heix", b"hevc", b"hevx",
    b"heim", b"heis", b"hevm", b"hevs",
    b"mif1", b"msf1",
}


def image_content_matches_mime_type(image_bytes: bytes, mime_type: str) -> bool:
    """Verify the declared MIME type against common image file signatures."""
    normalized_mime_type = mime_type.split(";", 1)[0].strip().lower()
    declared_type = {
        "image/jpeg": "jpeg",
        "image/jpg": "jpeg",
        "image/pjpeg": "jpeg",
        "image/png": "png",
        "image/webp": "webp",
        "image/heic": "heif",
        "image/heif": "heif",
    }.get(normalized_mime_type)
    if declared_type is None:
        return False

    detected_type: str | None = None
    if image_bytes.startswith(b"\xff\xd8\xff"):
        detected_type = "jpeg"
    elif image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        detected_type = "png"
    elif (
        len(image_bytes) >= 12
        and image_bytes[:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    ):
        detected_type = "webp"
    elif len(image_bytes) >= 12 and image_bytes[4:8] == b"ftyp":
        inspection_limit = min(len(image_bytes), 64)
        compatible_brands = {
            image_bytes[offset:offset + 4]
            for offset in range(16, inspection_limit - 3, 4)
        }
        if image_bytes[8:12] in _HEIF_BRANDS or compatible_brands & _HEIF_BRANDS:
            detected_type = "heif"

    return detected_type == declared_type


def describe_chat_image(
    image_bytes: bytes,
    mime_type: str,
    user_query: str,
) -> str:
    """Describe only visible, chat-relevant details for downstream RAG routing."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise ImportError("이미지 분석에는 google-genai 패키지가 필요합니다.") from error

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("이미지 분석을 위한 Gemini API 키가 설정되어 있지 않습니다.")

    prompt = f"""게하르방 AI 챗봇에 첨부된 이미지를 분석하세요.
사용자 질문에 답하는 데 필요한, 이미지에서 객관적으로 확인되는 내용만 한국어 3문장 이내로 설명하세요.
장소를 확실히 식별할 근거가 없으면 특정 장소명을 추측하지 마세요.
게스트하우스 분위기, 음식, 관광지, 표지판의 글자처럼 제주 여행 질문에 유용한 단서를 우선하세요.

사용자 질문: {user_query}
"""

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt,
            ],
        )
    except Exception as error:
        raise RuntimeError("Gemini 이미지 분석 중 오류가 발생했습니다.") from error

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    raise RuntimeError("Gemini가 이미지 설명을 반환하지 않았습니다.")
