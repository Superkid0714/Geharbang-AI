from __future__ import annotations

import os
import logging
import threading
import time
import uuid
import secrets
import json
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.chat.orchestrator import answer_chat
from src.chat.image_analyzer import describe_chat_image, image_content_matches_mime_type
from src.chat.schemas import (
    ConversationState,
    deserialize_conversation_state,
    serialize_conversation_state,
)
from src.guesthouses.vector_store import has_vector_store_documents as has_guesthouse_index
from src.staff_steps.vector_store import has_vector_store_documents as has_staff_index
from src.service_guide.vector_store import has_vector_store_documents as has_service_guide_index
from src.guesthouses.data_loader import load_guesthouse_from_backend
from src.guesthouses.document_builder import build_guesthouse_document
from src.guesthouses.vector_store import delete_guesthouse_document, upsert_guesthouse_document
from src.staff_steps.data_loader import load_staff_recruitment
from src.staff_steps.document_builder import build_staff_recruitment_document
from src.staff_steps.vector_store import (
    delete_staff_recruitment_document,
    upsert_staff_recruitment_document,
)
from src.index_reconciler import reconcile_dynamic_indexes


load_dotenv(dotenv_path=".env")

logger = logging.getLogger(__name__)

app = FastAPI(title="Geharbang AI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("AI_CORS_ORIGINS", "*").split(",")],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

SESSION_TTL_SECONDS = 30 * 60  # 30분 미사용 세션은 다음 접근/스윕 시 정리한다.
MAX_SESSIONS = 10_000
MAX_CHAT_IMAGE_BYTES = 5 * 1024 * 1024
SUPPORTED_CHAT_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/pjpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
}

# session_id -> (마지막 접근 시각, 대화 상태). FE가 명시적으로 reset을 호출하지 않아도
# 오래된 세션은 아래 _evict_expired_sessions_locked에서 자동으로 제거되어 메모리가 무한히 쌓이지 않는다.
_sessions: dict[str, tuple[float, ConversationState]] = {}
_sessions_lock = threading.Lock()
_inference_lock = threading.Lock()
_reconciliation_stop = threading.Event()
_reconciliation_thread: threading.Thread | None = None
_reconciliation_status: dict[str, object] = {
    "lastSuccessAt": None,
    "lastResult": None,
    "lastError": None,
}


def _evict_expired_sessions_locked() -> None:
    """_sessions_lock을 쥔 상태에서 호출해야 한다."""
    now = time.monotonic()
    expired = [
        session_id
        for session_id, (last_seen, _state) in _sessions.items()
        if now - last_seen > SESSION_TTL_SECONDS
    ]
    for session_id in expired:
        del _sessions[session_id]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    sessionId: str | None = Field(default=None, max_length=100)
    context: dict[str, Any] | None = None


class ChatResponseBody(BaseModel):
    sessionId: str
    answer: str
    domain: str
    confidence: float
    context: dict[str, Any]


@app.get("/health")
def health() -> dict:
    guesthouse_ready = has_guesthouse_index()
    staff_ready = has_staff_index()
    service_guide_ready = has_service_guide_index()
    return {
        "status": "ok" if guesthouse_ready and staff_ready and service_guide_ready else "degraded",
        "indexes": {
            "guesthouse": guesthouse_ready,
            "staffStep": staff_ready,
            "serviceGuide": service_guide_ready,
        },
        "geminiConfigured": bool(os.getenv("GEMINI_API_KEY")),
        "reconciliation": dict(_reconciliation_status),
    }


@app.get("/ready")
def ready() -> dict:
    status = health()
    if status["status"] != "ok" or not status["geminiConfigured"]:
        raise HTTPException(status_code=503, detail=status)
    return status


@app.post("/chat", response_model=ChatResponseBody)
def chat(request: ChatRequest) -> ChatResponseBody:
    session_id, state = _load_conversation_state(request.sessionId, request.context)

    try:
        # BGE-M3 is shared in-process; serialize inference to avoid model races.
        with _inference_lock:
            response, next_state = answer_chat(request.message, state)
    except (ValueError, RuntimeError, ImportError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return _save_session_and_build_response(session_id, response, next_state)


@app.post("/chat/image", response_model=ChatResponseBody)
async def chat_with_image(
    message: str = Form(..., min_length=1, max_length=1000),
    sessionId: str | None = Form(default=None, max_length=100),
    context: str | None = Form(default=None),
    image: UploadFile = File(...),
) -> ChatResponseBody:
    received_mime_type = (image.content_type or "").split(";", 1)[0].strip().lower()
    if received_mime_type not in SUPPORTED_CHAT_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="지원하지 않는 이미지 형식입니다.")
    mime_type = (
        "image/jpeg"
        if received_mime_type in {"image/jpg", "image/pjpeg"}
        else received_mime_type
    )

    image_bytes = await image.read(MAX_CHAT_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_CHAT_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="이미지는 5MB 이하만 가능합니다.")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="이미지 파일이 비어 있습니다.")
    if not image_content_matches_mime_type(image_bytes, received_mime_type):
        raise HTTPException(
            status_code=415,
            detail="파일 내용과 이미지 형식이 일치하지 않습니다.",
        )

    context_payload = None
    if context:
        try:
            parsed_context = json.loads(context)
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=400, detail="대화 문맥 형식이 올바르지 않습니다.") from error
        if not isinstance(parsed_context, dict):
            raise HTTPException(status_code=400, detail="대화 문맥은 JSON 객체여야 합니다.")
        context_payload = parsed_context

    session_id, state = _load_conversation_state(sessionId, context_payload)
    try:
        with _inference_lock:
            image_description = describe_chat_image(image_bytes, mime_type, message)
            enriched_query = (
                f"{message}\n\n"
                f"첨부 이미지에서 객관적으로 확인된 내용: {image_description}"
            )
            response, next_state = answer_chat(enriched_query, state)
    except (ValueError, RuntimeError, ImportError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return _save_session_and_build_response(session_id, response, next_state)


def _load_conversation_state(
    requested_session_id: str | None,
    context: dict[str, Any] | None,
) -> tuple[str, ConversationState]:
    session_id = requested_session_id or str(uuid.uuid4())
    with _sessions_lock:
        _evict_expired_sessions_locked()
        saved_session = _sessions.get(session_id)
        state = (
            saved_session[1]
            if saved_session is not None
            else deserialize_conversation_state(context)
        )
    return session_id, state


def _save_session_and_build_response(
    session_id: str,
    response: Any,
    next_state: ConversationState,
) -> ChatResponseBody:
    with _sessions_lock:
        _sessions[session_id] = (time.monotonic(), next_state)
        if len(_sessions) > MAX_SESSIONS:
            oldest_session_id = min(_sessions, key=lambda key: _sessions[key][0])
            del _sessions[oldest_session_id]

    return ChatResponseBody(
        sessionId=session_id,
        answer=response.answer,
        domain=response.domain.value,
        confidence=response.confidence,
        context=serialize_conversation_state(next_state),
    )


@app.delete("/chat/{session_id}", status_code=204)
def reset_chat(session_id: str) -> None:
    with _sessions_lock:
        _sessions.pop(session_id, None)


def _verify_sync_key(request_key: str) -> None:
    expected = os.getenv("AI_DATA_KEY", "")
    if not expected or not secrets.compare_digest(expected, request_key):
        raise HTTPException(status_code=401, detail="invalid AI sync key")


@app.post("/internal/indexes/guesthouses/{guesthouse_id}", status_code=204)
def sync_guesthouse(
    guesthouse_id: int,
    x_ai_data_key: str = Header(alias="X-AI-Data-Key"),
) -> Response:
    _verify_sync_key(x_ai_data_key)
    with _inference_lock:
        guesthouse = load_guesthouse_from_backend(guesthouse_id)
        upsert_guesthouse_document(build_guesthouse_document(guesthouse))
    return Response(status_code=204)


@app.delete("/internal/indexes/guesthouses/{guesthouse_id}", status_code=204)
def remove_guesthouse(
    guesthouse_id: int,
    x_ai_data_key: str = Header(alias="X-AI-Data-Key"),
) -> Response:
    _verify_sync_key(x_ai_data_key)
    with _inference_lock:
        delete_guesthouse_document(guesthouse_id)
    return Response(status_code=204)


@app.post("/internal/indexes/staff-recruitments/{recruitment_id}", status_code=204)
def sync_staff_recruitment(
    recruitment_id: int,
    x_ai_data_key: str = Header(alias="X-AI-Data-Key"),
) -> Response:
    _verify_sync_key(x_ai_data_key)
    with _inference_lock:
        recruitment = load_staff_recruitment(recruitment_id)
        upsert_staff_recruitment_document(build_staff_recruitment_document(recruitment))
    return Response(status_code=204)


@app.delete("/internal/indexes/staff-recruitments/{recruitment_id}", status_code=204)
def remove_staff_recruitment(
    recruitment_id: int,
    x_ai_data_key: str = Header(alias="X-AI-Data-Key"),
) -> Response:
    _verify_sync_key(x_ai_data_key)
    with _inference_lock:
        delete_staff_recruitment_document(recruitment_id)
    return Response(status_code=204)


def _reconciliation_loop() -> None:
    initial_delay = max(0, int(os.getenv("INDEX_RECONCILE_START_DELAY_SECONDS", "60")))
    interval = max(300, int(os.getenv("INDEX_RECONCILE_INTERVAL_SECONDS", "21600")))
    if _reconciliation_stop.wait(initial_delay):
        return
    while not _reconciliation_stop.is_set():
        try:
            with _inference_lock:
                result = reconcile_dynamic_indexes()
            _reconciliation_status.update({
                "lastSuccessAt": int(time.time()),
                "lastResult": result,
                "lastError": None,
            })
            logger.info("AI index reconciliation completed: %s", result)
        except Exception as error:  # background repair must not stop chat traffic
            _reconciliation_status["lastError"] = str(error)
            logger.exception("AI index reconciliation failed")
        if _reconciliation_stop.wait(interval):
            return


@app.on_event("startup")
def start_index_reconciliation() -> None:
    global _reconciliation_thread
    if os.getenv("INDEX_RECONCILE_ENABLED", "true").casefold() not in {"true", "1", "yes"}:
        return
    _reconciliation_stop.clear()
    _reconciliation_thread = threading.Thread(
        target=_reconciliation_loop,
        name="ai-index-reconciler",
        daemon=True,
    )
    _reconciliation_thread.start()


@app.on_event("shutdown")
def stop_index_reconciliation() -> None:
    _reconciliation_stop.set()
    if _reconciliation_thread is not None:
        _reconciliation_thread.join(timeout=5)
