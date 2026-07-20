from __future__ import annotations

import os
import threading
import time
import uuid
import secrets

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.chat.orchestrator import answer_chat
from src.chat.schemas import ConversationState
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


load_dotenv(dotenv_path=".env")

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

# session_id -> (마지막 접근 시각, 대화 상태). FE가 명시적으로 reset을 호출하지 않아도
# 오래된 세션은 아래 _evict_expired_sessions_locked에서 자동으로 제거되어 메모리가 무한히 쌓이지 않는다.
_sessions: dict[str, tuple[float, ConversationState]] = {}
_sessions_lock = threading.Lock()
_inference_lock = threading.Lock()


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


class ChatResponseBody(BaseModel):
    sessionId: str
    answer: str
    domain: str
    confidence: float


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
    }


@app.get("/ready")
def ready() -> dict:
    status = health()
    if status["status"] != "ok" or not status["geminiConfigured"]:
        raise HTTPException(status_code=503, detail=status)
    return status


@app.post("/chat", response_model=ChatResponseBody)
def chat(request: ChatRequest) -> ChatResponseBody:
    session_id = request.sessionId or str(uuid.uuid4())
    with _sessions_lock:
        _evict_expired_sessions_locked()
        _, state = _sessions.get(session_id, (0.0, ConversationState()))

    try:
        # BGE-M3 is shared in-process; serialize inference to avoid model races.
        with _inference_lock:
            response, next_state = answer_chat(request.message, state)
    except (ValueError, RuntimeError, ImportError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

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
