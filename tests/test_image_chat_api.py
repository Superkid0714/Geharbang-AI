from fastapi.testclient import TestClient

import src.api.main as api
from src.chat.schemas import ChatDomain, ChatResponse


def test_image_chat_accepts_multipart_and_returns_session(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "describe_chat_image",
        lambda image_bytes, mime_type, user_query: "바다가 보이는 조용한 숙소 사진",
    )
    monkeypatch.setattr(
        api,
        "answer_chat",
        lambda query, state: (
            ChatResponse("이미지 답변", ChatDomain.GUESTHOUSE, 0.9),
            state,
        ),
    )

    response = TestClient(api.app).post(
        "/chat/image",
        data={"message": "이런 분위기 게하 추천해줘"},
        files={"image": ("test.jpg", b"\xff\xd8\xff\xe0image-data", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "이미지 답변"
    assert response.json()["domain"] == "guesthouse"
    assert response.json()["sessionId"]


def test_image_chat_rejects_unsupported_file() -> None:
    response = TestClient(api.app).post(
        "/chat/image",
        data={"message": "이 파일은 뭐야?"},
        files={"image": ("test.txt", b"not-image", "text/plain")},
    )

    assert response.status_code == 415


def test_image_chat_rejects_spoofed_content_type() -> None:
    response = TestClient(api.app).post(
        "/chat/image",
        data={"message": "이 사진은 뭐야?"},
        files={"image": ("fake.jpg", b"not-an-image", "image/jpeg")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "파일 내용과 이미지 형식이 일치하지 않습니다."
