from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BACKEND_BASE_URL = "http://localhost:8080"


def load_staff_recruitments(
    backend_base_url: str | None = None,
    ai_data_key: str | None = None,
    timeout: float = 15.0,
) -> list[dict]:
    """Load active staff recruitments from the backend's protected AI endpoint."""
    base_url = (backend_base_url or os.getenv("BACKEND_BASE_URL", DEFAULT_BACKEND_BASE_URL)).rstrip("/")
    data_key = ai_data_key or os.getenv("AI_DATA_KEY")
    if not data_key:
        raise ValueError("AI_DATA_KEY가 설정되어 있지 않습니다.")

    request = Request(
        f"{base_url}/api/v1/staff-recruitment/internal/ai-data",
        headers={"Accept": "application/json", "X-AI-Data-Key": data_key},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"스텝 공고 API가 HTTP {error.code}를 반환했습니다.") from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"백엔드 스텝 공고 API에 연결할 수 없습니다: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError("스텝 공고 API 응답이 올바른 JSON이 아닙니다.") from error

    return validate_staff_recruitments_payload(payload)


def load_staff_recruitment(
    recruitment_id: int,
    backend_base_url: str | None = None,
    ai_data_key: str | None = None,
    timeout: float = 15.0,
) -> dict:
    """Load one active staff recruitment for incremental synchronization."""
    if recruitment_id <= 0:
        raise ValueError("recruitment_id must be positive")
    base_url = (backend_base_url or os.getenv("BACKEND_BASE_URL", DEFAULT_BACKEND_BASE_URL)).rstrip("/")
    data_key = ai_data_key or os.getenv("AI_DATA_KEY")
    if not data_key:
        raise ValueError("AI_DATA_KEY가 설정되어 있지 않습니다.")
    request = Request(
        f"{base_url}/api/v1/staff-recruitment/internal/ai-data/{recruitment_id}",
        headers={"Accept": "application/json", "X-AI-Data-Key": data_key},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            item = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"스텝 공고 API가 HTTP {error.code}를 반환했습니다.") from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"백엔드 스텝 공고 API에 연결할 수 없습니다: {error}") from error
    return validate_staff_recruitments_payload({"staffRecruitments": [item]})[0]


def validate_staff_recruitments_payload(payload: Any) -> list[dict]:
    if not isinstance(payload, dict):
        raise ValueError("staff recruitment payload must be an object")
    items = payload.get("staffRecruitments")
    if not isinstance(items, list):
        raise ValueError("staffRecruitments must be a list")

    validated: list[dict] = []
    for index, item in enumerate(items):
        label = f"staffRecruitments[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")
        recruitment_id = item.get("id")
        details = item.get("details")
        if not isinstance(recruitment_id, int) or isinstance(recruitment_id, bool):
            raise ValueError(f"{label}.id must be an integer")
        if not isinstance(details, dict):
            raise ValueError(f"{label}.details must be an object")
        for field in ("title", "guestHouseName", "region"):
            value = details.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label}.details.{field} must not be empty")
        validated.append({"id": recruitment_id, "details": details})

    return validated
