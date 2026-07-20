from __future__ import annotations

from typing import Any


def build_staff_recruitment_documents(recruitments: list[dict]) -> list[dict]:
    return [build_staff_recruitment_document(item) for item in recruitments]


def build_staff_recruitment_document(item: dict) -> dict:
    recruitment_id = item["id"]
    details = item["details"]
    title = details["title"]
    guesthouse_name = details["guestHouseName"]
    working = details.get("workingInformation") or {}
    feature = details.get("feature") or {}
    parts = [
        f"스텝 구인 공고 제목은 {title}입니다.",
        f"게스트하우스 이름은 {guesthouse_name}입니다.",
        f"지역은 {details['region']}입니다.",
    ]

    location = details.get("location") or {}
    _append(parts, "주소", location.get("address"))

    _append(parts, "근무 시작일", working.get("startDate"))
    if working.get("isStartDateNegotiable") is True:
        parts.append("근무 시작일은 협의할 수 있습니다.")
    _append(parts, "근무 기간", working.get("workingPeriod"))
    for job in working.get("jobs") or []:
        if isinstance(job, dict):
            parts.append(_build_job_sentence(job))

    introduction = details.get("introduction") or {}
    _append(parts, "공고 소개", introduction.get("content"))

    _append(parts, "지원 성별 조건", feature.get("gender"))
    _append_list(parts, "우대 사항", feature.get("advantages"))
    _append_list(parts, "제공 혜택과 복지", feature.get("employeeBenefits"))
    _append(parts, "사장님 메시지", details.get("ownerMessage"))

    return {
        "staffRecruitmentId": recruitment_id,
        "title": title,
        "guestHouseName": guesthouse_name,
        "region": details["region"],
        "workingPeriod": working.get("workingPeriod") or "",
        "gender": feature.get("gender") or "무관",
        "content": "\n".join(part for part in parts if part),
    }


def _build_job_sentence(job: dict) -> str:
    values: list[str] = []
    fields = [
        ("업무명", job.get("name")),
        ("업무 내용", job.get("job")),
        ("시작 시간", job.get("startTIme") or job.get("startTime")),
        ("종료 시간", job.get("endTime")),
        ("근무 형태", job.get("workType")),
        ("근무일", _day_value(job.get("workDays"))),
        ("휴무일", _day_value(job.get("restDays"))),
        ("근무 주기", job.get("weeklyWorkingDays")),
    ]
    for label, value in fields:
        if _has_value(value):
            values.append(f"{label}: {value}")
    return "업무 조건: " + ", ".join(values) + "."


def _append(parts: list[str], label: str, value: Any) -> None:
    if _has_value(value):
        parts.append(f"{label}: {value}")


def _append_list(parts: list[str], label: str, value: Any) -> None:
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if _has_value(item)]
        if cleaned:
            parts.append(f"{label}: {', '.join(cleaned)}")


def _day_value(value: Any) -> str | None:
    return f"주 {value}일" if isinstance(value, int) else None


def _has_value(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))
