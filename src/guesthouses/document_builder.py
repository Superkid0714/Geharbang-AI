from __future__ import annotations

from typing import Any


MOOD_SEARCH_PHRASES = {
    "바닷가": "바다 근처, 해변 분위기, 바닷가 감성을 원하는 여행자에게 어울릴 수 있습니다.",
    "동물": "동물을 좋아하거나 동물이 있는 분위기를 선호하는 여행자에게 어울릴 수 있습니다.",
    "자연_숲": "자연, 숲, 조용한 풍경, 초록 분위기, 힐링 여행을 원하는 여행자에게 어울릴 수 있습니다.",
    "대규모파티": "여러 사람과 활발하게 어울리고 싶은 여행자, 큰 규모의 파티 분위기를 원하는 사람에게 어울립니다.",
    "소규모파티": "조용하게 대화하며 친해지는 소규모 모임, 부담 없는 친목 분위기를 원하는 여행자에게 어울립니다.",
    "조용한": "시끄럽지 않은 숙소, 차분한 휴식, 조용한 분위기를 원하는 여행자에게 어울립니다.",
    "활발한": "활발하게 사람들과 어울리고 새로운 친구를 만들고 싶은 여행자에게 어울립니다.",
    "감성_느좋": "감성적인 분위기, 느낌 좋은 숙소, 분위기 좋은 게스트하우스를 찾는 여행자에게 어울립니다.",
    "파티X": "파티가 없는 숙소, 술자리나 시끄러운 모임 없이 조용히 쉬고 싶은 여행자에게 어울립니다.",
    "파티_X": "파티가 없는 숙소, 술자리나 시끄러운 모임 없이 조용히 쉬고 싶은 여행자에게 어울립니다.",
    "솔로": "혼자 여행 온 사람, 혼자 온 게스트, 솔로 여행자도 어색하지 않게 머물기 좋은 숙소입니다.",
    "한달살이": "장기 숙박, 제주 한달살이, 오래 머무는 여행자에게 어울릴 수 있습니다.",
}


def build_guesthouse_document(guesthouse: dict, index: int | None = None) -> dict:
    """Build one natural-language RAG document from a guesthouse dict."""
    guesthouse_id = _get_guesthouse_id(guesthouse, index)
    guesthouse_name = guesthouse["guestHouseName"]

    content_parts: list[str] = []
    _append_basic_info(content_parts, guesthouse)
    _append_introduction(content_parts, guesthouse)
    _append_moods(content_parts, guesthouse)
    _append_amenities(content_parts, guesthouse)
    _append_rooms(content_parts, guesthouse)
    _append_parties(content_parts, guesthouse)
    _append_review_summary(content_parts, guesthouse)
    _append_owner_message(content_parts, guesthouse)

    return {
        "guestHouseId": guesthouse_id,
        "guestHouseName": guesthouse_name,
        "averageRating": guesthouse.get("averageRating", 0),
        "reviewCount": guesthouse.get("reviewCount", 0),
        "content": "\n".join(part for part in content_parts if part),
    }


def build_guesthouse_documents(guesthouses: list[dict]) -> list[dict]:
    """Build natural-language RAG documents for all guesthouses."""
    return [
        build_guesthouse_document(guesthouse, index)
        for index, guesthouse in enumerate(guesthouses)
    ]


def _get_guesthouse_id(guesthouse: dict, index: int | None) -> Any:
    if _has_value(guesthouse.get("guestHouseId")):
        return guesthouse["guestHouseId"]
    raise ValueError("guestHouseId is required")


def _append_basic_info(parts: list[str], guesthouse: dict) -> None:
    # Uses guestHouseName, region, and location.lotNumberAddress only.
    name = guesthouse.get("guestHouseName")
    region = guesthouse.get("region")
    lot_number_address = guesthouse.get("location", {}).get("lotNumberAddress")

    sentences = []
    if _has_value(name):
        sentences.append(f"게스트하우스 이름은 {name}입니다.")
    if _has_value(region):
        sentences.append(f"지역은 {region}입니다.")
    if _has_value(lot_number_address):
        sentences.append(f"주소는 {lot_number_address}입니다.")

    if sentences:
        parts.append(" ".join(sentences))


def _append_introduction(parts: list[str], guesthouse: dict) -> None:
    introduction = guesthouse.get("introduction")
    if _has_value(introduction):
        parts.append(f"소개: {introduction}")


def _append_moods(parts: list[str], guesthouse: dict) -> None:
    moods = _clean_list(guesthouse.get("moods"))
    if not moods:
        return

    parts.append(f"분위기 키워드는 {_join_values(moods)}입니다.")

    search_phrases = [
        MOOD_SEARCH_PHRASES[mood]
        for mood in moods
        if mood in MOOD_SEARCH_PHRASES
    ]
    if search_phrases:
        parts.append(" ".join(search_phrases))


def _append_amenities(parts: list[str], guesthouse: dict) -> None:
    amenities = _clean_list(guesthouse.get("amenities"))
    if amenities:
        parts.append(f"편의시설은 {_join_values(amenities)}입니다.")


def _append_rooms(parts: list[str], guesthouse: dict) -> None:
    rooms = guesthouse.get("rooms") or []
    room_sentences: list[str] = []
    has_price = False

    for room in rooms:
        if not isinstance(room, dict):
            continue

        details = []
        _append_labeled_value(details, "객실명", room.get("name"))
        _append_labeled_value(details, "객실 타입", room.get("type"))
        _append_labeled_value(details, "인원", _format_head_count(room.get("headCount")))
        _append_labeled_value(details, "체크인", room.get("checkInTime"))
        _append_labeled_value(details, "체크아웃", room.get("checkOutTime"))

        price = room.get("pricePerNight")
        if _has_value(price):
            _append_labeled_value(details, "1박 가격", _format_price(price))
            has_price = True

        if details:
            room_sentences.append("객실 정보: " + ", ".join(details) + ".")

    if room_sentences:
        parts.append("\n".join(room_sentences))

    if has_price:
        parts.append("가격은 등록된 정보 기준이며 실제 예약 전 확인이 필요합니다.")


def _append_parties(parts: list[str], guesthouse: dict) -> None:
    parties = guesthouse.get("parties") or []
    if not parties:
        parts.append("등록된 파티 정보는 없습니다.")
        return

    party_sentences: list[str] = []
    for party in parties:
        if not isinstance(party, dict):
            continue

        details = []
        _append_labeled_value(details, "파티 타입", party.get("type"))
        _append_labeled_value(details, "기타 파티 타입", party.get("otherPartyType"))
        _append_labeled_value(details, "시작 시간", party.get("startTime"))
        _append_labeled_value(details, "종료 시간", party.get("endTime"))
        _append_labeled_value(details, "운영 요일", _join_values(_clean_list(party.get("weeklyDays"))))
        _append_labeled_value(details, "장소", party.get("place"))
        _append_labeled_value(details, "파티 분위기", _join_values(_clean_list(party.get("moods"))))

        if isinstance(party.get("isExternalGuestAllowed"), bool):
            if party["isExternalGuestAllowed"]:
                details.append("외부인 파티 참여가 가능합니다")
            else:
                details.append("외부인 파티 참여는 불가능합니다")

        if party.get("guestFee") is not None:
            _append_labeled_value(details, "숙박객 파티 참가비", _format_price(party.get("guestFee")))
        if party.get("externalGuestFee") is not None:
            _append_labeled_value(details, "외부인 파티 참가비", _format_price(party.get("externalGuestFee")))

        _append_labeled_value(details, "파티 안내", party.get("information"))

        if details:
            party_sentences.append("파티 정보: " + ", ".join(details) + ".")

    if party_sentences:
        parts.append("\n".join(party_sentences))
    else:
        parts.append("등록된 파티 정보는 없습니다.")


def _append_owner_message(parts: list[str], guesthouse: dict) -> None:
    owner_message = guesthouse.get("ownerMessage")
    if _has_value(owner_message):
        parts.append(f"사장님 메시지: {owner_message}")


def _append_review_summary(parts: list[str], guesthouse: dict) -> None:
    review_count = guesthouse.get("reviewCount")
    average_rating = guesthouse.get("averageRating")
    if isinstance(review_count, int) and review_count > 0 and isinstance(average_rating, (int, float)):
        parts.append(f"이용자 리뷰는 {review_count}개이고 평균 평점은 5점 만점에 {average_rating}점입니다.")


def _append_labeled_value(details: list[str], label: str, value: Any) -> None:
    if _has_value(value):
        details.append(f"{label}: {value}")


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if _has_value(item)]


def _join_values(values: list[str]) -> str:
    return ", ".join(values)


def _format_head_count(value: Any) -> str | None:
    if not _has_value(value):
        return None
    return f"{value}명"


def _format_price(value: Any) -> str | None:
    if not _has_value(value):
        return None
    return f"{value}원"


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value)
    return True
