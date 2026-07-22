from __future__ import annotations

import copy
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)


REGION_KEYWORDS = {
    "애월": ["애월", "협재", "한림", "금능", "곽지", "서쪽", "제주 서쪽"],
    "성산_구좌": ["성산", "구좌", "구좌읍", "월정리", "세화", "하도", "종달", "송당", "동쪽", "제주 동쪽"],
    "우도_기타": ["우도", "도서지역", "추자", "비양도", "기타"],
    "중문": ["중문", "대정", "중문관광단지", "중문 관광단지"],
    "서귀포시": ["서귀포", "서귀포시", "남원", "표선"],
    "제주시": ["제주시", "제주 시내", "제주공항", "제주공항 근처", "공항 근처", "시내"],
}

ADDRESS_REGION_KEYWORDS = {
    "애월": ["애월", "협재", "한림", "금능", "곽지", "서쪽", "제주 서쪽"],
    "성산_구좌": ["성산", "구좌", "구좌읍", "월정리", "세화", "하도", "종달", "송당", "동쪽", "제주 동쪽"],
    "우도_기타": ["우도", "도서지역", "추자", "비양도", "기타"],
    "중문": ["중문", "대정", "중문관광단지", "중문 관광단지"],
    "서귀포시": ["서귀포", "서귀포시", "남원", "남원읍", "표선"],
    "제주시": ["제주시", "제주 시내", "제주공항", "제주공항 근처", "공항 근처", "시내"],
}

ROOM_TYPE_KEYWORDS = {
    "여성전용": ["여성전용", "여성 전용", "여성 도미토리", "여자 도미토리", "여자방", "여성 객실"],
    "남성전용": ["남성전용", "남성 전용", "남성 도미토리", "남자 도미토리", "남자방", "남성 객실"],
    "기타": ["혼성", "혼성 도미토리", "남녀공용", "남녀 공용"],
}

PARTY_TYPE_KEYWORDS = {
    "술파티": ["술파티", "술 파티", "술 마시는 파티"],
    "포틀럭": ["포틀럭", "포트럭"],
    "디너_파티": ["디너파티", "디너 파티", "저녁 파티", "밥 파티"],
    "클럽_파티": ["클럽파티", "클럽 파티"],
    "기타": ["티타임", "차 마시는 모임", "불멍", "기타 파티"],
}

HAS_PARTY_KEYWORDS = ["파티 있는", "파티 있는 곳", "파티 게하", "파티 가능한 곳", "파티하는 곳", "게하 파티"]
NO_PARTY_KEYWORDS = ["파티 없는", "파티 안 하는", "파티x", "파티 x", "파티 싫은", "파티 없는 곳", "조용히 쉬는 곳"]
EXTERNAL_GUEST_KEYWORDS = [
    "외부인 가능",
    "숙박 안 해도 파티 가능",
    "파티만 참여 가능",
    "외부인 파티 가능",
    "숙박 없이 파티",
]

MOOD_KEYWORDS = {
    "바닷가": ["바닷가", "바다 근처", "해변 근처", "오션뷰"],
    "동물": ["동물", "강아지", "고양이", "반려동물"],
    "자연_숲": ["자연", "숲", "초록", "힐링"],
    "대규모파티": ["대규모 파티", "큰 파티", "대형 파티"],
    "소규모파티": ["소규모 파티", "작은 파티", "소수 파티"],
    "조용한": ["조용", "차분한", "한적한"],
    "활발한": ["활발한", "활기찬", "사교적인"],
    "감성_느좋": ["감성", "느좋", "분위기 좋은"],
    "파티_X": ["파티 없는", "파티 안 하는", "파티x", "파티 x"],
    "솔로": ["혼자 여행", "혼행", "솔로"],
    "한달살이": ["한달살이", "한 달 살기", "장기 숙박"],
}

AMENITY_KEYWORDS = {
    "수영장": ["수영장", "풀장"],
    "주차": ["주차"],
    "조식": ["조식", "아침 제공"],
    "무료 Wi-Fi": ["무료 와이파이", "무료 wi-fi", "무료 wifi", "와이파이"],
    "바비큐": ["바비큐", "바베큐"],
}


def extract_structured_conditions(query: str) -> dict[str, Any]:
    """사용자 질문에서 명확한 정형 조건만 추출합니다."""
    normalized_query = _normalize_text(query)
    conditions: dict[str, Any] = {}

    # region -> guesthouse["region"] + location.*Address 보완 판정
    region = _find_by_keywords(normalized_query, REGION_KEYWORDS)
    if region:
        conditions["region"] = region

    moods = [
        mood
        for mood, keywords in MOOD_KEYWORDS.items()
        if any(_normalize_text(keyword) in normalized_query for keyword in keywords)
    ]
    if moods:
        conditions["moods"] = moods

    amenities = [
        amenity
        for amenity, keywords in AMENITY_KEYWORDS.items()
        if any(_normalize_text(keyword) in normalized_query for keyword in keywords)
    ]
    if amenities:
        conditions["amenities"] = amenities

    minimum_rating = _extract_minimum_rating(normalized_query)
    if minimum_rating is not None:
        conditions["minAverageRating"] = minimum_rating

    # roomType -> rooms[].type
    room_type = _find_by_keywords(normalized_query, ROOM_TYPE_KEYWORDS)
    if room_type:
        conditions["roomType"] = room_type

    # headCount -> rooms[].headCount
    head_count = _extract_head_count(normalized_query)
    if head_count is not None:
        conditions["headCount"] = head_count

    # maxPricePerNight -> rooms[].pricePerNight, 파티 가격 표현과 분리
    max_price = _extract_room_max_price(normalized_query)
    if max_price is not None:
        conditions["maxPricePerNight"] = max_price

    # hasParties -> parties
    if any(keyword in normalized_query for keyword in NO_PARTY_KEYWORDS):
        conditions["hasParties"] = False
    elif any(keyword in normalized_query for keyword in HAS_PARTY_KEYWORDS):
        conditions["hasParties"] = True

    # partyType -> parties[].type 또는 parties[].otherPartyType
    party_type = _find_by_keywords(normalized_query, PARTY_TYPE_KEYWORDS)
    if party_type:
        conditions["partyType"] = party_type

    # isExternalGuestAllowed -> parties[].isExternalGuestAllowed
    if any(keyword in normalized_query for keyword in EXTERNAL_GUEST_KEYWORDS):
        conditions["isExternalGuestAllowed"] = True

    # partyGuestFee -> parties[].guestFee
    party_guest_fee = _extract_party_guest_fee(normalized_query)
    if party_guest_fee is not None:
        conditions["partyGuestFee"] = party_guest_fee

    # partyExternalGuestFee -> parties[].externalGuestFee
    party_external_guest_fee = _extract_party_external_guest_fee(normalized_query)
    if party_external_guest_fee is not None:
        conditions["partyExternalGuestFee"] = party_external_guest_fee

    logger.debug("Extracted guesthouse conditions: %s", conditions)
    return conditions


def filter_guesthouses(guesthouses: list[dict], conditions: dict) -> list[dict]:
    """정형 조건으로 게스트하우스를 1차 필터링합니다."""
    strong_conditions = {key: value for key, value in conditions.items() if value not in (None, [], "")}

    if not strong_conditions:
        results = [_with_filter_fields(guesthouse) for guesthouse in guesthouses]
        logger.debug("Filtered guesthouse candidates: %d / %d", len(results), len(guesthouses))
        return results

    filtered: list[dict] = []

    for guesthouse in guesthouses:
        notes: list[str] = []

        if "region" in strong_conditions and not _matches_region(guesthouse, strong_conditions["region"]):
            continue

        if "moods" in strong_conditions and not _contains_all_values(
            guesthouse.get("moods", []), strong_conditions["moods"]
        ):
            continue

        if "amenities" in strong_conditions and not _contains_all_values(
            guesthouse.get("amenities", []), strong_conditions["amenities"], allow_partial=True
        ):
            continue

        if "minAverageRating" in strong_conditions:
            average_rating = guesthouse.get("averageRating")
            review_count = guesthouse.get("reviewCount")
            if (
                not isinstance(average_rating, (int, float))
                or not isinstance(review_count, int)
                or review_count <= 0
                or average_rating < strong_conditions["minAverageRating"]
            ):
                continue

        matched_rooms = _filter_rooms(guesthouse.get("rooms", []), strong_conditions, notes)
        if _has_room_conditions(strong_conditions) and not matched_rooms:
            continue

        parties = guesthouse.get("parties", [])
        if strong_conditions.get("hasParties") is False:
            if parties:
                continue
            matched_parties = []
        else:
            matched_parties = _filter_parties(parties, strong_conditions)
            if _has_party_conditions(strong_conditions) and not matched_parties:
                continue

        result = copy.deepcopy(guesthouse)
        result["matchedRooms"] = matched_rooms if _has_room_conditions(strong_conditions) else copy.deepcopy(guesthouse.get("rooms", []))
        result["matchedParties"] = matched_parties if _has_party_conditions(strong_conditions) else copy.deepcopy(guesthouse.get("parties", []))
        result["filterNotes"] = notes
        filtered.append(result)

    logger.debug("Filtered guesthouse candidates: %d / %d", len(filtered), len(guesthouses))
    return filtered


def normalize_room_type(value: Any) -> str:
    text = _normalize_text(str(value or ""))
    if any(keyword in text for keyword in ["여성전용", "여성 전용", "여성", "여자"]):
        return "여성전용"
    if any(keyword in text for keyword in ["남성전용", "남성 전용", "남성", "남자"]):
        return "남성전용"
    return "기타"


def normalize_party_type(value: Any) -> str:
    text = _normalize_text(str(value or ""))
    compact = text.replace("_", "").replace(" ", "")
    if "술파티" in compact:
        return "술파티"
    if "포틀럭" in compact or "포트럭" in compact:
        return "포틀럭"
    if "디너파티" in compact or "저녁파티" in compact or "밥파티" in compact:
        return "디너_파티"
    if "클럽파티" in compact:
        return "클럽_파티"
    if any(keyword in compact for keyword in ["티타임", "차마시는모임", "불멍", "기타파티"]):
        return "기타"
    return "기타"


def _filter_rooms(rooms: list[dict], conditions: dict, notes: list[str]) -> list[dict]:
    matched: list[dict] = []

    for room in rooms:
        if "roomType" in conditions and normalize_room_type(room.get("type")) != conditions["roomType"]:
            continue

        if "headCount" in conditions:
            room_head_count = _get_room_head_count(room)
            if room_head_count is None:
                _append_note(notes, "객실 인원 정보 확인 필요")
                continue
            if room_head_count != conditions["headCount"]:
                continue

        if "maxPricePerNight" in conditions:
            price = room.get("pricePerNight")
            if not isinstance(price, (int, float)):
                _append_note(notes, "객실 가격 정보 확인 필요")
                continue
            if not _is_room_price_comparable(room, conditions):
                _append_note(notes, "일부 객실은 1인 기준/객실 기준이 불명확해 가격 필터에서 제외됨")
                continue
            if price > conditions["maxPricePerNight"]:
                continue

        matched.append(copy.deepcopy(room))

    return matched


def _filter_parties(parties: list[dict], conditions: dict) -> list[dict]:
    matched: list[dict] = []

    for party in parties:
        if conditions.get("hasParties") is True:
            pass

        if "partyType" in conditions:
            normalized_types = {
                normalize_party_type(party.get("type")),
                normalize_party_type(party.get("otherPartyType")),
            }
            if conditions["partyType"] not in normalized_types:
                continue

        if "isExternalGuestAllowed" in conditions and party.get("isExternalGuestAllowed") is not True:
            continue

        if "partyGuestFee" in conditions:
            guest_fee = party.get("guestFee")
            if guest_fee is None or guest_fee > conditions["partyGuestFee"]:
                continue

        if "partyExternalGuestFee" in conditions:
            external_fee = party.get("externalGuestFee")
            if external_fee is None or external_fee > conditions["partyExternalGuestFee"]:
                continue

        matched.append(copy.deepcopy(party))

    return matched


def _with_filter_fields(guesthouse: dict) -> dict:
    result = copy.deepcopy(guesthouse)
    result["matchedRooms"] = copy.deepcopy(guesthouse.get("rooms", []))
    result["matchedParties"] = copy.deepcopy(guesthouse.get("parties", []))
    result["filterNotes"] = []
    return result


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _find_by_keywords(query: str, mapping: dict[str, list[str]]) -> str | None:
    for standard_value, keywords in mapping.items():
        if any(_normalize_text(keyword) in query for keyword in keywords):
            return standard_value
    return None


def _append_note(notes: list[str], note: str) -> None:
    if note not in notes:
        notes.append(note)


def _extract_head_count(query: str) -> int | None:
    if any(keyword in query for keyword in ["다인실", "도미토리"]):
        exact_match = re.search(r"(\d+)\s*(?:인실|명)", query)
        return int(exact_match.group(1)) if exact_match else None

    if any(keyword in query for keyword in ["혼자 방", "개인실", "1인실"]):
        return 1
    if any(keyword in query for keyword in ["둘이", "2인실", "2명"]):
        return 2

    match = re.search(r"(\d+)\s*인실", query)
    if match:
        return int(match.group(1))

    return None


def _extract_room_max_price(query: str) -> int | None:
    for match in re.finditer(r"(\d+)\s*만원대", query):
        if not _near_party_expression(query, match.start(), match.end()):
            return int(match.group(1)) * 10000 + 9999

    for match in re.finditer(r"(\d+)\s*만원\s*이하", query):
        if not _near_party_expression(query, match.start(), match.end()):
            return int(match.group(1)) * 10000

    for match in re.finditer(r"(\d[\d,]*)\s*원\s*이하", query):
        if not _near_party_expression(query, match.start(), match.end()):
            return int(match.group(1).replace(",", ""))

    return None


def _extract_minimum_rating(query: str) -> float | None:
    match = re.search(r"(?:평점|별점)\s*(\d(?:\.\d+)?)\s*점?\s*이상", query)
    if not match:
        return None
    rating = float(match.group(1))
    return rating if 0 <= rating <= 5 else None


def _extract_party_guest_fee(query: str) -> int | None:
    if "파티 무료" in query:
        return 0

    match = _search_non_external_party_fee(query, r"(?:파티비|파티 가격)\s*(\d+)\s*만원\s*이하")
    if match:
        return int(match.group(1)) * 10000

    match = _search_non_external_party_fee(query, r"(?:파티비|파티 가격)\s*(\d[\d,]*)\s*원\s*이하")
    if match:
        return int(match.group(1).replace(",", ""))

    return None


def _search_non_external_party_fee(query: str, pattern: str) -> re.Match[str] | None:
    for match in re.finditer(pattern, query):
        before = query[max(0, match.start() - 8) : match.start()]
        if "외부인" not in before:
            return match
    return None


def _extract_party_external_guest_fee(query: str) -> int | None:
    match = re.search(r"외부인\s*(?:파티비|참가비|파티 가격)\s*(\d+)\s*만원\s*이하", query)
    if match:
        return int(match.group(1)) * 10000

    match = re.search(r"외부인\s*(?:파티비|참가비|파티 가격)\s*(\d[\d,]*)\s*원\s*이하", query)
    if match:
        return int(match.group(1).replace(",", ""))

    return None


def _near_party_expression(query: str, start: int, end: int) -> bool:
    window = query[max(0, start - 8) : min(len(query), end + 4)]
    return any(keyword in window for keyword in ["파티", "파티비", "참가비", "외부인"])


def _matches_region(guesthouse: dict, expected_region: str) -> bool:
    region_candidates = {
        guesthouse.get("region"),
        _infer_region_from_location(guesthouse.get("location", {})),
    }
    normalized_candidates = {_normalize_region(region) for region in region_candidates if region}
    return expected_region in normalized_candidates


def _infer_region_from_location(location: dict) -> str | None:
    address_text = _normalize_text(
        " ".join(
            str(location.get(field, ""))
            for field in ["roadNameAddress", "lotNumberAddress"]
        )
    )
    return _find_by_keywords(address_text, ADDRESS_REGION_KEYWORDS)


def _normalize_region(value: Any) -> str:
    text = _normalize_text(str(value or ""))
    if text in ["제주시", "제주"]:
        return "제주시"
    if text in ["애월", "애월·협재"]:
        return "애월"
    if text in ["서귀포", "서귀포시"]:
        return "서귀포시"
    if text in ["성산·우도", "성산·구좌", "성산_구좌"]:
        return "성산_구좌"
    if text in ["우도", "우도·기타", "우도_기타", "기타"]:
        return "우도_기타"
    inferred = _find_by_keywords(text, REGION_KEYWORDS)
    return inferred or "우도_기타"


def _get_room_head_count(room: dict) -> int | None:
    value = room.get("headCount")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group())
    return None


def _is_room_price_comparable(room: dict, conditions: dict) -> bool:
    if "headCount" in conditions:
        return True

    room_text = _normalize_text(f"{room.get('name', '')} {room.get('type', '')}")
    return "도미토리" in room_text


def _has_room_conditions(conditions: dict) -> bool:
    return any(key in conditions for key in ["roomType", "headCount", "maxPricePerNight"])


def _has_party_conditions(conditions: dict) -> bool:
    return any(
        key in conditions
        for key in ["hasParties", "partyType", "isExternalGuestAllowed", "partyGuestFee", "partyExternalGuestFee"]
    )


def _contains_all_values(actual_values: Any, expected_values: list[str], allow_partial: bool = False) -> bool:
    if not isinstance(actual_values, (list, set, tuple)):
        return False
    normalized_actual = [_normalize_text(str(value)).replace("_", " ") for value in actual_values]
    for expected in expected_values:
        normalized_expected = _normalize_text(expected).replace("_", " ")
        if allow_partial:
            if not any(
                normalized_expected in actual or actual in normalized_expected
                for actual in normalized_actual
            ):
                return False
        elif normalized_expected not in normalized_actual:
            return False
    return True
