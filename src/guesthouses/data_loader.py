from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def load_guesthouses(file_path: str = "data/guesthouses.json") -> list[dict]:
    """Read and validate guesthouse JSON data, then return a deep copy."""
    path = Path(file_path)

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {file_path}: {error}") from error
    except OSError as error:
        raise ValueError(f"failed to read {file_path}: {error}") from error

    if not isinstance(data, list):
        raise ValueError("guesthouses root must be a list")

    for index, guesthouse in enumerate(data):
        if not isinstance(guesthouse, dict):
            raise ValueError(f"[guesthouse index {index}] must be a dict")
        _validate_guesthouse(guesthouse, index)

    return copy.deepcopy(data)


def _validate_guesthouse(guesthouse: dict, index: int) -> None:
    name = guesthouse.get("guestHouseName")
    label = _guesthouse_label(guesthouse, index)

    # guestHouseName -> guesthouse["guestHouseName"]
    _require_non_empty_str(guesthouse, "guestHouseName", f"[guesthouse index {index}]")
    label = f"[{name}]"

    # region -> guesthouse["region"]
    _require_non_empty_str(guesthouse, "region", label)

    # location -> guesthouse["location"]
    location = _require_dict(guesthouse, "location", label)
    _validate_location(location, label)

    # imageUrls -> guesthouse["imageUrls"]
    _validate_image_urls(_require_field(guesthouse, "imageUrls", label), f"{label} imageUrls")

    # introduction -> guesthouse["introduction"]
    _require_non_empty_str(guesthouse, "introduction", label)

    # amenities -> guesthouse["amenities"]
    _require_list(guesthouse, "amenities", label)

    # moods -> guesthouse["moods"]
    _require_list(guesthouse, "moods", label)

    # rooms -> guesthouse["rooms"]
    rooms = _require_list(guesthouse, "rooms", label)
    if not rooms:
        raise ValueError(f"{label} rooms must not be empty")
    for room_index, room in enumerate(rooms):
        _validate_room(room, room_index, label)

    # parties -> guesthouse["parties"]
    parties = _require_list(guesthouse, "parties", label)
    for party_index, party in enumerate(parties):
        _validate_party(party, party_index, label)

    # contact -> guesthouse["contact"]
    contact = _require_dict(guesthouse, "contact", label)
    _validate_contact(contact, label)

    # ownerMessage -> guesthouse["ownerMessage"], optional
    if "ownerMessage" in guesthouse and not _is_str_or_none(guesthouse["ownerMessage"]):
        raise ValueError(f"{label} ownerMessage must be str or None")


def _validate_location(location: dict, label: str) -> None:
    # lotNumberAddress -> location["lotNumberAddress"]
    _require_non_empty_str(location, "lotNumberAddress", f"{label} location")

    # roadNameAddress -> location["roadNameAddress"], optional
    if "roadNameAddress" in location and not _is_str_or_none(location["roadNameAddress"]):
        raise ValueError(f"{label} location.roadNameAddress must be str or None")

    # coordinates -> location["coordinates"]
    coordinates = _require_field(location, "coordinates", f"{label} location")
    if (
        not isinstance(coordinates, list)
        or len(coordinates) != 2
        or not all(_is_number(value) for value in coordinates)
    ):
        raise ValueError(f"{label} location.coordinates must be [longitude, latitude]")


def _validate_room(room: Any, index: int, label: str) -> None:
    if not isinstance(room, dict):
        raise ValueError(f"{label} rooms[{index}] must be a dict")

    # name -> rooms[].name
    _require_non_empty_str(room, "name", f"{label} rooms[{index}]")

    # type -> rooms[].type
    _require_non_empty_str(room, "type", f"{label} rooms[{index}]")

    # headCount -> rooms[].headCount
    head_count = _require_field(room, "headCount", f"{label} rooms[{index}]")
    if not isinstance(head_count, int) or isinstance(head_count, bool):
        raise ValueError(f"{label} rooms[{index}].headCount must be int")

    # checkInTime -> rooms[].checkInTime
    _require_non_empty_str(room, "checkInTime", f"{label} rooms[{index}]")

    # checkOutTime -> rooms[].checkOutTime
    _require_non_empty_str(room, "checkOutTime", f"{label} rooms[{index}]")

    # pricePerNight -> rooms[].pricePerNight
    price = _require_field(room, "pricePerNight", f"{label} rooms[{index}]")
    if not _is_number(price):
        raise ValueError(f"{label} rooms[{index}].pricePerNight must be int or float")
    if price < 0:
        raise ValueError(f"{label} rooms[{index}].pricePerNight must not be negative")

    # imageUrls -> rooms[].imageUrls
    _validate_image_urls(_require_field(room, "imageUrls", f"{label} rooms[{index}]"), f"{label} rooms[{index}].imageUrls")


def _validate_party(party: Any, index: int, label: str) -> None:
    if not isinstance(party, dict):
        raise ValueError(f"{label} parties[{index}] must be a dict")

    # type -> parties[].type
    _require_non_empty_str(party, "type", f"{label} parties[{index}]")

    # otherPartyType -> parties[].otherPartyType
    other_party_type = _require_field(party, "otherPartyType", f"{label} parties[{index}]")
    if not _is_str_or_none(other_party_type):
        raise ValueError(f"{label} parties[{index}].otherPartyType must be str or None")

    # startTime -> parties[].startTime
    start_time = _require_field(party, "startTime", f"{label} parties[{index}]")
    if not _is_str_or_none(start_time):
        raise ValueError(f"{label} parties[{index}].startTime must be str or None")

    # endTime -> parties[].endTime
    end_time = _require_field(party, "endTime", f"{label} parties[{index}]")
    if not _is_str_or_none(end_time):
        raise ValueError(f"{label} parties[{index}].endTime must be str or None")

    # weeklyDays -> parties[].weeklyDays
    _require_list(party, "weeklyDays", f"{label} parties[{index}]")

    # place -> parties[].place
    _require_non_empty_str(party, "place", f"{label} parties[{index}]")

    # moods -> parties[].moods
    _require_list(party, "moods", f"{label} parties[{index}]")

    # isExternalGuestAllowed -> parties[].isExternalGuestAllowed
    is_external_guest_allowed = _require_field(party, "isExternalGuestAllowed", f"{label} parties[{index}]")
    if not isinstance(is_external_guest_allowed, bool):
        raise ValueError(f"{label} parties[{index}].isExternalGuestAllowed must be bool")

    # guestFee -> parties[].guestFee
    _validate_optional_non_negative_number(
        _require_field(party, "guestFee", f"{label} parties[{index}]"),
        f"{label} parties[{index}].guestFee",
    )

    # externalGuestFee -> parties[].externalGuestFee
    _validate_optional_non_negative_number(
        _require_field(party, "externalGuestFee", f"{label} parties[{index}]"),
        f"{label} parties[{index}].externalGuestFee",
    )

    # imageUrls -> parties[].imageUrls
    _validate_image_urls(_require_field(party, "imageUrls", f"{label} parties[{index}]"), f"{label} parties[{index}].imageUrls")

    # information -> parties[].information
    _require_non_empty_str(party, "information", f"{label} parties[{index}]")


def _validate_contact(contact: dict, label: str) -> None:
    # contact.* fields are optional, but values must be str or None when present.
    for field in ["phoneNumber", "instagramId", "webSite", "reservationUrl"]:
        if field in contact and not _is_str_or_none(contact[field]):
            raise ValueError(f"{label} contact.{field} must be str or None")


def _validate_image_urls(value: Any, label: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    if not value:
        raise ValueError(f"{label} must not be empty")

    for index, image_url in enumerate(value):
        if not isinstance(image_url, str):
            raise ValueError(f"{label}[{index}] must be str")
        if not image_url.strip():
            raise ValueError(f"{label}[{index}] must not be empty")


def _require_field(data: dict, field: str, label: str) -> Any:
    if field not in data:
        raise ValueError(f"{label} missing required field: {field}")
    return data[field]


def _require_non_empty_str(data: dict, field: str, label: str) -> str:
    value = _require_field(data, field, label)
    if not isinstance(value, str):
        raise ValueError(f"{label} {field} must be str")
    if not value.strip():
        raise ValueError(f"{label} {field} must not be empty")
    return value


def _require_dict(data: dict, field: str, label: str) -> dict:
    value = _require_field(data, field, label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} {field} must be a dict")
    return value


def _require_list(data: dict, field: str, label: str) -> list:
    value = _require_field(data, field, label)
    if not isinstance(value, list):
        raise ValueError(f"{label} {field} must be a list")
    return value


def _validate_optional_non_negative_number(value: Any, label: str) -> None:
    if value is None:
        return
    if not _is_number(value):
        raise ValueError(f"{label} must be int, float, or None")
    if value < 0:
        raise ValueError(f"{label} must not be negative")


def _guesthouse_label(guesthouse: dict, index: int) -> str:
    name = guesthouse.get("guestHouseName")
    return f"[{name}]" if isinstance(name, str) and name.strip() else f"[guesthouse index {index}]"


def _is_str_or_none(value: Any) -> bool:
    return isinstance(value, str) or value is None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
