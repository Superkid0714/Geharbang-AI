from __future__ import annotations

import re


REGION_KEYWORDS = {
    "애월_협재": ["애월", "협재", "한림", "금능", "곽지", "서쪽"],
    "성산_구좌": ["성산", "구좌", "월정리", "세화", "동쪽"],
    "서귀포시": ["서귀포", "남원", "표선"],
    "중문": ["중문", "대정"],
    "제주시": ["제주시", "제주 시내", "공항 근처"],
    "우도_기타": ["우도", "추자", "비양도"],
}


def extract_staff_conditions(query: str) -> dict[str, str]:
    normalized = re.sub(r"\s+", " ", query.casefold()).strip()
    conditions: dict[str, str] = {}

    for region, keywords in REGION_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            conditions["region"] = region
            break

    if any(word in normalized for word in ["단기", "4주 이하", "일주일", "1주", "2주", "3주"]):
        conditions["workingPeriod"] = "단기"
    elif any(word in normalized for word in ["중기", "한 달", "한달", "1개월", "두 달", "2개월", "세 달", "3개월"]):
        conditions["workingPeriod"] = "중기"
    elif any(word in normalized for word in ["장기", "3개월 이상", "반년", "6개월"]):
        conditions["workingPeriod"] = "장기"

    if any(word in normalized for word in ["여성만", "여자만", "여성 스텝", "여자 스텝"]):
        conditions["gender"] = "여"
    elif any(word in normalized for word in ["남성만", "남자만", "남성 스텝", "남자 스텝"]):
        conditions["gender"] = "남"

    return conditions
