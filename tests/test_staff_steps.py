import unittest

from src.staff_steps.data_loader import validate_staff_recruitments_payload
from src.staff_steps.document_builder import build_staff_recruitment_document
from src.staff_steps.structured_filter import extract_staff_conditions


SAMPLE = {
    "staffRecruitments": [
        {
            "id": 12,
            "details": {
                "title": "애월 한 달 스텝 모집",
                "guestHouseName": "테스트 게하",
                "region": "애월",
                "location": {"address": "제주시 애월읍"},
                "workingInformation": {
                    "startDate": "2026-08-01",
                    "isStartDateNegotiable": False,
                    "workingPeriod": "한달",
                    "jobs": [{
                        "name": "객실 관리", "startTIme": "10:00:00", "endTime": "13:00:00",
                        "job": "침구 정리", "workDays": 5, "restDays": 2,
                        "workType": "시간", "weeklyWorkingDays": "주5일",
                    }],
                },
                "introduction": {"content": "밝은 분을 찾아요."},
                "feature": {"gender": "무관", "advantages": ["경력"], "employeeBenefits": ["숙소 제공"]},
                "ownerMessage": "함께 즐겁게 일해요.",
            },
        }
    ]
}


class StaffStepDataTest(unittest.TestCase):
    def test_validates_and_builds_search_document(self) -> None:
        items = validate_staff_recruitments_payload(SAMPLE)
        document = build_staff_recruitment_document(items[0])
        self.assertEqual(document["staffRecruitmentId"], 12)
        self.assertIn("애월 한 달 스텝 모집", document["content"])
        self.assertIn("침구 정리", document["content"])
        self.assertIn("숙소 제공", document["content"])
        self.assertEqual(document["workingPeriod"], "한달")

    def test_rejects_missing_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "id must be an integer"):
            validate_staff_recruitments_payload({"staffRecruitments": [{"details": {}}]})

    def test_extracts_hard_search_conditions(self) -> None:
        self.assertEqual(
            extract_staff_conditions("애월에서 한 달 일할 여성 스텝 공고"),
            {"region": "애월_협재", "workingPeriod": "중기", "gender": "여"},
        )


if __name__ == "__main__":
    unittest.main()
