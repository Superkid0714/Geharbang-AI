import unittest
from unittest.mock import patch

from src.staff_steps.data_loader import validate_staff_recruitments_payload
from src.staff_steps.document_builder import build_staff_recruitment_document
from src.staff_steps.recommender import _select_by_reviews, answer_staff_chat
from src.staff_steps.structured_filter import extract_staff_conditions, filter_staff_recruitments


SAMPLE = {
    "staffRecruitments": [
        {
            "id": 12,
            "averageRating": 4.8,
            "reviewCount": 7,
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
        self.assertIn("스텝 후기는 7개", document["content"])
        self.assertEqual(document["workingPeriod"], "한달")
        self.assertEqual(document["averageRating"], 4.8)
        self.assertEqual(document["reviewCount"], 7)

    def test_rejects_missing_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "id must be an integer"):
            validate_staff_recruitments_payload({"staffRecruitments": [{"details": {}}]})

    def test_extracts_hard_search_conditions(self) -> None:
        self.assertEqual(
            extract_staff_conditions("애월에서 한 달 일할 여성 스텝 공고"),
            {"region": "애월_협재", "workingPeriod": "중기", "gender": "여"},
        )
        self.assertEqual(
            extract_staff_conditions("제주시 단기 성별 무관 스텝"),
            {"region": "제주시", "workingPeriod": "단기", "gender": "무관"},
        )

    def test_filters_latest_backend_records(self) -> None:
        items = validate_staff_recruitments_payload(SAMPLE)
        self.assertEqual(
            filter_staff_recruitments(items, {"region": "애월", "workingPeriod": "한달"}),
            items,
        )
        self.assertEqual(filter_staff_recruitments(items, {"region": "제주시"}), [])

    def test_selects_staff_recruitment_by_review_request(self) -> None:
        documents = [
            {"staffRecruitmentId": 1, "averageRating": 4.9, "reviewCount": 2},
            {"staffRecruitmentId": 2, "averageRating": 4.7, "reviewCount": 15},
        ]

        self.assertEqual(
            _select_by_reviews("후기 좋은 스텝 공고", documents)["staffRecruitmentId"],
            1,
        )
        self.assertEqual(
            _select_by_reviews("리뷰 많은 스텝 공고", documents)["staffRecruitmentId"],
            2,
        )

    @patch("src.staff_steps.recommender._generate_gemini_answer", side_effect=lambda prompt: prompt)
    @patch("src.staff_steps.recommender.search_staff_recruitments")
    @patch("src.staff_steps.recommender.load_staff_recruitments")
    def test_replaces_stale_staff_address_with_current_backend_data(
        self, load_current, search, _generate
    ) -> None:
        current = validate_staff_recruitments_payload(SAMPLE)
        load_current.return_value = current
        search.return_value = [{
            "staffRecruitmentId": 12,
            "title": "예전 공고",
            "guestHouseName": "예전 게하",
            "content": "주소: 예전 주소",
            "distance": 0.2,
        }]

        answer, result = answer_staff_chat("애월 스텝 공고 추천해줘")

        self.assertIn("주소: 제주시 애월읍", answer)
        self.assertNotIn("예전 주소", answer)
        self.assertEqual(result["title"], "애월 한 달 스텝 모집")


if __name__ == "__main__":
    unittest.main()
