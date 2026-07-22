import unittest
from unittest.mock import patch

from src.guesthouses.document_builder import build_guesthouse_document
from src.guesthouses.recommender import _select_best_guesthouse, answer_guesthouse_chat
from src.guesthouses.structured_filter import extract_structured_conditions, filter_guesthouses


GUESTHOUSE = {
    "guestHouseId": 1,
    "guestHouseName": "최신 숙소",
    "region": "제주시",
    "location": {
        "lotNumberAddress": "제주시 최신로 1",
        "roadNameAddress": "최신로 1",
        "coordinates": [126.5, 33.5],
    },
    "imageUrls": [],
    "introduction": "바다 근처에서 조용히 쉴 수 있습니다.",
    "amenities": ["수영장", "무료 Wi-Fi"],
    "moods": ["바닷가", "조용한"],
    "rooms": [{
        "name": "여성 도미토리", "type": "여성전용", "headCount": 4,
        "checkInTime": "16:00", "checkOutTime": "11:00", "pricePerNight": 30000,
    }],
    "parties": [],
    "contact": {},
    "ownerMessage": None,
    "averageRating": 4.8,
    "reviewCount": 12,
}


class GuesthouseRecommendationTest(unittest.TestCase):
    def test_extracts_and_applies_hard_conditions(self) -> None:
        conditions = extract_structured_conditions(
            "제주시에서 바닷가이고 조용하며 수영장 있는 1박 3만원 이하 평점 4.5점 이상 게하 추천"
        )
        self.assertEqual(conditions["region"], "제주시")
        self.assertEqual(conditions["moods"], ["바닷가", "조용한"])
        self.assertEqual(conditions["amenities"], ["수영장"])
        self.assertEqual(conditions["maxPricePerNight"], 30000)
        self.assertEqual(conditions["minAverageRating"], 4.5)
        self.assertEqual(len(filter_guesthouses([GUESTHOUSE], conditions)), 1)

    def test_review_summary_is_grounded_in_document(self) -> None:
        document = build_guesthouse_document(GUESTHOUSE)
        self.assertIn("리뷰는 12개", document["content"])
        self.assertIn("4.8점", document["content"])
        self.assertEqual(document["averageRating"], 4.8)

    def test_detail_question_keeps_semantic_target_instead_of_highest_rating(self) -> None:
        results = [
            {"guestHouseId": 1, "averageRating": 3.5, "reviewCount": 2, "distance": 0.1},
            {"guestHouseId": 2, "averageRating": 5.0, "reviewCount": 20, "distance": 0.2},
        ]
        selected = _select_best_guesthouse("첫 번째 숙소 평점 알려줘", results, rank_by_reviews=False)
        self.assertEqual(selected["guestHouseId"], 1)

    @patch("src.guesthouses.recommender._generate_gemini_answer", side_effect=lambda prompt: prompt)
    @patch("src.guesthouses.recommender.search_guesthouses")
    @patch("src.guesthouses.recommender.load_guesthouses_from_backend")
    def test_replaces_stale_vector_content_with_current_backend_data(
        self, load_current, search, _generate
    ) -> None:
        load_current.return_value = [GUESTHOUSE]
        search.return_value = [{
            "guestHouseId": 1,
            "guestHouseName": "예전 숙소",
            "content": "주소는 제주시 예전로 9입니다.",
            "distance": 0.1,
        }]

        answer, result = answer_guesthouse_chat("제주시 조용한 게하 추천해줘")

        self.assertIn("제주시 최신로 1", answer)
        self.assertNotIn("제주시 예전로 9", answer)
        self.assertEqual(result["guestHouseName"], "최신 숙소")
        search.assert_called_once_with(
            "제주시 조용한 게하 추천해줘",
            top_k=1,
            persist_directory="storage/vector_db/guesthouses",
            candidate_ids=[1],
        )


if __name__ == "__main__":
    unittest.main()
