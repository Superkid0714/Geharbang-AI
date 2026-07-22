import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.jeju_travel.data_loader import build_documents_from_pages, load_jeju_travel_documents
from src.jeju_travel.recommender import answer_jeju_travel
from src.jeju_travel.vector_store import _rerank_results


class JejuTravelDataTest(unittest.TestCase):
    def test_builds_page_grounded_chunks_with_normalized_source(self) -> None:
        documents = build_documents_from_pages(
            "제주 여행.pdf",
            ["짧음", "성산일출봉은 제주 동쪽의 대표적인 자연 명소입니다. " * 80],
        )

        self.assertGreater(len(documents), 1)
        self.assertEqual(documents[0]["sourceTitle"], "제주 여행")
        self.assertEqual(documents[0]["page"], 2)
        self.assertIn("페이지: 2", documents[0]["content"])
        self.assertTrue(all(len(document["content"]) < 1400 for document in documents))

    def test_rejects_directory_without_pdfs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "PDF 파일이 없습니다"):
                load_jeju_travel_documents(directory, str(Path(directory) / "cache"))


class JejuTravelAnswerTest(unittest.TestCase):
    @patch("src.jeju_travel.recommender._generate_gemini_answer")
    @patch("src.jeju_travel.recommender.search_jeju_travel")
    def test_answers_without_exposing_source_metadata(self, mock_search, mock_generate) -> None:
        mock_search.return_value = [
            {"sourceTitle": "제주 안내서", "page": 12, "content": "성산일출봉 설명"},
            {"sourceTitle": "제주 안내서", "page": 12, "content": "성산일출봉 추가 설명"},
            {"sourceTitle": "제주 문화", "page": 31, "content": "제주 자연 설명"},
        ]
        mock_generate.return_value = "성산일출봉을 아침 일정으로 추천합니다."

        answer = answer_jeju_travel("성산일출봉 여행 코스 알려줘")

        self.assertEqual(answer, "성산일출봉을 아침 일정으로 추천합니다.")
        self.assertNotIn("참고 자료", answer)
        prompt = mock_generate.call_args.args[0]
        self.assertIn("출처를 드러내는 표현은 절대 사용하지 않습니다", prompt)
        self.assertIn("AI 여행 선배", prompt)
        self.assertIn("복합형 동반자", prompt)

    def test_reranking_prioritizes_requested_region_source(self) -> None:
        results = _rerank_results(
            "제주 북쪽에서 조용히 산책하기 좋은 곳",
            [
                {"sourceTitle": "제주 일반 여행", "content": "조용한 숲 산책", "distance": 0.70},
                {"sourceTitle": "대한민국 도슨트 제주 북쪽", "content": "마을과 해변을 걷는 길", "distance": 0.78},
            ],
            top_k=2,
        )

        self.assertEqual(results[0]["sourceTitle"], "대한민국 도슨트 제주 북쪽")


if __name__ == "__main__":
    unittest.main()
