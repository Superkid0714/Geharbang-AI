import unittest

from src.service_guide.data_loader import load_service_guides
from src.service_guide.document_builder import build_service_guide_documents


class ServiceGuideTest(unittest.TestCase):
    def test_loads_valid_unique_guides(self) -> None:
        guides = load_service_guides()
        self.assertGreaterEqual(len(guides), 8)
        self.assertEqual(len(guides), len({guide["docId"] for guide in guides}))

    def test_builds_rag_documents_with_routes(self) -> None:
        document = build_service_guide_documents(load_service_guides())[0]
        self.assertIn("문서 제목:", document["content"])
        self.assertIn("관련 화면:", document["content"])


if __name__ == "__main__":
    unittest.main()
