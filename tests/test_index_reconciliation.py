import unittest
from unittest.mock import patch

from src.guesthouses.vector_store import synchronize_guesthouse_documents


class FakeCollection:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def get(self, include: list[str]) -> dict:
        return {
            "ids": ["guesthouse-1", "guesthouse-99"],
            "documents": ["unchanged", "stale"],
        }

    def delete(self, ids: list[str]) -> None:
        self.deleted.extend(ids)


class IndexReconciliationTest(unittest.TestCase):
    @patch("src.guesthouses.vector_store.build_vector_store")
    @patch("src.guesthouses.vector_store._get_collection")
    def test_repairs_changed_rows_and_removes_stale_ids(self, get_collection, build_store) -> None:
        collection = FakeCollection()
        get_collection.return_value = collection
        documents = [
            {"guestHouseId": 1, "guestHouseName": "유지", "content": "unchanged"},
            {"guestHouseId": 2, "guestHouseName": "신규", "content": "new"},
        ]

        result = synchronize_guesthouse_documents(documents, persist_directory="test-db")

        build_store.assert_called_once_with(
            [documents[1]], persist_directory="test-db", reset=False
        )
        self.assertEqual(collection.deleted, ["guesthouse-99"])
        self.assertEqual(result, {"upserted": 1, "deleted": 1, "total": 2})


if __name__ == "__main__":
    unittest.main()
