import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from src.guesthouses.vector_store import (
    has_vector_store_documents as has_guesthouse_documents,
    is_vector_store_initialized as is_guesthouse_index_initialized,
    synchronize_guesthouse_documents,
)
from src.staff_steps.vector_store import (
    has_vector_store_documents as has_staff_documents,
    is_vector_store_initialized as is_staff_index_initialized,
)
from rebuild_index import rebuild_guesthouse_index


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

    def test_empty_dynamic_collections_are_initialized_without_documents(self) -> None:
        empty_collection = SimpleNamespace(count=lambda: 0)
        fake_chromadb = SimpleNamespace(
            PersistentClient=lambda path: SimpleNamespace(
                get_collection=lambda name: empty_collection,
            )
        )

        with TemporaryDirectory() as directory, patch.dict(
            "sys.modules",
            {"chromadb": fake_chromadb},
        ):
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.assertTrue(is_guesthouse_index_initialized(directory))
            self.assertTrue(is_staff_index_initialized(directory))
            self.assertFalse(has_guesthouse_documents(directory))
            self.assertFalse(has_staff_documents(directory))

    @patch("rebuild_index.build_vector_store")
    @patch("rebuild_index.clear_guesthouse_vector_store")
    @patch("rebuild_index.load_guesthouses_from_backend", return_value=[])
    def test_guesthouse_rebuild_initializes_empty_collection(
        self,
        _load_guesthouses,
        clear_guesthouse_store,
        build_guesthouse_store,
    ) -> None:
        rebuild_guesthouse_index()

        clear_guesthouse_store.assert_called_once_with()
        build_guesthouse_store.assert_not_called()


if __name__ == "__main__":
    unittest.main()
