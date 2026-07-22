from __future__ import annotations

from pathlib import Path
from typing import Any

from src.guesthouses.vector_store import _embed_texts


COLLECTION_NAME = "staff_recruitments"
DEFAULT_PERSIST_DIRECTORY = "storage/vector_db/staff_recruitments"


def build_vector_store(
    documents: list[dict],
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    reset: bool = True,
) -> None:
    if not documents:
        raise ValueError("documents must not be empty")
    collection = _get_collection(persist_directory, reset)
    collection.upsert(
        ids=[f"staff-recruitment-{item['staffRecruitmentId']}" for item in documents],
        documents=[item["content"] for item in documents],
        embeddings=_embed_texts([item["content"] for item in documents]),
        metadatas=[
            {
                "staffRecruitmentId": item["staffRecruitmentId"],
                "title": item["title"],
                "guestHouseName": item["guestHouseName"],
                "region": item["region"],
                "workingPeriod": item["workingPeriod"],
                "gender": item["gender"],
            }
            for item in documents
        ],
    )


def clear_vector_store(persist_directory: str = DEFAULT_PERSIST_DIRECTORY) -> None:
    """Remove stale staff documents when the backend has no active postings."""
    _get_collection(persist_directory, reset=True)


def upsert_staff_recruitment_document(
    document: dict,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
) -> None:
    collection = _get_collection(persist_directory, reset=False)
    collection.upsert(
        ids=[f"staff-recruitment-{document['staffRecruitmentId']}"],
        documents=[document["content"]],
        embeddings=_embed_texts([document["content"]]),
        metadatas=[{
            "staffRecruitmentId": document["staffRecruitmentId"],
            "title": document["title"],
            "guestHouseName": document["guestHouseName"],
            "region": document["region"],
            "workingPeriod": document["workingPeriod"],
            "gender": document["gender"],
        }],
    )


def delete_staff_recruitment_document(
    recruitment_id: int,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
) -> None:
    if recruitment_id <= 0:
        raise ValueError("recruitment_id must be positive")
    _get_collection(persist_directory, reset=False).delete(ids=[f"staff-recruitment-{recruitment_id}"])


def synchronize_staff_recruitment_documents(
    documents: list[dict],
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
) -> dict[str, int]:
    """Repair missing, changed, and stale rows without resetting the collection."""
    collection = _get_collection(persist_directory, reset=False)
    indexed = collection.get(include=["documents"])
    indexed_documents = dict(zip(indexed.get("ids", []), indexed.get("documents", [])))
    expected_documents = {
        f"staff-recruitment-{document['staffRecruitmentId']}": document["content"]
        for document in documents
    }
    changed = [
        document
        for document in documents
        if indexed_documents.get(f"staff-recruitment-{document['staffRecruitmentId']}") != document["content"]
    ]
    stale_ids = sorted(set(indexed_documents) - set(expected_documents))
    if changed:
        build_vector_store(changed, persist_directory=persist_directory, reset=False)
    if stale_ids:
        collection.delete(ids=stale_ids)
    return {"upserted": len(changed), "deleted": len(stale_ids), "total": len(documents)}


def search_staff_recruitments(
    query: str,
    top_k: int = 3,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    conditions: dict[str, str] | None = None,
    candidate_ids: list[int] | None = None,
) -> list[dict]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if candidate_ids is not None and not candidate_ids:
        return []
    collection = _get_collection(persist_directory, reset=False)
    count = collection.count()
    if count == 0:
        return []
    query_arguments: dict[str, Any] = {
        "query_embeddings": [_embed_texts([query])[0]],
        "n_results": min(top_k, count),
        "include": ["documents", "metadatas", "distances"],
    }
    where = _build_where(conditions or {})
    if candidate_ids is not None:
        unique_ids = sorted(set(candidate_ids))
        id_clause = (
            {"staffRecruitmentId": {"$eq": unique_ids[0]}}
            if len(unique_ids) == 1
            else {"staffRecruitmentId": {"$in": unique_ids}}
        )
        where = _combine_where(where, id_clause)
        query_arguments["n_results"] = min(top_k, len(unique_ids), count)
    if where:
        query_arguments["where"] = where
    result = collection.query(
        **query_arguments,
    )
    return _format_results(result)


def has_vector_store_documents(persist_directory: str = DEFAULT_PERSIST_DIRECTORY) -> bool:
    if not Path(persist_directory).exists():
        return False
    try:
        import chromadb
        collection = chromadb.PersistentClient(path=persist_directory).get_collection(COLLECTION_NAME)
        return collection.count() > 0
    except Exception:
        return False


def _get_collection(persist_directory: str, reset: bool) -> Any:
    try:
        import chromadb
    except ImportError as error:
        raise ImportError("Chroma vector store requires package: chromadb") from error
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_directory)
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION_NAME)


def _format_results(result: dict) -> list[dict]:
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    return [
        {**(metadata or {}), "content": document, "distance": distance}
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]


def _build_where(conditions: dict[str, str]) -> dict | None:
    allowed = {key: value for key, value in conditions.items() if key in {"region", "workingPeriod", "gender"}}
    clauses = [{key: {"$eq": value}} for key, value in allowed.items()]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _combine_where(first: dict | None, second: dict) -> dict:
    if first is None:
        return second
    return {"$and": [first, second]}
