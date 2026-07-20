from __future__ import annotations

from pathlib import Path
from typing import Any

from src.guesthouses.vector_store import _embed_texts


COLLECTION_NAME = "service_guide"
DEFAULT_PERSIST_DIRECTORY = "storage/vector_db/service_guide"


def build_vector_store(
    documents: list[dict],
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    reset: bool = True,
) -> None:
    if not documents:
        raise ValueError("documents must not be empty")
    collection = _get_collection(persist_directory, reset)
    contents = [document["content"] for document in documents]
    collection.upsert(
        ids=[f"service-guide-{document['docId']}" for document in documents],
        documents=contents,
        embeddings=_embed_texts(contents),
        metadatas=[
            {
                "docId": document["docId"],
                "title": document["title"],
                "audience": document["audience"],
                "lastVerified": document["lastVerified"],
                "routes": document["routes"],
                "sourceFile": document["sourceFile"],
            }
            for document in documents
        ],
    )


def search_service_guides(
    query: str,
    top_k: int = 3,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
) -> list[dict]:
    if not query.strip():
        raise ValueError("query must not be empty")
    collection = _get_collection(persist_directory, reset=False)
    count = collection.count()
    if count == 0:
        return []
    result = collection.query(
        query_embeddings=[_embed_texts([query])[0]],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    return [
        {**(metadata or {}), "content": document, "distance": distance}
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]


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
