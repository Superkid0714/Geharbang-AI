from __future__ import annotations

from pathlib import Path
from typing import Any


COLLECTION_NAME = "guesthouses"
MODEL_NAME = "BAAI/bge-m3"
DEFAULT_PERSIST_DIRECTORY = "storage/vector_db/guesthouses"
_BGE_MODEL: Any | None = None


def build_vector_store(
    documents: list[dict],
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    reset: bool = True,
) -> None:
    """Embed guesthouse documents with BGE-M3 and store them in local Chroma."""
    _validate_documents(documents)

    collection = _get_collection(persist_directory, reset=reset)
    texts = [document["content"] for document in documents]
    embeddings = _embed_texts(texts)

    ids = [f"guesthouse-{document['guestHouseId']}" for document in documents]
    metadatas = [
        {
            "guestHouseId": document["guestHouseId"],
            "guestHouseName": document["guestHouseName"],
        }
        for document in documents
    ]

    # Upsert keeps reset=False usable without failing on duplicate ids.
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def search_guesthouses(
    query: str,
    top_k: int = 3,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
) -> list[dict]:
    """Search semantically similar guesthouse documents from local Chroma."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    collection = _get_collection(persist_directory, reset=False)
    query_embedding = _embed_texts([query])[0]

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return _format_search_results(result)


def has_vector_store_documents(
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
) -> bool:
    """Return True when the existing Chroma collection contains documents."""
    if not Path(persist_directory).exists():
        return False

    try:
        import chromadb
    except ImportError as error:
        raise ImportError("Chroma vector store requires package: chromadb") from error

    client = chromadb.PersistentClient(path=persist_directory)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return False

    return collection.count() > 0


def _get_model() -> Any:
    global _BGE_MODEL

    if _BGE_MODEL is None:
        try:
            import torch
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as error:
            raise ImportError(
                "BGE-M3 embedding requires packages: FlagEmbedding and torch"
            ) from error

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
        _BGE_MODEL = BGEM3FlagModel(
            MODEL_NAME,
            use_fp16=False,
            device=device,
        )

    return _BGE_MODEL


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
            # Chroma raises when the collection does not exist; that is fine.
            pass

    return client.get_or_create_collection(name=COLLECTION_NAME)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    output = model.encode(
        texts,
        batch_size=4,
        max_length=8192,
    )
    dense_embeddings = output["dense_vecs"]

    if hasattr(dense_embeddings, "tolist"):
        dense_embeddings = dense_embeddings.tolist()

    return [
        embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        for embedding in dense_embeddings
    ]


def _validate_documents(documents: list[dict]) -> None:
    if not documents:
        raise ValueError("documents must not be empty")

    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"documents[{index}] must be a dict")

        for field in ["guestHouseId", "guestHouseName", "content"]:
            if field not in document:
                raise ValueError(f"documents[{index}] missing required field: {field}")

        if not isinstance(document["guestHouseName"], str) or not document["guestHouseName"].strip():
            raise ValueError(f"documents[{index}].guestHouseName must not be empty")
        if not isinstance(document["content"], str) or not document["content"].strip():
            raise ValueError(f"documents[{index}].content must not be empty")


def _format_search_results(result: dict) -> list[dict]:
    documents = _first_result_list(result.get("documents"))
    metadatas = _first_result_list(result.get("metadatas"))
    distances = _first_result_list(result.get("distances"))

    formatted_results: list[dict] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        formatted_results.append(
            {
                "guestHouseId": metadata.get("guestHouseId"),
                "guestHouseName": metadata.get("guestHouseName"),
                "content": document,
                "distance": distance,
            }
        )

    return formatted_results


def _first_result_list(value: Any) -> list:
    if not value:
        return []
    return value[0]
