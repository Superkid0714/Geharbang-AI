from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from src.guesthouses.vector_store import _embed_texts


COLLECTION_NAME = "jeju_travel"
DEFAULT_PERSIST_DIRECTORY = "storage/vector_db/jeju_travel"
UPSERT_BATCH_SIZE = 32


def build_vector_store(
    documents: list[dict],
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    reset: bool = True,
) -> None:
    _validate_documents(documents)
    collection = _get_collection(persist_directory, reset=reset)
    for start in range(0, len(documents), UPSERT_BATCH_SIZE):
        batch = documents[start : start + UPSERT_BATCH_SIZE]
        contents = [document["content"] for document in batch]
        collection.upsert(
            ids=[f"jeju-travel-{document['chunkId']}" for document in batch],
            documents=contents,
            embeddings=_embed_texts(contents),
            metadatas=[{
                "sourceFile": document["sourceFile"],
                "sourceTitle": document["sourceTitle"],
                "page": document["page"],
                "chunkIndex": document["chunkIndex"],
                "region": _infer_source_region(document["sourceTitle"]),
            } for document in batch],
        )


def search_jeju_travel(
    query: str,
    top_k: int = 6,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
) -> list[dict]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    collection = _get_collection(persist_directory, reset=False)
    count = collection.count()
    if count == 0:
        return []
    candidate_count = min(max(top_k * 6, 30), count)
    query_embedding = _embed_texts([query])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )
    candidates = _format_query_results(result)

    requested_region = _requested_region_key(query)
    if requested_region in {"east", "north"}:
        regional_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(max(top_k * 3, 12), count),
            where={"region": {"$eq": requested_region}},
            include=["documents", "metadatas", "distances"],
        )
        candidates = _merge_candidates(candidates, _format_query_results(regional_result))
    return _rerank_results(query, candidates, top_k)


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


def _validate_documents(documents: list[dict]) -> None:
    if not documents:
        raise ValueError("documents must not be empty")
    required = {"chunkId", "sourceFile", "sourceTitle", "page", "chunkIndex", "content"}
    for index, document in enumerate(documents):
        missing = required - document.keys()
        if missing:
            raise ValueError(f"documents[{index}] missing fields: {', '.join(sorted(missing))}")
        if not isinstance(document["content"], str) or not document["content"].strip():
            raise ValueError(f"documents[{index}].content must not be empty")


_KOREAN_SUFFIXES = (
    "으로부터", "에서부터", "까지", "부터", "처럼", "보다", "에서", "으로",
    "하고", "이랑", "에게", "한테", "과", "와", "은", "는", "이", "가", "을", "를", "의", "에", "도", "만",
)
_QUERY_STOP_WORDS = {
    "제주", "여행", "여행지", "추천", "추천해줘", "알려줘", "어디야", "어디", "좋은", "함께",
}
_REGION_ALIASES = {
    "동쪽": ("동쪽", "동부"),
    "동부": ("동쪽", "동부"),
    "서쪽": ("서쪽", "서부"),
    "서부": ("서쪽", "서부"),
    "남쪽": ("남쪽", "남부"),
    "남부": ("남쪽", "남부"),
    "북쪽": ("북쪽", "북부"),
    "북부": ("북쪽", "북부"),
}
_REGION_KEYS = {
    "east": ("동쪽", "동부"),
    "west": ("서쪽", "서부"),
    "south": ("남쪽", "남부"),
    "north": ("북쪽", "북부"),
}


def _rerank_results(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Combine dense similarity with exact place/region evidence from the query."""
    normalized_query = unicodedata.normalize("NFC", query).casefold()
    terms = _extract_query_terms(normalized_query)
    requested_regions = {
        alias
        for marker, aliases in _REGION_ALIASES.items()
        if marker in normalized_query
        for alias in aliases
    }

    ranked: list[tuple[float, int, dict]] = []
    for index, candidate in enumerate(candidates):
        title = unicodedata.normalize("NFC", str(candidate.get("sourceTitle", ""))).casefold()
        content = unicodedata.normalize("NFC", str(candidate.get("content", ""))).casefold()
        distance = float(candidate.get("distance", 2.0))
        lexical_bonus = sum(0.035 for term in terms if term in content)
        title_bonus = sum(0.05 for term in terms if term in title)
        region_bonus = 0.18 if requested_regions and any(region in title for region in requested_regions) else 0.0
        ranked.append((distance - lexical_bonus - title_bonus - region_bonus, index, candidate))

    ranked.sort(key=lambda item: (item[0], item[1]))
    selected: list[dict] = []
    seen_signatures: set[tuple[str, str]] = set()
    for _score, _index, candidate in ranked:
        signature = (
            str(candidate.get("sourceTitle", "")),
            re.sub(r"\s+", "", str(candidate.get("content", "")))[:180],
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def _extract_query_terms(query: str) -> set[str]:
    terms: set[str] = set()
    for raw_term in re.findall(r"[0-9a-z가-힣]{2,}", query):
        term = raw_term
        for suffix in _KOREAN_SUFFIXES:
            if term.endswith(suffix) and len(term) - len(suffix) >= 2:
                term = term[: -len(suffix)]
                break
        if term not in _QUERY_STOP_WORDS and len(term) >= 2:
            terms.add(term)
    return terms


def _infer_source_region(source_title: str) -> str:
    normalized_title = unicodedata.normalize("NFC", source_title).casefold()
    for region, markers in _REGION_KEYS.items():
        if any(marker in normalized_title for marker in markers):
            return region
    return "general"


def _requested_region_key(query: str) -> str | None:
    normalized_query = unicodedata.normalize("NFC", query).casefold()
    for region, markers in _REGION_KEYS.items():
        if any(marker in normalized_query for marker in markers):
            return region
    return None


def _format_query_results(result: dict) -> list[dict]:
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    return [
        {**(metadata or {}), "id": result_id, "content": document, "distance": distance}
        for result_id, document, metadata, distance in zip(ids, documents, metadatas, distances)
    ]


def _merge_candidates(primary: list[dict], additional: list[dict]) -> list[dict]:
    merged = list(primary)
    seen_ids = {candidate.get("id") for candidate in primary}
    for candidate in additional:
        if candidate.get("id") not in seen_ids:
            merged.append(candidate)
            seen_ids.add(candidate.get("id"))
    return merged
