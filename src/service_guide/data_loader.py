from __future__ import annotations

from pathlib import Path


DEFAULT_GUIDE_DIRECTORY = "data/service_guide"
REQUIRED_METADATA = {"doc_id", "title", "audience", "last_verified", "routes"}


def load_service_guides(directory: str = DEFAULT_GUIDE_DIRECTORY) -> list[dict]:
    path = Path(directory)
    if not path.is_dir():
        raise ValueError(f"서비스 안내 문서 폴더를 찾을 수 없습니다: {directory}")

    guides = [_parse_guide(file_path) for file_path in sorted(path.glob("*.md"))]
    if not guides:
        raise ValueError("서비스 안내 문서가 없습니다.")

    ids = [guide["docId"] for guide in guides]
    if len(ids) != len(set(ids)):
        raise ValueError("서비스 안내 문서의 doc_id가 중복되었습니다.")
    return guides


def _parse_guide(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError(f"{path.name}: YAML front matter가 필요합니다.")
    raw_metadata, body = text[4:].split("\n---\n", 1)
    metadata: dict[str, str] = {}
    for line in raw_metadata.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    missing = REQUIRED_METADATA - metadata.keys()
    if missing:
        raise ValueError(f"{path.name}: 필수 메타데이터 누락: {', '.join(sorted(missing))}")
    if not body.strip():
        raise ValueError(f"{path.name}: 본문이 비어 있습니다.")
    return {
        "docId": metadata["doc_id"],
        "title": metadata["title"],
        "audience": metadata["audience"],
        "lastVerified": metadata["last_verified"],
        "routes": metadata["routes"],
        "content": body.strip(),
        "sourceFile": path.name,
    }
