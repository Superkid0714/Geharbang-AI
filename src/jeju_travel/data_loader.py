from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import unicodedata
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path


DEFAULT_PDF_DIRECTORY = "data/jeju_travel"
DEFAULT_CACHE_DIRECTORY = "storage/vector_db/jeju_travel_extraction_cache"
CACHE_VERSION = 1
MAX_CHUNK_CHARACTERS = 1200
CHUNK_OVERLAP_CHARACTERS = 180
MIN_CONTENT_CHARACTERS = 20


def load_jeju_travel_documents(
    directory: str = DEFAULT_PDF_DIRECTORY,
    cache_directory: str = DEFAULT_CACHE_DIRECTORY,
    enable_ocr: bool = True,
) -> list[dict]:
    """Extract and chunk every local Jeju PDF, OCRing image-only books when needed."""
    pdf_directory = Path(directory)
    if not pdf_directory.is_dir():
        raise ValueError(f"제주 여행 PDF 폴더를 찾을 수 없습니다: {directory}")

    pdf_paths = sorted(pdf_directory.glob("*.pdf"))
    if not pdf_paths:
        raise ValueError("제주 여행 PDF 파일이 없습니다.")

    documents: list[dict] = []
    for pdf_path in pdf_paths:
        pages = _load_or_extract_pages(pdf_path, Path(cache_directory), enable_ocr)
        documents.extend(build_documents_from_pages(pdf_path.name, pages))

    if not documents:
        raise ValueError("제주 여행 PDF에서 검색 가능한 텍스트를 추출하지 못했습니다.")
    return documents


def build_documents_from_pages(source_file: str, pages: list[str]) -> list[dict]:
    """Build page-grounded chunks with stable identifiers and citation metadata."""
    normalized_file = unicodedata.normalize("NFC", source_file)
    source_title = Path(normalized_file).stem.strip()
    source_id = hashlib.sha256(normalized_file.encode("utf-8")).hexdigest()[:16]
    documents: list[dict] = []

    for page_number, raw_text in enumerate(pages, start=1):
        text = _normalize_extracted_text(raw_text)
        if len(text) < MIN_CONTENT_CHARACTERS:
            continue
        for chunk_index, chunk in enumerate(_split_text(text), start=1):
            chunk_id = f"{source_id}-p{page_number}-c{chunk_index}"
            documents.append({
                "chunkId": chunk_id,
                "sourceFile": normalized_file,
                "sourceTitle": source_title,
                "page": page_number,
                "chunkIndex": chunk_index,
                "content": (
                    f"자료명: {source_title}\n"
                    f"페이지: {page_number}\n\n"
                    f"{chunk}"
                ),
            })
    return documents


def _load_or_extract_pages(
    pdf_path: Path,
    cache_directory: Path,
    enable_ocr: bool,
) -> list[str]:
    fingerprint = _file_sha256(pdf_path)
    cache_path = cache_directory / f"{fingerprint}.json"
    if cache_path.is_file():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("version") == CACHE_VERSION and isinstance(payload.get("pages"), list):
                return [str(page) for page in payload["pages"]]
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    pages = _extract_pdf_pages(pdf_path, enable_ocr=enable_ocr)
    cache_directory.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps({"version": CACHE_VERSION, "pages": pages}, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(cache_path)
    return pages


def _extract_pdf_pages(pdf_path: Path, enable_ocr: bool) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise ImportError("PDF 추출에는 pypdf 패키지가 필요합니다.") from error

    try:
        reader = PdfReader(pdf_path)
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as error:
        raise ValueError(f"PDF를 읽을 수 없습니다: {pdf_path.name}") from error

    if not pages:
        raise ValueError(f"페이지가 없는 PDF입니다: {pdf_path.name}")

    text_page_count = sum(len(_normalize_extracted_text(text)) >= MIN_CONTENT_CHARACTERS for text in pages)
    # OCR only when a meaningful portion of the book has no embedded text. This avoids
    # spending time on intentionally blank/photo pages in normal text PDFs.
    needs_ocr = text_page_count / len(pages) < 0.8
    if needs_ocr:
        if not enable_ocr:
            raise ValueError(f"OCR이 필요한 스캔 PDF입니다: {pdf_path.name}")
        pages = _ocr_missing_pages(pdf_path, pages)

    usable_pages = sum(len(_normalize_extracted_text(text)) >= MIN_CONTENT_CHARACTERS for text in pages)
    if usable_pages == 0:
        raise ValueError(f"PDF에서 텍스트를 추출하지 못했습니다: {pdf_path.name}")
    return pages


def _ocr_missing_pages(pdf_path: Path, pages: list[str]) -> list[str]:
    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        raise RuntimeError(
            f"스캔 PDF OCR에 tesseract와 한국어 언어팩이 필요합니다: {pdf_path.name}"
        )
    _verify_korean_tesseract(tesseract_path)

    try:
        import pymupdf
    except ImportError as error:
        raise ImportError("스캔 PDF 렌더링에는 PyMuPDF 패키지가 필요합니다.") from error

    missing_indexes = [
        index
        for index, text in enumerate(pages)
        if len(_normalize_extracted_text(text)) < MIN_CONTENT_CHARACTERS
    ]
    if not missing_indexes:
        return pages

    worker_count = max(1, min(int(os.getenv("JEJU_TRAVEL_OCR_WORKERS", "3")), 6))
    pending_limit = worker_count * 2
    results = list(pages)
    document = pymupdf.open(pdf_path)
    pending: dict[Future[str], int] = {}

    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for page_index in missing_indexes:
                while len(pending) >= pending_limit:
                    _collect_completed_ocr(pending, results)
                page = document.load_page(page_index)
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), colorspace=pymupdf.csGRAY)
                future = executor.submit(_run_tesseract, tesseract_path, pixmap.tobytes("png"))
                pending[future] = page_index
            while pending:
                _collect_completed_ocr(pending, results)
    finally:
        document.close()

    return results


def _collect_completed_ocr(pending: dict[Future[str], int], results: list[str]) -> None:
    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
    for future in completed:
        page_index = pending.pop(future)
        results[page_index] = future.result()


def _run_tesseract(tesseract_path: str, image_bytes: bytes) -> str:
    timeout = max(10, int(os.getenv("JEJU_TRAVEL_OCR_PAGE_TIMEOUT_SECONDS", "90")))
    try:
        process = subprocess.run(
            [tesseract_path, "stdin", "stdout", "-l", "kor+eng", "--psm", "6"],
            input=image_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("PDF 한 페이지의 OCR 처리 시간이 초과되었습니다.") from error
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Tesseract OCR 처리에 실패했습니다: {detail[:300]}")
    return process.stdout.decode("utf-8", errors="replace").strip()


def _verify_korean_tesseract(tesseract_path: str) -> None:
    process = subprocess.run(
        [tesseract_path, "--list-langs"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=15,
    )
    languages = set(process.stdout.decode("utf-8", errors="replace").split())
    if process.returncode != 0 or "kor" not in languages:
        raise RuntimeError("Tesseract 한국어 언어팩(kor)이 설치되어 있지 않습니다.")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _normalize_extracted_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value or "")
    value = value.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _split_text(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARACTERS:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + MAX_CHUNK_CHARACTERS, len(text))
        end = hard_end
        if hard_end < len(text):
            candidates = [
                text.rfind("\n\n", start + MAX_CHUNK_CHARACTERS // 2, hard_end),
                text.rfind(". ", start + MAX_CHUNK_CHARACTERS // 2, hard_end),
                text.rfind("\n", start + MAX_CHUNK_CHARACTERS // 2, hard_end),
                text.rfind(" ", start + MAX_CHUNK_CHARACTERS // 2, hard_end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (1 if text[boundary] == "." else 0)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP_CHARACTERS)
    return chunks
