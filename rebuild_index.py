from src.guesthouses.data_loader import load_guesthouses
from src.guesthouses.document_builder import build_guesthouse_documents
from src.guesthouses.vector_store import build_vector_store


def main() -> None:
    """Rebuild the guesthouse Chroma vector index from JSON data."""
    print("guesthouses.json 로드 및 검증 중...")
    guesthouses = load_guesthouses()
    print(f"게스트하우스 개수: {len(guesthouses)}")

    print("검색용 문서 생성 중...")
    documents = build_guesthouse_documents(guesthouses)
    print(f"검색용 문서 개수: {len(documents)}")

    print("기존 Chroma 컬렉션 초기화 및 Dense 임베딩 저장 중...")
    build_vector_store(documents=documents, reset=True)

    print("벡터 DB 재구축이 완료되었습니다.")


if __name__ == "__main__":
    main()
