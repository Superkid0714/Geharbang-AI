import argparse

from src.guesthouses.data_loader import load_guesthouses_from_backend
from src.guesthouses.document_builder import build_guesthouse_documents
from src.guesthouses.vector_store import build_vector_store
from src.guesthouses.vector_store import clear_vector_store as clear_guesthouse_vector_store
from src.staff_steps.data_loader import load_staff_recruitments
from src.staff_steps.document_builder import build_staff_recruitment_documents
from src.staff_steps.vector_store import (
    build_vector_store as build_staff_vector_store,
    clear_vector_store as clear_staff_vector_store,
)
from src.service_guide.data_loader import load_service_guides
from src.service_guide.document_builder import build_service_guide_documents
from src.service_guide.vector_store import build_vector_store as build_service_guide_vector_store
from src.jeju_travel.data_loader import load_jeju_travel_documents
from src.jeju_travel.vector_store import build_vector_store as build_jeju_travel_vector_store


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="게하르방 RAG 인덱스 재구축")
    parser.add_argument(
        "--domain",
        choices=["all", "guesthouses", "staff-steps", "service-guide", "jeju-travel"],
        default="all",
    )
    args = parser.parse_args()

    if args.domain in {"all", "guesthouses"}:
        rebuild_guesthouse_index()
    if args.domain in {"all", "staff-steps"}:
        rebuild_staff_step_index()
    if args.domain in {"all", "service-guide"}:
        rebuild_service_guide_index()
    if args.domain in {"all", "jeju-travel"}:
        rebuild_jeju_travel_index()


def rebuild_guesthouse_index() -> None:
    print("백엔드 DB의 활성 게스트하우스 로드 및 검증 중...")
    guesthouses = load_guesthouses_from_backend()
    print(f"게스트하우스 개수: {len(guesthouses)}")
    if not guesthouses:
        clear_guesthouse_vector_store()
        print("활성 게스트하우스가 없어 기존 게스트하우스 인덱스를 비웠습니다.")
        return

    print("검색용 문서 생성 중...")
    documents = build_guesthouse_documents(guesthouses)
    print(f"검색용 문서 개수: {len(documents)}")

    print("기존 Chroma 컬렉션 초기화 및 Dense 임베딩 저장 중...")
    build_vector_store(documents=documents, reset=True)

    print("벡터 DB 재구축이 완료되었습니다.")


def rebuild_staff_step_index() -> None:
    print("백엔드 DB의 활성 스텝 공고 로드 및 검증 중...")
    recruitments = load_staff_recruitments()
    if not recruitments:
        clear_staff_vector_store()
        print("활성 스텝 공고가 없어 기존 스텝 인덱스를 비웠습니다.")
        return
    print(f"스텝 공고 개수: {len(recruitments)}")
    documents = build_staff_recruitment_documents(recruitments)
    print("스텝 공고 Chroma 컬렉션 초기화 및 Dense 임베딩 저장 중...")
    build_staff_vector_store(documents, reset=True)
    print("스텝 공고 벡터 DB 재구축이 완료되었습니다.")


def rebuild_service_guide_index() -> None:
    print("게하르방 서비스 안내 문서 로드 및 검증 중...")
    guides = load_service_guides()
    documents = build_service_guide_documents(guides)
    print(f"서비스 안내 문서 개수: {len(documents)}")
    build_service_guide_vector_store(documents, reset=True)
    print("서비스 안내 벡터 DB 재구축이 완료되었습니다.")


def rebuild_jeju_travel_index() -> None:
    print("제주 여행 PDF 텍스트 추출 및 문서 분할 중...")
    documents = load_jeju_travel_documents()
    print(f"제주 여행 검색 청크 개수: {len(documents)}")
    print("제주 여행 Chroma 컬렉션 초기화 및 Dense 임베딩 저장 중...")
    build_jeju_travel_vector_store(documents, reset=True)
    print("제주 여행 벡터 DB 재구축이 완료되었습니다.")


if __name__ == "__main__":
    main()
