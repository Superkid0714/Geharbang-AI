from src.guesthouses.recommender import recommend_guesthouse
from src.guesthouses.vector_store import has_vector_store_documents


MISSING_VECTOR_DB_MESSAGE = "벡터 DB가 구축되어 있지 않습니다. python rebuild_index.py를 먼저 실행해 주세요."


def main() -> None:
    """Run an interactive search loop against an existing vector DB."""
    if not _is_vector_store_ready():
        print(MISSING_VECTOR_DB_MESSAGE)
        return

    print("게스트하우스 추천 챗봇을 시작합니다. 종료하려면 exit를 입력하세요.")

    while True:
        query = input("질문: ").strip()

        if query.casefold() == "exit":
            print("검색을 종료합니다.")
            return

        if not query:
            print("질문을 입력해 주세요.")
            continue

        try:
            answer = recommend_guesthouse(query)
        except (ValueError, RuntimeError, ImportError) as error:
            print(error)
            continue

        print(answer)


def _is_vector_store_ready() -> bool:
    try:
        return has_vector_store_documents()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
