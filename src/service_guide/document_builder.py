from __future__ import annotations


def build_service_guide_documents(guides: list[dict]) -> list[dict]:
    return [
        {
            **guide,
            "content": (
                f"문서 제목: {guide['title']}\n"
                f"대상: {guide['audience']}\n"
                f"관련 화면: {guide['routes']}\n\n"
                f"{guide['content']}"
            ),
        }
        for guide in guides
    ]
