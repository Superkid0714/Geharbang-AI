# -*- coding: utf-8 -*-
"""review.py의 분석 로직을 API로 노출하는 FastAPI 서버.

로컬 실행:
    ./.venv/bin/uvicorn api:app --reload --port 8000

엔드포인트:
    GET  /health      상태 확인
    POST /analyze     리뷰 하나 -> 카테고리별 매칭된 긍정/부정/중립 키워드
    POST /report      숙소 리뷰 리스트 -> 사장님용 리포트 텍스트
    POST /categorize  리뷰(id 포함) 리스트 -> 카테고리별 리뷰 id 목록 (앱에서 카테고리 클릭 시 사용)
    POST /keywords    리뷰(id 포함) 리스트 -> "기타 특이 언급"(12개 카테고리 밖 TF-IDF 키워드)별 리뷰 id 목록
"""

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from review import (
    analyze_review_categories,
    build_house_report,
    group_reviews_by_category,
    group_reviews_by_tfidf_keyword,
)

app = FastAPI(title="리뷰 키워드 분석 API")


class AnalyzeRequest(BaseModel):
    review: str


class ReportRequest(BaseModel):
    house_name: str
    reviews: List[str]


class ReviewItem(BaseModel):
    id: int
    review: str


class CategorizeRequest(BaseModel):
    reviews: List[ReviewItem]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(payload: AnalyzeRequest):
    """리뷰 하나를 분석해서 카테고리별 매칭된 긍정/부정/중립 키워드를 반환"""
    matched = analyze_review_categories(payload.review)
    return {
        "categories": [
            {
                "category": category,
                "positive_keywords": pos,
                "negative_keywords": neg,
                "neutral_keywords": neutral,
            }
            for category, pos, neg, neutral in matched
        ]
    }


@app.post("/report")
def report(payload: ReportRequest):
    """숙소 하나의 리뷰 텍스트 리스트를 받아 사장님용 리포트 텍스트를 반환"""
    report_text = build_house_report(payload.house_name, payload.reviews)
    return {"house_name": payload.house_name, "report": report_text}


@app.post("/categorize")
def categorize(payload: CategorizeRequest):
    """리뷰 목록(id 포함)을 받아 카테고리별 리뷰 id 목록을 반환.

    앱에서 카테고리 버튼(예: '청결도')을 누르면, 여기서 받은 id 리스트로 원문 리뷰를
    DB에서 조회해서 보여주면 된다.
    """
    reviews = [item.model_dump() for item in payload.reviews]
    return group_reviews_by_category(reviews)


@app.post("/keywords")
def keywords(payload: CategorizeRequest):
    """리뷰 목록(id 포함)을 받아 "기타 특이 언급"(12개 카테고리 사전에 없는 TF-IDF 키워드)별 리뷰 id 목록을 반환.

    앱의 "기타 특이 언급" 섹션에서 키워드(예: '바베큐')를 누르면, 여기서 받은
    review_ids로 원문 리뷰를 DB에서 조회해서 보여주면 된다.
    """
    reviews = [item.model_dump() for item in payload.reviews]
    return group_reviews_by_tfidf_keyword(reviews)
