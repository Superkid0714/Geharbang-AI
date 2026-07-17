# -*- coding: utf-8 -*-
"""리뷰 텍스트에서 카테고리별 키워드를 뽑고 긍정/부정을 분석해 숙소별 리포트를 만드는 모듈.

API 서버 등 다른 코드에서 쓸 때는 이 파일을 import해서 필요한 함수만 호출하면 된다.

    from review import build_house_report
    report_text = build_house_report(house_name, review_texts)

터미널에서 직접 실행하면(`python review.py`) review_ai/리뷰 데이터.xlsx를 읽어서
review_ai/reports/*.txt 로 전체 숙소 리포트를 한 번에 생성한다 (로컬 개발/테스트용).
"""

import os
from collections import Counter
from datetime import date

import pandas as pd
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_ai")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

kiwi = Kiwi()

# TF-IDF 키워드 추출 시 제외할, 의미가 거의 없는 명사/형용사(불용어).
# 뒷부분은 실제 리뷰 데이터(review_ai/리뷰 데이터.xlsx) 분석 결과, 카테고리와 무관하게
# 어디서나 흔히 나오는 서술 명사/동사 어간을 추가한 것.
STOPWORDS = {
    "저", "제", "저희", "우리", "여기", "거기", "저기", "이곳", "그곳",
    "정말", "진짜", "너무", "완전", "그냥",
    "생각", "느낌", "정도", "번째", "이번", "다음", "때문", "동안", "이후", "이전",
    "사람", "분들", "분", "것", "거", "수", "때", "곳", "점", "부분",
    "숙소", "방문", "여행", "게스트", "예약", "시간", "혼자", "처음", "공간",
    "친구", "참여", "이용", "게하", "게스트하우스", "하우스", "추억", "이야기",
    "하루", "대화", "참석", "주변", "진행", "기대", "자체", "고민", "사진",
    "부담", "경험", "여자", "기억", "다양", "마음", "자리", "개인", "대비", "있",
    "없", "같", "많",
}

# 카테고리마다 그 자체로 긍/부정을 나타내는 키워드(positive/negative)와,
# 감성은 없지만 주제 언급 여부만 알려주는 키워드(neutral)를 구분한다.
# 실제 리뷰 데이터(review_ai/리뷰 데이터.xlsx) 형태소 분석 결과를 기준으로 커버리지를 넓힘.
CATEGORY_KEYWORDS = {
    "만족도": {
        # "좋았"(과거형만 매칭) -> 어간 "좋"으로 교체: 좋아요/좋네요/좋은 등도 매칭됨
        # "재밌게"(특정 활용형만 매칭) -> 어간 "재밌"으로 교체: 재밌었어요/재밌음 등도 매칭됨
        # "편안"/"불편" 신규 추가
        # "재미있"("재밌"의 비축약형), "편하"("편안"의 동사/형용사 어간) 신규 추가
        # "즐겁"만 있으면 됨: 토큰 매칭이 "즐거워요"도 kiwi가 정규화한 "즐겁"으로 잡아줘서 "즐거"는 불필요
        "positive": ["추천", "꿀잼", "재밌", "재미있", "즐겁", "최고", "좋", "만족", "감사", "편안", "편하"],
        "negative": ["후회", "별로", "비추천", "불만", "불편"],
        "neutral": [],
    },
    "서비스": {
        "positive": ["친절", "재밌으셔", "응대", "배려"],
        "negative": ["불친절", "무례"],
        # "직원"/"스태프"("스텝/스탭"의 동의어), "체크인" 신규 추가
        "neutral": ["스텝", "스탭", "사장", "매니저", "직원", "스태프", "체크인"],
    },
    "공연/전시": {
        "positive": [],
        "negative": [],
        "neutral": ["공연", "전시", "레크레이션", "강사", "게임", "술게임"],
    },
    "분위기": {
        # "힐링" 신규 추가
        "positive": ["친해지", "힐링"],
        # "시끄럽"만 있으면 됨: 토큰 매칭이 "시끄러워요"도 kiwi가 정규화한 "시끄럽"으로 잡아줌
        "negative": ["어색함", "시끄럽"],
        # "포틀럭"/"포트럭"(파티 형식 지칭) 신규 추가
        "neutral": ["파티", "불멍", "분위기", "게임", "술게임", "만남", "조용", "포틀럭", "포트럭"],
    },
    "주차공간": {
        "positive": [],
        "negative": [],
        "neutral": ["주차", "주차장", "주차비"],
    },
    "위치": {
        "positive": [],
        "negative": [],
        # "위치" 자체, "서쪽", "시장"/"동문시장" 신규 추가
        "neutral": [
            "제주", "공항", "성산", "애월", "협재", "해변", "근처", "거리", "편의",
            "위치", "서쪽", "시장", "동문시장","조용"
        ],
    },
    "가격": {
        # "비싸"/"아깝"만 있으면 됨: 토큰 매칭이 "비쌌어요"도 "비싸"로, "아까워요"도 kiwi가
        # 정규화한 "아깝"으로 잡아줘서 "비쌌"/"아까워" 별도 추가 불필요
        "positive": ["합리", "가성비", "저렴"],
        "negative": ["비싸", "아깝"],
        "neutral": ["가격", "비용"],
    },
    "청결도": {
        "positive": ["깨끗", "깔끔", "편하"],
        # "먼지", "머리카락" 신규 추가. "더럽"만 있으면 됨: 토큰 매칭이 "더러워요"도
        # kiwi가 정규화한 "더럽"으로 잡아줘서 "더러" 별도 추가 불필요
        "negative": ["더럽", "곰팡이", "냄새", "먼지", "머리카락"],
        "neutral": ["화장실", "침구", "수건", "타일","잠자리"],
    },
    "편의시설": {
        "positive": [],
        # "춥"/"덥"만 있으면 됨: 토큰 매칭이 "추워요"/"더워요"도 kiwi가 정규화한 "춥"/"덥"으로
        # 잡아줘서 "추워"/"더워" 별도 추가 불필요
        "negative": ["춥", "덥"],
        # "객실", "공용"(공용 화장실/공간 등), "도미토리"(방 형태 지칭) 신규 추가
        "neutral": [
            "수영장", "시설", "에어컨", "보일러", "베란다", "슬리퍼", "엘베", "드라이기",
            "객실", "공용", "도미토리",
        ],
    },
    "전망": {
        "positive": [],
        "negative": [],
        "neutral": ["오션뷰", "바다", "뷰", "경치", "해돋이", "풍경"],
    },
    "음식/조식": {
        "positive": ["맛있"],
        "negative": [],
        # "맛집"(주변 맛집 추천 언급) 신규 추가
        "neutral": ["음식", "안주", "삼겹살", "무한리필", "고기", "조식", "저녁", "메뉴", "맛집"],
    },
    "비품": {
        "positive": [],
        "negative": [],
        # "커튼"(침대별 커튼 언급) 신규 추가
        "neutral": ["침대", "옷장", "냉장고", "바디워시", "샴푸", "치약", "비누", "타올", "고데기", "커튼"],
    },
}

# CATEGORY_KEYWORDS에 이미 등록된 모든 단어. TF-IDF로 "기타 특이 언급"을 뽑을 때
# 12개 카테고리와 겹치는 단어(예: "친절", "곰팡이")를 걸러내는 데 쓴다.
_ALL_CATEGORY_KEYWORDS = {
    kw for spec in CATEGORY_KEYWORDS.values() for kws in spec.values() for kw in kws
}

# 카테고리에 부정 언급이 있을 때 종합 보완점에 쓸 제안 문장
IMPROVEMENT_SUGGESTIONS = {
    "만족도": "전반적인 만족도를 끌어올릴 수 있는 부분을 점검해보면 좋겠습니다.",
    "서비스": "응대 방식과 친절도 관련 교육을 보완하면 좋겠습니다.",
    "분위기": "소음 관리와 다소 어색한 분위기를 풀어주는 프로그램 보완이 필요합니다.",
    "가격": "가격 대비 만족도를 높일 수 있는 구성이나 가격 조정을 검토해보면 좋겠습니다.",
    "청결도": "청소 상태와 환기·탈취 관리를 강화하면 좋겠습니다.",
}
DEFAULT_IMPROVEMENT_SUGGESTION = "관련 불편 사항을 점검해보면 좋겠습니다."


def extract_content_words(text):
    """형태소 분석 결과에서 명사(N)와 형용사 어간(VA)을 (형태, 품사) 쌍으로 추출.

    kiwi.tokenize()가 한 번에 모든 품사를 태깅하므로, 태그 필터를 늘려도 kiwi 호출
    자체가 늘지 않아 속도 영향은 거의 없다. "친절"/"깨끗" 같은 감성 표현은 명사가
    아니라 형용사 어간이라 VA도 포함해야 TF-IDF 대표 키워드 후보에 잡힌다.
    """
    return [
        (token.form, token.tag) for token in kiwi.tokenize(text)
        if token.tag.startswith('N') or token.tag == 'VA'
    ]


def _filter_content_words(tagged_words, min_len):
    """불용어와 짧은 명사를 거른다.

    형용사 어간(VA)은 "좋"/"많"처럼 1글자가 정상적인 형태라 길이 제한을 안 걸고,
    명사(N)만 min_len 이상으로 걸러서 의미 없는 파편을 줄인다.
    """
    return [
        form for form, tag in tagged_words
        if form not in STOPWORDS and (tag == 'VA' or len(form) >= min_len)
    ]


def preprocess_reviews(reviews, min_len=2):
    """리뷰 텍스트들에서 명사/형용사 어간을 뽑아 불용어/한 글자 명사를 제거하고 공백으로 합친 문자열 리스트를 반환"""
    processed = []
    for review in reviews:
        filtered = _filter_content_words(extract_content_words(str(review)), min_len)
        if filtered:
            processed.append(" ".join(filtered))
    return processed


def preprocess_reviews_with_ids(reviews, min_len=2):
    """[{"id": ..., "review": "..."}, ...] 리스트를 받아, 명사/형용사 어간 추출/불용어 제거 후

    내용이 남은 리뷰만 (id, 전처리된 텍스트) 쌍으로 반환. 필터링으로 일부 리뷰가
    통째로 빠질 수 있어 인덱스로 원본과 대응시킬 수 없기 때문에, id를 같이 들고 다닌다.
    """
    pairs = []
    for item in reviews:
        filtered = _filter_content_words(extract_content_words(str(item["review"])), min_len)
        if filtered:
            pairs.append((item["id"], " ".join(filtered)))
    return pairs


def get_tfidf_scores(processed_reviews, top_n=None, exclude=frozenset()):
    """전처리된 리뷰 리스트에서 TF-IDF 점수 상위 (단어, 점수) 목록을 반환.

    exclude에 들어있는 단어는 결과에서 제외한다 (12개 카테고리 사전과 겹치는 단어를
    빼고 "사전 밖의 특이 언급"만 남기는 용도).
    """
    # 기본 token_pattern(\b\w\w+\b)은 1글자 토큰을 버리는데, "좋"/"많"처럼 1글자
    # 형용사 어간도 걸러졌기 때문에 1글자도 허용하는 패턴으로 바꿈. 전처리 단계에서
    # 이미 형태소 분석+불용어 제거를 끝냈으니 여기서 더 걸러낼 필요가 없다.
    # min_df=2: 리뷰 1개에만 우연히 나온 단어는 "특이 언급"에서 제외하고,
    # 최소 2개 이상 리뷰에서 반복된 단어만 후보로 남긴다.
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=2)
    try:
        x = vectorizer.fit_transform(processed_reviews)
    except ValueError:
        return []

    words = vectorizer.get_feature_names_out()
    scores = x.sum(axis=0).A1
    sorted_words = sorted(
        ((w, s) for w, s in zip(words, scores) if w not in exclude),
        key=lambda pair: pair[1], reverse=True,
    )
    return sorted_words[:top_n] if top_n else sorted_words


def get_house_tfidf_keywords(reviews, top_n=10):
    """숙소 리뷰 텍스트 리스트를 받아, 12개 카테고리 사전에 없는 TF-IDF 기준 대표 키워드("기타 특이 언급")를 반환"""
    processed = preprocess_reviews(reviews)
    if not processed:
        return []
    return get_tfidf_scores(processed, top_n=top_n, exclude=_ALL_CATEGORY_KEYWORDS)


def group_reviews_by_tfidf_keyword(reviews, top_n=10):
    """리뷰 목록을 받아, 12개 카테고리 사전에 없는 TF-IDF 대표 키워드("기타 특이 언급")별로

    해당 리뷰의 id를 모아서 반환한다. CATEGORY_KEYWORDS에 이미 있는 단어는 제외한다
    (그건 group_reviews_by_category가 이미 카테고리 버튼으로 처리하기 때문).

    reviews: [{"id": ..., "review": "..."}, ...] 형태의 리스트 (DB에서 가져온 리뷰 행 그대로 넘기면 됨)
    반환: [{"keyword": "바베큐", "score": 1.41, "review_ids": [12, 45]}, ...] (점수 높은 순, top_n개)

    앱에서 "기타 특이 언급" 섹션의 키워드(예: '바베큐')를 클릭하면, 여기서 받은
    review_ids로 원문 리뷰를 조회해서 화면에 보여주면 된다.
    """
    id_text_pairs = preprocess_reviews_with_ids(reviews)
    if not id_text_pairs:
        return []
    ids, processed = zip(*id_text_pairs)

    # 기본 token_pattern(\b\w\w+\b)은 1글자 토큰을 버리는데, "좋"/"많"처럼 1글자
    # 형용사 어간도 걸러졌기 때문에 1글자도 허용하는 패턴으로 바꿈. 전처리 단계에서
    # 이미 형태소 분석+불용어 제거를 끝냈으니 여기서 더 걸러낼 필요가 없다.
    # min_df=2: 리뷰 1개에만 우연히 나온 단어는 "특이 언급"에서 제외하고,
    # 최소 2개 이상 리뷰에서 반복된 단어만 후보로 남긴다.
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=2)
    try:
        x = vectorizer.fit_transform(processed)
    except ValueError:
        return []

    words = vectorizer.get_feature_names_out()
    scores = x.sum(axis=0).A1
    candidate_cols = [i for i in range(len(words)) if words[i] not in _ALL_CATEGORY_KEYWORDS]
    top_cols = sorted(candidate_cols, key=lambda i: scores[i], reverse=True)[:top_n]

    result = []
    for col in top_cols:
        row_indices = x[:, col].nonzero()[0]
        review_ids = [ids[row] for row in row_indices]
        result.append({
            "keyword": words[col],
            "score": round(float(scores[col]), 4),
            "review_ids": review_ids,
        })
    return result


# 부정어 처리: 긍/부정 키워드가 부정문 안에 있으면 매칭에서 빼는 게 아니라 반대 극성으로 뒤집는다.
#   "안 더러워요" -> "더러"가 있어도 부정(negative)이 아니라 긍정으로 카운트
#   "친절하지 않았어요" -> "친절"이 있어도 긍정이 아니라 부정으로 카운트
#   "냄새 없이 좋았어요" -> "냄새"가 있어도 부정이 아니라 긍정으로 카운트
#
# 키워드 매칭은 리뷰를 kiwi로 토큰화한 뒤, 토큰의 form이 키워드와 정확히 일치하는지로 판단한다
# (사전 키워드 128개 중 110개가 토큰 하나로 정확히 떨어짐). 이러면:
#   1) 부정어가 정확히 "이 단어를 부정하는지"를 품사 태그로 판단할 수 있다. 글자 수 기반이면
#      "냄새 없이 좋았어요"에서 "없이"(부사, MAG)가 "좋았어요"까지 같이 오염시키는 문제가 있었는데,
#      "없이"(MAG)와 "없다"의 "없"(VA)은 품사가 달라서 구분된다.
#   2) kiwi가 불규칙 활용을 원형으로 정규화해주기 때문에("추워요"->"춥", "즐거워요"->"즐겁"),
#      활용형을 따로 안 넣어도 원형 하나로 다 잡힌다.
# "꿀잼"/"오션뷰"처럼 토큰 2개 이상으로 쪼개지는 복합어는 토큰 매칭으로 못 잡으므로,
# 문자열 부분검색(+글자 수 기반 부정 판단)으로 폴백한다.
NEGATION_ADVERBS = {"안", "못"}  # 서술어 직전에 붙는 부정 부사 (짧은 부정문: "안 친절해요")
NEGATION_AUX_VERBS = {"않", "못하", "아니"}  # "-지" 뒤에 붙는 부정 보조용언 (긴 부정문: "친절하지 않았어요")
NEGATION_WORDS_FALLBACK = ["안", "않", "못", "없", "아니"]  # 복합어 폴백에서만 쓰는 글자 수 기반 부정어
NEGATION_BEFORE = 4
NEGATION_AFTER = 7


def _is_negated_by_chars(review, start_idx, end_idx):
    """토큰 매칭이 안 되는 복합어 키워드용 폴백. 글자 수 기반이라 오염 가능성이 있지만 예외 케이스만 처리."""
    before_text = review[max(0, start_idx - NEGATION_BEFORE):start_idx]
    after_text = review[end_idx:end_idx + NEGATION_AFTER]
    return any(neg in before_text for neg in NEGATION_WORDS_FALLBACK) or any(
        neg in after_text for neg in NEGATION_WORDS_FALLBACK
    )


def _is_negated_at_token(tokens, ti):
    """ti번째 토큰이 부정됐는지, 인접 토큰의 품사/형태로 정확히 판단"""
    # 직전 토큰이 "안"/"못"(부사)이면 짧은 부정문: "안 친절해요"
    if ti > 0:
        prev = tokens[ti - 1]
        if prev.tag == 'MAG' and prev.form in NEGATION_ADVERBS:
            return True

    # 직후: 파생접미사(XSA, "친절+하"의 "하")/선어말어미(EP)/조사(JKS,JKO,JX)는
    # 의미 없이 문법적으로만 붙는 것들이라 건너뛰고, 그다음 토큰에서 부정 패턴을 찾는다.
    # (너무 멀리까지 건너뛰지 않도록 최대 3칸으로 제한)
    j = ti + 1
    while j < len(tokens) and j - ti <= 3 and tokens[j].tag in ('XSA', 'EP', 'JKS', 'JKO', 'JX'):
        j += 1

    if j < len(tokens):
        tok = tokens[j]
        # "없이"(부사): "냄새 없이" -> 명사를 부정 (있다/없다의 "없"과는 다른 토큰이라 별도 체크)
        if tok.tag == 'MAG' and tok.form == '없이':
            return True
        # "없다"(서술어): "곰팡이가 없어요" -> 명사를 부정
        if tok.tag == 'VA' and tok.form == '없':
            return True
        # "-지"(EC) + "않/못하/아니"(VX): 긴 부정문 ("친절하지 않았어요")
        if tok.tag == 'EC' and tok.form == '지' and j + 1 < len(tokens):
            nxt = tokens[j + 1]
            if nxt.tag == 'VX' and nxt.form in NEGATION_AUX_VERBS:
                return True

    return False


def _match_sentiment_keywords(review, tokens, token_positions, keywords):
    """긍/부정 키워드 리스트 중 실제로 등장한 것을, 부정 여부와 함께 나눠서 반환.

    반환: (부정 없이 매칭된 키워드 리스트, 부정어로 극성이 뒤집힌 키워드 리스트)
    """
    matched, negated = [], []
    for keyword in keywords:
        positions = token_positions.get(keyword)
        if positions:
            if any(not _is_negated_at_token(tokens, ti) for ti in positions):
                matched.append(keyword)
            else:
                negated.append(keyword)
            continue

        # 토큰 하나로 안 잡히는 복합어 -> 문자열 검색 폴백
        start = 0
        found_clean = found_negated = False
        while True:
            idx = review.find(keyword, start)
            if idx == -1:
                break
            end_idx = idx + len(keyword)
            if _is_negated_by_chars(review, idx, end_idx):
                found_negated = True
            else:
                found_clean = True
                break
            start = end_idx
        if found_clean:
            matched.append(keyword)
        elif found_negated:
            negated.append(keyword)
    return matched, negated


def analyze_review_categories(review):
    """리뷰 하나에서 언급된 카테고리와, 그 안에서 실제로 매칭된 긍정/부정/중립 키워드를 반환.

    부정어로 극성이 뒤집힌 경우("냄새 없이" -> 청결도 긍정, "친절하지 않았어요" -> 서비스 부정)
    반대 극성 리스트에 합쳐서 넣는다.
    """
    if pd.isna(review):
        return []
    review = str(review)
    tokens = kiwi.tokenize(review)
    token_positions = {}
    for i, tok in enumerate(tokens):
        token_positions.setdefault(tok.form, []).append(i)

    result = []
    for category, spec in CATEGORY_KEYWORDS.items():
        pos_clean, pos_negated = _match_sentiment_keywords(review, tokens, token_positions, spec["positive"])
        neg_clean, neg_negated = _match_sentiment_keywords(review, tokens, token_positions, spec["negative"])
        # 부정어로 뒤집힌 키워드는 원래 단어 그대로 두면("불친절"이 장점에 뜸) 리포트를 읽는
        # 사장님이 오해하기 쉬워서, "OO 아님"으로 표시해 극성이 반대로 뒤집혔음을 드러낸다.
        matched_pos = pos_clean + [f"{kw} 아님" for kw in neg_negated]
        matched_neg = neg_clean + [f"{kw} 아님" for kw in pos_negated]
        matched_neutral = [kw for kw in spec["neutral"] if kw in review]
        if not matched_pos and not matched_neg and not matched_neutral:
            continue
        result.append((category, matched_pos, matched_neg, matched_neutral))

    return result


def group_reviews_by_category(reviews):
    """리뷰 목록을 받아 카테고리별로 해당 리뷰의 id를 모아서 반환한다.

    reviews: [{"id": ..., "review": "..."}, ...] 형태의 리스트 (DB에서 가져온 리뷰 행 그대로 넘기면 됨)
    반환: {"청결도": [12, 45, 78], "서비스": [3, 12, ...], ...}

    앱에서 카테고리(예: '청결도')를 클릭하면, 여기서 받은 id 리스트로 원문 리뷰를 조회해서
    화면에 보여주면 된다.
    """
    grouped = {}
    for item in reviews:
        matched = analyze_review_categories(item["review"])
        for category, _pos_kws, _neg_kws, _neutral_kws in matched:
            grouped.setdefault(category, []).append(item["id"])

    return grouped


def build_house_report(house_name, reviews):
    """숙소 하나의 리뷰 텍스트 리스트를 받아 리포트 문자열을 반환한다.

    API에서 새 리뷰가 들어올 때마다 그 숙소의 전체 리뷰 리스트를 다시 모아서
    이 함수를 호출하면 최신 리포트를 얻을 수 있다.
    """
    mention_count = Counter()
    positive_keyword_count = {}
    negative_keyword_count = {}
    neutral_keyword_count = {}
    negative_examples = {}

    for review in reviews:
        for category, pos_kws, neg_kws, neutral_kws in analyze_review_categories(review):
            mention_count[category] += 1
            if pos_kws:
                positive_keyword_count.setdefault(category, Counter()).update(pos_kws)
            if neg_kws:
                negative_keyword_count.setdefault(category, Counter()).update(neg_kws)
                examples = negative_examples.setdefault(category, [])
                if len(examples) < 3:
                    examples.append(str(review)[:120])
            if neutral_kws:
                neutral_keyword_count.setdefault(category, Counter()).update(neutral_kws)

    categories = sorted(mention_count, key=lambda c: mention_count[c], reverse=True)

    title = f" {house_name} 리뷰 분석 리포트 "
    divider = "=" * max(len(title), 30)

    report_lines = [
        divider,
        title,
        divider,
        f"분석 리뷰 수: {len(reviews)}건",
        f"생성일: {date.today().isoformat()}",
        "",
        "[카테고리별 요약] (언급 많은 순, 0건 항목은 생략)",
    ]
    for rank, category in enumerate(categories, start=1):
        pos_counter = positive_keyword_count.get(category)
        neg_counter = negative_keyword_count.get(category)
        neutral_counter = neutral_keyword_count.get(category)

        report_lines.append("")
        report_lines.append(f"{rank}. {category} ({mention_count[category]}건 언급)")

        if pos_counter:
            top_pos = ", ".join(f"'{kw}'" for kw, _ in pos_counter.most_common(3))
            report_lines.append(f"   장점: {top_pos} 등 긍정적인 평가가 많았습니다. ({sum(pos_counter.values())}건)")

        if neg_counter:
            top_neg = ", ".join(f"'{kw}'" for kw, _ in neg_counter.most_common(3))
            report_lines.append(f"   단점: {top_neg} 관련 불만이 있었습니다. ({sum(neg_counter.values())}건)")
            for example in negative_examples.get(category, []):
                report_lines.append(f"     - 부정 리뷰: {example}")

        if not pos_counter and not neg_counter:
            if neutral_counter:
                top_neutral = ", ".join(f"'{kw}'" for kw, _ in neutral_counter.most_common(3))
                report_lines.append(f"   내용: {top_neutral} 등이 자주 언급됐습니다.")
            else:
                report_lines.append("   내용: 단순 언급만 확인됐습니다.")

    # 형태소 분석+TF-IDF로 12개 카테고리 사전에 없는 특이 언급을 추가로 발견
    tfidf_keywords = get_house_tfidf_keywords(reviews, top_n=10)
    report_lines.append("")
    report_lines.append("[기타 특이 언급]")
    if tfidf_keywords:
        report_lines.append(", ".join(f"'{word}'" for word, _score in tfidf_keywords))
    else:
        report_lines.append("추출된 키워드가 없습니다.")

    # 종합 보완점: 숙소 전체에서 부정 언급이 가장 많았던 카테고리 기준으로 2줄 요약
    report_lines.append("")
    report_lines.append("[종합 보완점]")
    if negative_keyword_count:
        ranked_negative_categories = sorted(
            negative_keyword_count, key=lambda c: sum(negative_keyword_count[c].values()), reverse=True
        )[:2]
        category_summary = ", ".join(
            f"'{c}'(부정 {sum(negative_keyword_count[c].values())}건)" for c in ranked_negative_categories
        )
        suggestions = " ".join(
            IMPROVEMENT_SUGGESTIONS.get(c, DEFAULT_IMPROVEMENT_SUGGESTION) for c in ranked_negative_categories
        )
        report_lines.append(f"가장 개선이 필요한 부분은 {category_summary}입니다.")
        report_lines.append(suggestions)
    else:
        report_lines.append("뚜렷한 불만 언급이 없어 특별한 보완점은 발견되지 않았습니다.")

    return "\n".join(report_lines) + "\n"


def _run_local_batch():
    """review_ai/리뷰 데이터.xlsx 를 읽어서 전체 숙소 리포트를 한 번에 생성한다 (로컬 개발/테스트용)."""
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    excel_file_path = os.path.join(BASE_DIR, '리뷰 데이터.xlsx')
    df_excel = pd.read_excel(excel_file_path)

    house_names = df_excel['house_name'].dropna().unique()

    for house_name in house_names:
        house_df = df_excel[df_excel['house_name'] == house_name]
        reviews_text = house_df['review'].dropna().astype(str).tolist()
        reviews_with_id = house_df[['id', 'review']].dropna(subset=['review']).to_dict('records')

        # 터미널에는 "카테고리 -> 리뷰 id" 매칭 결과만 출력 (앱에서 키워드 클릭 시 쓸 데이터)
        grouped = group_reviews_by_category(reviews_with_id)
        print(f"\n[{house_name}] 카테고리 -> 리뷰 id")
        for category, ids in grouped.items():
            print(f"  {category}: {ids}")

        # 리포트 파일은 그대로 저장 (터미널에는 출력하지 않음)
        report_text = build_house_report(house_name, reviews_text)
        report_path = os.path.join(REPORT_DIR, f"{house_name}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

    print(f"\n✅ 숙소 {len(house_names)}개 처리 완료. 리포트는 '{REPORT_DIR}'에 저장했습니다.")


if __name__ == "__main__":
    _run_local_batch()
