# 제주 여행 PDF RAG 운영 안내

## 원본 자료 보관

제주 여행 PDF는 Git과 Docker 이미지에 포함하지 않는다. 개발 환경에서는
`data/jeju_travel`에 두고, 운영 서버에서는 저장소 밖의 비공개 디렉터리에
보관한다. `.gitignore`와 `.dockerignore`가 모든 하위 PDF를 제외한다.

운영 서버의 `.env`에는 비공개 PDF 디렉터리의 절대 경로를 지정한다.

```dotenv
JEJU_TRAVEL_PDF_DIRECTORY=/absolute/private/path/jeju-travel
```

Docker Compose는 이 경로를 컨테이너의 `/app/data/jeju_travel`에 읽기 전용으로
마운트한다. 따라서 GitHub Actions 배포 전에 해당 경로에 PDF가 있어야 한다.

## 인덱스 구축

```bash
.venv/bin/python rebuild_index.py --domain jeju-travel
```

텍스트 PDF는 내장 텍스트를 추출한다. 이미지 스캔 PDF는 Tesseract의 `kor+eng`
OCR을 사용한다. 최초 추출 결과는
`storage/vector_db/jeju_travel_extraction_cache`에 저장되므로 같은 PDF를 다시
색인할 때 OCR을 반복하지 않는다. PDF가 바뀌면 파일 해시가 바뀌어 자동으로 새로
추출한다.

OCR 동시 작업 수와 페이지별 제한 시간은 다음 값으로 조정할 수 있다.

```dotenv
JEJU_TRAVEL_OCR_WORKERS=3
JEJU_TRAVEL_OCR_PAGE_TIMEOUT_SECONDS=90
```

로컬 macOS에서는 `brew install tesseract tesseract-lang`이 필요하다. 운영 Docker
이미지는 Tesseract와 한국어 언어팩을 자체 설치한다.

## 답변 동작

제주 관광·음식·문화·자연 질문은 BGE-M3로 관련 청크를 찾고 Gemini가 이를
요약한다. 각 청크에는 내부 검증용 자료명과 PDF 페이지가 있지만 사용자 답변에는
출처 정보를 노출하지 않는다. 지역 표현과 장소명은 Dense 검색 결과에 키워드
점수를 더해 재정렬한다.

책에 적힌 영업시간, 요금, 휴무일, 교통편과 음식점 정보는 현재와 다를 수 있으므로
답변에서 방문 전 공식 채널 재확인을 안내한다. 실시간 날씨나 혼잡도는 PDF RAG가
확인한 것처럼 답하지 않는다.

준비 상태는 `GET /health`의 `indexes.jejuTravel`에서 확인한다. 제주 여행 인덱스가
없으면 `GET /ready`는 503을 반환한다.
