# Geharbang RAG

게하르방의 AI 채팅 프로젝트입니다. 현재 공통 라우터가 질문을 게스트하우스,
스텝 공고, 제주 관광, 게하르방 이용 안내로 분류합니다. 실제 RAG 답변은 우선
게스트하우스와 스텝 공고는 백엔드 DB의 활성 데이터에 연결되어 있고,
게하르방 이용 안내는 검증된 Markdown 문서 기반 RAG에 연결되어 있습니다.

## 설치

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`.env.example`을 참고해 `.env`에 Gemini API 키, 백엔드 주소와 `AI_DATA_KEY`를
설정합니다. 백엔드에도 동일한 `AI_DATA_KEY` 환경 변수가 필요합니다.

## 실행

```bash
python rebuild_index.py --domain all
python main.py
```

## HTTP API

```bash
.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

- `GET /health`: Gemini 및 인덱스 준비 상태
- `GET /ready`: 운영 트래픽을 받을 준비가 되었는지 확인 (미준비 시 503)
- `POST /chat`: 질문 전송
- `DELETE /chat/{sessionId}`: 대화 상태 초기화

요청 예시:

```json
{
  "message": "애월에서 한 달 일할 스텝 공고 추천해줘",
  "sessionId": null
}
```

스텝 공고만 갱신하려면 다음 명령을 사용합니다.

```bash
python rebuild_index.py --domain staff-steps
```

서비스 안내 문서만 갱신하려면 다음 명령을 사용합니다.

```bash
python rebuild_index.py --domain service-guide
```

## DB 실시간 동기화

백엔드에서 활성 게스트하우스나 스텝 공고가 등록·수정·활성화되면 트랜잭션
커밋 후 AI 내부 API가 해당 문서 하나만 upsert합니다. 비활성화·삭제 시에는 해당
문서를 인덱스에서 제거합니다. 백엔드와 AI 서버의 `AI_DATA_KEY`는 같아야 하며,
백엔드의 `GEHARBANG_AI_BASE_URL`은 이 FastAPI 서버를 가리켜야 합니다.

전달은 사용자 요청과 분리된 비동기 작업으로 실행되고 일시적 실패 시 최대 3번
재시도합니다. 전체 `rebuild_index.py` 명령은 초기 구축과 정합성 복구용으로
계속 사용할 수 있습니다. 또한 AI 서버는 기본 6시간마다 운영 DB와 동적 인덱스를
대조해 누락·변경·삭제 문서를 자동 복구합니다. 주기는
`INDEX_RECONCILE_INTERVAL_SECONDS`로 조정할 수 있습니다.

채팅 답변에서는 Chroma가 추천 후보 ID만 선정합니다. 지역·가격·객실·파티처럼
정확해야 하는 조건은 최신 백엔드 DB 응답으로 먼저 필터링하고, 최종 후보의 주소·
가격·운영 정보도 최신 DB 문서로 다시 교체한 뒤 Gemini가 설명합니다. 따라서 인덱스
갱신이 잠시 늦더라도 오래된 상세 정보를 그대로 답변하지 않습니다.

## 운영 배포

운영 환경에서는 AI 포트를 인터넷에 직접 공개하지 않습니다. 앱은 기존 HTTPS
주소인 `https://geharbang.org/api/v1/ai/chat`을 호출하고, Spring 백엔드가 같은
Docker `guesthouse` 네트워크의 `http://geharbang-rag-ai:8001`로 요청을 전달합니다.
내부 색인 동기화 API도 이 네트워크 안에서만 사용합니다.

운영 서버에서 필요한 값은 다음과 같습니다.

- RAG 컨테이너: `GEMINI_API_KEY`, `AI_DATA_KEY`
- Spring 백엔드: `AI_DATA_KEY`를 RAG와 동일하게 설정
- Spring 백엔드: `GEHARBANG_AI_BASE_URL=http://geharbang-rag-ai:8001`

AI 저장소의 GitHub Actions secrets에도 `GEMINI_API_KEY`와 `AI_DATA_KEY`를
등록해야 합니다. `rag` 브랜치가 배포되면 Docker 이미지를 만들고, 영구 볼륨을
사용해 Chroma DB와 Hugging Face 모델 캐시를 유지하며, 운영 DB 기준으로 최초
인덱스를 구축합니다. BGE-M3 모델을 한 프로세스에서 공유하므로 Uvicorn worker는
1개로 고정합니다.

### 공개 채팅 요청 제한

Gemini 비용 악용을 막기 위해 운영 Nginx의 `http` 블록에 IP별 요청 영역과
초과 상태 코드를 설정합니다.

```nginx
limit_req_zone $binary_remote_addr zone=ai_chat_per_ip:10m rate=12r/m;
limit_req_status 429;
```

`geharbang.org`의 `server` 블록에서는 AI 채팅 경로에만 순간 6회까지 허용합니다.

```nginx
location = /api/v1/ai/chat {
    limit_req zone=ai_chat_per_ip burst=6 nodelay;
    proxy_pass http://api;
    proxy_connect_timeout 5s;
    proxy_read_timeout 100s;
}
```

운영 서버에서 수동으로 확인하려면 다음 명령을 사용합니다.

```bash
docker compose -p geharbang-rag -f docker-compose.prod.yml ps
curl http://127.0.0.1:8001/ready
```

## 테스트

라우터 테스트는 외부 API나 임베딩 모델 없이 실행할 수 있습니다.

```bash
python -m unittest discover -s tests
```
