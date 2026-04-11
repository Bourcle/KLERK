# KLERK (Korean Law Engine for Retrieval and Knowledge)

한국 법률 질의응답을 위한 LangGraph 기반 로컬 RAG 에이전트 스캐폴드

## 아키텍처 요약

### 1) 메모리 분리 전략
- **단기 메모리**: LangGraph checkpointer(SQLite)
  - thread 단위 대화 상태 저장
  - HITL / 재개 / 디버깅 / 회귀 실험에 적합
- **장기 메모리**: 별도 SQLite memory store
  - 중요도(`importance`) + 시간 감쇠(`half-life`) + 토픽 일치도(`topic match`)로 점수화
  - 사용자의 반복 관심사, 최근 법률 주제, ongoing issue를 보존
- **법률 지식 캐시**: Chroma persistent collections
  - `korean_law`
  - `law_constitution`
  - `law_civil`
  - `law_criminal`
  - `law_commercial`
  - `law_civil_procedure`
  - `law_criminal_procedure`
  - `korean_precedent`

### 2) 그래프 구조
- **Main Graph**
  - `normalize_and_route`
  - `memory_subgraph_wrapper`
  - `legal_rag_subgraph_wrapper`
  - `persist_turn_memory`
- **Memory Subgraph**
  - 관련 장기 메모리 검색
  - 점수 기반 상위 memory attach
- **Legal RAG Subgraph**
  - 벡터 DB 검색
  - sufficiency 판단
  - 부족 시 MCP 검색/상세조회
  - MCP 결과 캐싱
  - 최종 답변 생성

### 3) 질의 흐름
1. 질문을 LLM + 휴리스틱으로 라우팅한다.
2. 관련 long-term memory를 붙인다.
3. 적절한 collection에서 Chroma 검색을 수행한다.
4. 충분하지 않으면 Korean Law MCP를 호출한다.
5. 가져온 상세 텍스트를 직접 근거 문서로 사용한다.
6. 검색 문서와 memory를 함께 사용해 답변을 생성한다.

---

## 디렉토리 구조

```text
legal_memory_agent_repo/
├─ src/
│  ├─ app.py
│  ├─ service.py
│  ├─ router.py
│  ├─ llm_model/
│  │  └─ llm.py
│  ├─ storages/
│  │  └─ vector_store.py
│  ├─ utils/
│  │  ├─ config.py
│  │  ├─ exceptions.py
│  │  └─ logging_utils.py
│  ├─ graphs/
│  │  ├─ state.py
│  │  ├─ main_graph.py
│  │  └─ subgraphs/
│  │     ├─ legal_rag_subgraph.py
│  │     └─ memory_subgraph.py
│  └─ ui/
│     └─ gradio_app.py
├─ pyproject.toml
├─ .env.example
├─ README.md
└─ tests/
   ├─ test_memory_scoring.py
   ├─ test_parsers.py
   └─ test_router.py
```

---

## 사전 준비

### 1) Python 의존성 설치
```bash
uv sync
```

### 2) Korean Law MCP 서버 준비
이 프로젝트는 `woongaro/korean-law-mcp` 서버를 별도 디렉토리에 두고 stdio 방식으로 실행하는 구성을 기본값으로 가정한다.

예시:
```bash
mkdir -p external
cd external
git clone https://github.com/woongaro/korean-law-mcp.git
cd korean-law-mcp
npm install
cp .env.example .env
# .env에 LAW_API_OC 입력
npm run build
```

그 다음 루트 `.env`에서 다음 값을 맞춘다.
```env
LAW_API_OC=...
MCP_SERVER_COMMAND=node
MCP_SERVER_ENTRYPOINT=./external/korean-law-mcp/dist/index.js
```

### 3) 모델 준비

기본 채팅 모델을 Ollama로 쓸 경우:
```bash
ollama pull qwen3:8b
```

기본 임베딩은 로컬 Hugging Face 모델을 사용한다.

```env
LOCAL_EMBEDDING_PROVIDER=huggingface_local
LOCAL_EMB_MODEL=BAAI/bge-m3
```

채팅 모델을 OpenAI 호환 endpoint로 바꾸려면:

```env
LOCAL_LLM_PROVIDER=openai_compatible
LOCAL_LLM_MODEL=gpt-4o-mini
LOCAL_LLM_BASE_URL=https://api.openai.com/v1
LOCAL_LLM_API_KEY=...
```

Ollama를 계속 사용할 경우 앱은 첫 요청 시 `/api/tags` health check를 수행하고 다음을 구분해서 안내한다.

- Ollama 서버 미실행 또는 주소 오류
- 설정한 모델 미설치
- 기타 HTTP 연결 오류

---

## 실행

```bash
uv run python src/app.py
```

브라우저에서 Gradio UI가 열린다.

---

## 스트리밍 동작 방식

현재 UI는 async generator 기반으로 다음 두 단계를 분리한다.

1. 그래프 실행으로 최종 답변 생성
2. 생성된 답변을 **한 글자씩** 채팅창에 `yield`

즉, 토큰 레벨 streaming transport 가 아니더라도 사용자 체감상 스트리밍처럼 보이도록 구현했다.
채팅 히스토리는 Gradio messages 포맷으로 일관되게 유지한다.

---

## 메모리 점수 설계

장기 memory 검색 점수는 다음 요소를 곱해 계산한다.

- `importance`: 저장 시점 중요도
- `freshness`: 시간 감쇠 점수
- `topic_overlap`: 현재 질의와 memory 간 토픽 중첩
- `access_boost`: 과거 재사용 횟수
- `domain_boost`: 민사/형사/판례/헌법 등 도메인 일치 보너스

대략적인 형태는 다음과 같다.

```text
score = importance × freshness × (0.5 + topic_overlap) × domain_boost × access_boost
```

이 구조로 “시간이 지나면 약해지지만, 같은 토픽이면 다시 살아나는 memory” 실험이 가능하다.

---

## 다음 실험 포인트

1. memory store를 SQLite 대신 Postgres + pgvector로 변경
2. long-term memory를 별도 Chroma collection으로 이관
3. MCP search 결과를 structured artifact 기반으로 직접 파싱하도록 개선
4. 판례/조문 citation formatting 강화
5. HITL 승인 노드를 추가해 “MCP 호출 전 사용자 승인” 패턴 실험
