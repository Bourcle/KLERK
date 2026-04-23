# KLERK (Korean Law Engine for Retrieval and Knowledge)

Retrieval harness engineering 중심의 Korean legal QA agent.

좋은 모델 하나에 의존하는 대신, **검색 범위 제한 → query rewrite → reranking → sufficiency 판단 → ReAct-style fallback loop → MCP augmentation** 으로 이어지는 retrieval harness를 통해 법률 QA의 안정성을 높이는 시스템이다.

---

## 문제 인식 (Situation)

법률 문서는 구조와 용어가 반복되는 규격화 문서다. 이로 인해 naive vector retrieval만으로는 겉으로 비슷하지만 실제 질문과 맞지 않는 문서가 다수 검색된다.

- **법률 문서의 높은 구조적 유사성**: 조문 형식, 법률 용어가 유사해 embedding 공간에서 가까이 위치하지만 실제 맥락이 다른 경우가 빈번
- **약어/축어/조문 참조**: 민소, 형소, 헌재 등의 축약어와 "제N조" 형태 참조가 one-shot retrieval을 어렵게 만듦
- **법률 MCP 단독 사용의 한계**: 데이터 공백과 품질 편차가 있어 단독 검색 수단으로 불충분
- **초기 성능**: BERT 0.5, BLEURT 0.21 수준 (GPT reference: BERT 0.65, BLEURT 0.35)

## 원인 분석 (Task)

실제 병목은 모델 성능이 아니라 **retrieval/control harness의 부재**에 있었다.

- One-shot retrieval은 법률 도메인에서 안정적 근거 확보가 어려움
- 검색 실패 = 답변 실패로 직결되는 구조
- 검색 결과의 충분성을 판단하지 않고 바로 답변을 생성
- MCP를 primary backend로 사용하면 데이터 공백에 취약

## 해결 과정 (Action)

### 1. 도메인/주제 기반 Collection 분리

법률 도메인별로 Chroma collection을 분리해 검색 범위를 구조적으로 제한한다.

| Collection | 대상 |
|---|---|
| `korean_law` | 기본 (fallback) |
| `law_constitution` | 대한민국헌법 |
| `law_civil` | 민법 |
| `law_criminal` | 형법 |
| `law_commercial` | 상법 |
| `law_civil_procedure` | 민사소송법 |
| `law_criminal_procedure` | 형사소송법 |
| `korean_precedent` | 판례 |

질문을 먼저 분류하여 어떤 collection을 우선 검색할지 결정하는 routing 로직(LLM + 휴리스틱)을 적용한다.

### 2. Summary + Provenance Chunking

각 chunk에 출처 설명 요약과 provenance 정보를 함께 저장한다.

```text
<요약>
민법 제750조 (불법행위의 내용)
문서유형: 법령 | 조문번호: 제750조 | 위치: 제5편 불법행위 > 제1장 총칙
</요약>

<출처>
다음 조항은 **민법**의 **제5편 불법행위 > 제1장 총칙**에서 발췌한 내용입니다.
</출처>

<법률조항>
(원문 조문)
</법률조항>
```

Embedding은 summary + 출처 + 원문 전체 기준으로 수행되어 검색 정확도를 높인다.

### 3. Query Rewriting

질문을 검색 친화적인 질의로 변환하는 단계를 추가한다.

- 원문 질문은 유지하되, retrieval용 rewritten query를 별도 생성
- 약어 전개 (민소 → 민사소송법, 헌재 → 헌법재판소)
- 법률 용어 보강, 회화체 제거
- 조문형/판례탐색형 질의에 맞게 최적화

### 4. Reranking

Initial retrieval 후 LLM 기반 reranking으로 query relevance를 강화한다.

- 상위 문서들의 query 관련성을 0-10 스케일로 평가
- 단순 similarity score 재정렬이 아닌, 법률적 관련성 기준 재배치
- 현재 스택(LangChain + LLM)에 맞는 practical reranking

### 5. Sufficiency 판단

검색 결과의 근거 충분성을 판단한다.

- 최소 문서 수 + similarity threshold 기반 사전 필터링
- LLM 기반 충분성 판정: 문서가 질문의 핵심 법적 쟁점을 다루는지 평가
- insufficient 판정 시 suggested_action과 함께 반환

### 6. ReAct-style Fallback Loop

검색 실패 시 자동으로 재검색/정제가 일어나는 loop를 구현한다.

```text
rewrite_query → retrieve → rerank → judge_sufficiency
                                          ↓
                                    [sufficient] → answer
                                    [insufficient, retries left] → fallback → retrieve (loop)
                                    [exhausted] → answer (with uncertainty)
```

Fallback 전략 (우선순위):
1. **Query refinement**: 다른 관점/용어로 질의 재작성
2. **Collection broadening**: 특정 법 collection → default collection 확장
3. **MCP augmentation**: 법률 API를 통한 보조 검색
4. Max iteration 초과 시 불확실성을 드러내는 답변 생성

핵심은 **"한 번 검색 실패 = 답변 실패"가 되지 않는 구조**다.

### 7. Legal MCP를 보조 채널로 통합

- MCP는 primary retrieval이 아닌 **context augmentation용 보조 채널**
- 벡터 검색이 충분하면 MCP를 호출하지 않음
- 벡터 검색 부족 시에만 fallback loop에서 MCP 활용
- MCP에서 성공적으로 가져온 법률 정보는 **vector store에 적재하여 후속 재사용**

### 8. Grounded Answer Generation

- 확보된 근거를 법령/판례/헌재결정 유형별로 구분하여 context 조립
- 근거가 부족하면 부족하다고 명시하고 hallucination 방지
- 모든 답변에 출처 근거를 기반으로 생성

### 9. Structured Logging / Observability

모든 retrieval harness 단계를 structured JSON logging으로 추적한다.

| 단계 | 로그 항목 |
|---|---|
| Routing | route, source_type, topic, selected_collection |
| Query Rewrite | rewritten_query |
| Retrieval | retrieval_count, top_score, selected_collection |
| Reranking | rerank_scores |
| Sufficiency | sufficiency_decision, sufficiency_reason |
| Fallback Loop | fallback_action, fallback_iteration |
| MCP | mcp_called, mcp_upserted |

## 결과 (Result)

- 일반 법률 QA 50문항 기준 **BERT 0.63, BLEURT 0.32**
- MCP-only 대비 약 **10~15% 개선**
- 단일 검색 실패가 바로 답변 실패로 이어지지 않는 **더 안정적인 retrieval 흐름** 확보

### 배운 점

좋은 모델 하나보다 **retrieval control/harness engineering**이 더 중요하다.

향후 주석서, 해설본, 공개 판결문 확장 시 **판결문 초안 작성 보조**까지 확장 가능하다.

이 경험이 회사의 AI agent 개발과 evaluation pipeline 개선에도 이어져:
- RACE 20% 상승
- FACT 400% 상승
- BERT 10% 상승
- BLEURT 10% 상승

---

## 아키텍처

### 그래프 구조

**Main Graph**
```
START → normalize_and_route → memory_wrapper → legal_rag_wrapper → persist_turn_memory → END
```

**Legal RAG Subgraph (Retrieval Harness)**
```
START → rewrite_query → retrieve → rerank → judge_sufficiency
                                                    ↓
                                              [sufficient] → answer → END
                                              [insufficient] → fallback → retrieve (loop)
                                              [exhausted] → answer → END
```

**Memory Subgraph**
```
START → load_memories → END
```

### 메모리 분리 전략
- **단기 메모리**: LangGraph checkpointer (SQLite) — thread 단위 대화 상태
- **장기 메모리**: 별도 SQLite memory store — 중요도 + 시간 감쇠 + 토픽 일치도 기반 점수화
- **법률 지식 캐시**: Chroma persistent collections — 도메인별 분리

### 메모리 점수 설계

```text
score = importance × freshness × (0.5 + topic_overlap) × domain_boost × access_boost
```

---

## 디렉토리 구조

```text
KLERK/
├─ src/
│  ├─ app.py                          # Entry point
│  ├─ service.py                      # LegalAgentService, graph lifecycle
│  ├─ router.py                       # Routing + query rewriting + query refinement
│  ├─ answering.py                    # Grounded answer generation (source-type grouping)
│  ├─ mcp_client.py                   # Korean Law MCP gateway
│  ├─ parsers.py                      # ID parsing, chunking, tokenization
│  ├─ data_structure/
│  │  └─ schemas.py                   # Pydantic models (RouteDecision, RetrievedChunk, etc.)
│  ├─ graphs/
│  │  ├─ state.py                     # TypedDict states (AgentState, LegalRAGState, etc.)
│  │  ├─ main_graph.py               # Main LangGraph pipeline
│  │  └─ subgraphs/
│  │     ├─ legal_rag_subgraph.py     # Retrieval harness (rewrite→retrieve→rerank→judge→fallback loop)
│  │     └─ memory_subgraph.py        # Long-term memory search
│  ├─ llm_model/
│  │  └─ llm.py                       # LLM/embedding factory, ainvoke_text/ainvoke_json
│  ├─ storages/
│  │  ├─ vector_store.py              # Chroma wrapper + MCP result caching
│  │  └─ memory_store.py              # SQLite memory repository
│  ├─ utils/
│  │  ├─ config.py                    # Pydantic settings
│  │  ├─ exceptions.py                # Exception hierarchy
│  │  └─ logging_utils.py             # JSON structured logging
│  ├─ vector_db_spec/
│  │  └─ legal_specs.py               # Law specs, alias matching
│  └─ ui/
│     └─ gradio_app.py                # Gradio UI with harness status display
├─ build_vector_db/
│  └─ build_vector_db.py              # PDF → Chroma ingestion (summary+provenance chunking)
├─ tests/
│  ├─ test_router.py
│  ├─ test_memory_scoring.py
│  └─ test_parsers.py
├─ pyproject.toml
├─ .env
└─ README.md
```

---

## 사전 준비

### 1) Python 의존성 설치
```bash
uv sync
```

### 2) Korean Law MCP 서버 준비
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

루트 `.env`에서 다음 값을 설정한다.
```env
LAW_API_OC=...
MCP_SERVER_COMMAND=node
MCP_SERVER_ENTRYPOINT=./external/korean-law-mcp/dist/index.js
```

### 3) 모델 준비

Ollama 사용 시:
```bash
ollama pull qwen3:8b
```

임베딩 (로컬 Hugging Face):
```env
LOCAL_EMBEDDING_PROVIDER=huggingface_local
LOCAL_EMB_MODEL=BAAI/bge-m3
```

OpenAI 호환 endpoint:
```env
LOCAL_LLM_PROVIDER=openai_compatible
LOCAL_LLM_MODEL=gpt-4o-mini
LOCAL_LLM_BASE_URL=https://api.openai.com/v1
LOCAL_LLM_API_KEY=...
```

### 4) 벡터 DB 구축 (육법전서 인덱싱)
```bash
uv run python build_vector_db/build_vector_db.py --pdf-path ./data/yukbeop.pdf --persist-dir .data/chroma
```

---

## 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `APP_ENV` | `dev` | 실행 환경 |
| `LOG_LEVEL` | `INFO` | 로그 레벨 |
| `LOCAL_LLM_PROVIDER` | `ollama` | LLM provider (ollama / openai_compatible) |
| `LOCAL_LLM_MODEL` | `qwen3:8b` | LLM 모델명 |
| `LOCAL_LLM_BASE_URL` | `http://localhost:11434` | LLM 서버 주소 |
| `LOCAL_LLM_API_KEY` | - | OpenAI 호환 API 키 |
| `LOCAL_EMBEDDING_PROVIDER` | `ollama` | Embedding provider |
| `LOCAL_EMB_MODEL` | `BAAI/bge-m3` | Embedding 모델명 |
| `DEFAULT_COLLECTION` | `korean_law` | 기본 검색 collection |
| `USE_TOPIC_COLLECTIONS` | `true` | 주제별 collection 분리 사용 여부 |
| `VECTOR_TOP_K` | `4` | 벡터 검색 상위 문서 수 |
| `SIMILARITY_THRESHOLD` | `0.40` | 충분성 판단 최소 similarity |
| `MIN_RETRIEVED_DOCS` | `2` | 충분성 판단 최소 문서 수 |
| `MAX_RETRIEVAL_ITERATIONS` | `3` | ReAct fallback loop 최대 반복 |
| `RERANK_TOP_K` | `4` | Reranking 후 상위 유지 문서 수 |
| `MCP_FETCH_TOP_N` | `3` | MCP 상세 조회 최대 건수 |
| `MAX_CONTEXT_CHARS` | `5000` | 답변 생성 시 최대 context 길이 |

---

## 실행

```bash
uv run python src/app.py
```

브라우저에서 Gradio UI가 열린다. 좌측 패널에서 retrieval harness의 각 단계별 상태 (routing, query rewrite, iterations, fallback actions, MCP 사용 여부)를 확인할 수 있다.

---

## 검증 방법

1. **단위 테스트**:
```bash
uv run pytest tests/ -v
```

2. **Routing 검증**: 다양한 법률 질문으로 올바른 collection routing 확인
3. **Fallback loop 확인**: LOG_LEVEL=DEBUG 설정 후, 특이 질문으로 fallback 진입 여부 확인
4. **MCP 통합 확인**: LAW_API_OC 설정 후, vector 검색 부족 시 MCP fallback 동작 확인
5. **Structured logging 확인**: JSON 로그에서 `event` 필드별 harness 단계 추적

---

## 남은 리스크 / 향후 확장 포인트

1. **Cross-encoder reranking**: 현재 LLM 기반 reranking을 dedicated cross-encoder 모델로 교체하면 속도와 정확도 개선 가능
2. **Memory store 이관**: SQLite → Postgres + pgvector로 변경하여 memory에도 vector search 적용
3. **판례 collection 확장**: 공개 판결문 데이터를 수집하여 `korean_precedent` collection 구축
4. **주석서/해설본 통합**: 법률 해설 자료를 별도 collection으로 추가하여 판결문 초안 작성 보조까지 확장
5. **MCP 결과 구조화**: MCP search 결과를 structured artifact 기반으로 파싱 개선
6. **HITL 승인 노드**: MCP 호출 전 사용자 승인 패턴 실험
7. **Token-level streaming**: 현재 char-by-char 시뮬레이션을 실제 LLM streaming으로 전환
8. **Evaluation pipeline 자동화**: 법률 QA 벤치마크 기반 자동 평가 구축
