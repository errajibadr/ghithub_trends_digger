# Roadmap

**Pillars:**

- **Foundation**: LangGraph runtime, LangSmith traces, GitOps CI/CD -- all production-ready today.
- **Data and Knowledge**: Elasticsearch (HybridRAG), GraphRAG (Memgraph/TigerGraph), enterprise connectors (ServiceNow, Jira, Confluence sync).
- **Agent Layer**: Where orchestration patterns live -- from single ReAct agents to multi-agent supervisors and deep research agents.
- **Governance**: Evaluation (`sta-eval` CLI), safety guardrails, orchestration-level KPIs.


```
                    ┌─────────────────────────┬─────────────────────────┐
                    │     FOUNDATION          │     DATA & KNOWLEDGE    │
                    │                         │                         │
                    │  [DONE] LangGraph       │  [DONE] ES HybridRAG    │
                    │  [DONE] Middleware (12+)│  [DONE] Postgres Repo   │
                    │  [DONE] GitOps + CI     │  [DONE] Graph Adapters  │
                    │  [DONE] LangSmith       │  [TODO] Memgraph/Light  │
                    │  [DONE] DOC             │  [TODO] Jira Adapter    │
                    │  [TODO] Orch KPIs       │  [TODO] Confluence++    │
                    │  [TODO] Cost Tracking   │  [TODO] ServiceNow++    │
                    │                         │                         │
                    ├─────────────────────────┼─────────────────────────┤
                    │     AGENTS              │     GOVERNANCE          │
                    │                         │                         │
                    │  [DONE] ReAct + Router  │  [DONE] sta-eval CLI   │
                    │  [DONE] AdaptiveRAG     │  [DONE] 3 EvalSuites   │
                    │  [DONE] TwinOps (multi) │  [TODO] RAG Evals      │
                    │  [DONE] Navigator       │  [TODO] Auth/RBAC !!   │
                    │  [DONE] Tech Experts x4 │  [TODO] Guardrails     │
                    │  [TODO] Research Agent   │  [TODO] Audit Logging  │
                    │  [TODO] Jira Agent       │  [TODO] SSO            │
                    │  [TODO] Knowledge Mgr   │  [TODO] KPI Dashboard  │
                    │                         │                         │
                    └─────────────────────────┴─────────────────────────┘
```
## Pillar 2 -- Data and Knowledge


| Objective | Status | Target |
|-----------|--------|--------|
| `AsyncElasticsearchAdapter` + `AuditbeatRepository`, `K8sAuditRepository`, `GenericRepository` | DONE | -- |
| `ElasticRetriever` -- BM25 + kNN, RRF / reranker fusion, `BaseRetriever[T]` protocol | DONE | -- |
| `AsyncPostgresAdapter` + `AsyncPgGenericRepository[T]` -- typed CRUD, Django-style filters | DONE | -- |
| `NetworkXAdapter` (GrandCypher) + `TigerGraphAdapter` (OpenCypher + installed queries) | DONE | -- |
| `UKGRepository` -- topology, impact analysis, communication flows | DONE | -- |
| Confluence content synced into ES HybridRAG index | DONE (basic) | -- |
| ServiceNow incident creation via Twin Router tool | DONE (basic) | -- |
| **Memgraph / LightRAG**: adapter, `LightRAGRetriever` implementing `BaseRetriever[T]`, HTTP consumption server | TODO | Q2 |
| **Jira**: `JiraAdapter` (async REST), `JiraRepository` (CRUD, JQL, transitions), agent tools | TODO | Q2-Q3 |
| Confluence incremental sync (webhook / polling), multi-space indexing | TODO | Q3 |
| ServiceNow expansion: changes, problems, CMDB queries, proper adapter + repository | TODO | Q3 |


## Pillar 3 -- Agent Layer

### -- Single-Agent Runtime Baseline

**Goal**: One reliable runtime that can call tools and maintain state.

**STATUS: DONE** -- This is fully implemented.

```mermaid
graph LR
  START --> ReAct[ReAct Loop]
  ReAct -->|tool call| Tools[Tools]
  Tools -->|observe| ReAct
  ReAct -->|reflect| Think[think_tool]
  Think --> ReAct
  ReAct -->|switch_mode| Gen[Generation]
  Gen --> END_NODE[END]
```

### -- Advanced Multi-Agent Designs

**Goal**: Planner-executor-reviewer, hierarchical flows, adaptive routing.

#### 3.1 Planner-Executor-Reviewer Loop

```mermaid
graph TD
  Planner[Planner] -->|subtasks| Exec[Executor Pool]
  Exec -->|results| Review[Reviewer]
  Review -->|approved| Output[Final Output]
  Review -->|revise| Planner
```

- **Formal Planner node**: A planning step that decomposes complex tasks into subtasks with a plan schema. The `RagPlanningNode` mentioned in the creative phase doc is not yet implemented.
- **Executor pool**: Parallel execution of planned subtasks (builds on the fan-out pattern from Phase 2).
- **Reviewer with feedback loop**: A dedicated reviewer that can reject and send back to the planner with structured feedback.

Key interraogation : **DeepAgents vs this pattern**

### Adatprive RAG architecture

 Adaptive RAG does well:

```mermaid
graph LR
  Q[Query] --> Parse --> Retrieve --> Merge --> Grade{Relevant?}
  Grade -->|yes| Generate --> Halluc{Hallucinated?}
  Grade -->|no| Rewrite --> Retrieve
  Halluc -->|no| QualityCheck{Complete?}
  Halluc -->|yes| Generate
  QualityCheck -->|yes| Output
  QualityCheck -->|no| Rewrite
```

- Self-correcting retrieval (query rewriting on failure)
- Hallucination detection
- Answer quality assessment
- Multi-query support with merging
- Configurable precision modes


### Research Agent 

```mermaid
graph TD
  subgraph research_agent [Research Agent - Orchestrator Level]
    Clarify[Clarify / Scope] --> Plan[Research Planner]
    Plan -->|"research_brief + subtopics"| Supervisor[Research Supervisor]
  end

  subgraph retrieval [Retrieval Layer - Reusable]
    Supervisor -->|subtopic| RAG1[AdaptiveRagGraph - Elasticsearch]
    Supervisor -->|subtopic| RAG2[AdaptiveRagGraph - Confluence]
    Supervisor -->|subtopic| GR[GraphRAG Query]
    Supervisor -->|subtopic| Web[Web Search]
  end

  subgraph synthesis [Synthesis Layer]
    RAG1 --> Merge[Evidence Merger]
    RAG2 --> Merge
    GR --> Merge
    Web --> Merge
    Merge --> Synth[Synthesizer - Draft Report]
  end

  subgraph review [Review Layer]
    Synth --> Review{Reviewer}
    Review -->|"gaps found"| Supervisor
    Review -->|"complete"| Format[Report Formatter]
    Format --> Output[Final Output]
  end
```

**Key design decisions:**

1. **AdaptiveRagGraph as a retriever tool** (not the orchestrator): The existing RAG graph becomes one of N retrieval strategies that the research supervisor dispatches to. It already has the right interface (`ainvoke` with messages -> results).

2. **Research Planner** (new): Decomposes the user's research question into a research brief with subtopics. This is the `standalone_mode` / `RagPlanningNode` mentioned in the [creative phase doc](memory_bank/creative_phase_adaptive_rag_refactor.md) but not yet built.

3. **Research Supervisor** (new): Iteratively dispatches subtopics to retrieval sources, collects evidence, and decides when enough evidence is gathered. This could be built using the DeepAgent pattern already used by Twin Ops.

4. **Synthesizer** (new): Takes all evidence and produces a structured report. This is a generation-focused agent with no tools, similar to the "generation mode" in the existing mode system.

5. **Reviewer** (new): Validates completeness and quality of the report. Can send back to the supervisor for more research. The existing `RagAnswerQualityNode` and `RagHallucinationCheckNode` can be reused as part of this.




## Pillar 4 -- Security, Evaluation and Governance

### 4a. Security (Critical Gap)

Current state: `auth.py` returns hardcoded `"user-123"`. Zero real authentication.

```mermaid
graph LR
  S0["Today: stub"] --> S1["Step 1: API Key Auth"]
  S1 --> S2["Step 2: RBAC"]
  S2 --> S3["Step 3: Enterprise SSO + Audit"]
```

| Objective | Status | Target |
|-----------|--------|--------|
| Real API key validation, reject invalid keys with 401 | TODO | Q2 |
| RBAC: roles (admin, dev, viewer), per-graph and per-tool permissions | TODO | Q2-Q3 |
| Enterprise SSO (OIDC/SAML), per-user token scoping (Jira/ServiceNow tokens) | TODO | Q3 |
| Structured audit logging (who, which agent, when, what context) | TODO | Q3 |

### 4b. Evaluation

| Objective | Status | Target |
|-----------|--------|--------|
| `sta-eval` CLI with `EvalSuite` pattern (dataset + evaluators + flavors) | DONE | -- |
| Agent eval suites: `TwinRouterEvalSuite`, `TemplateEvalSuite`, `RagEvalSuite` | DONE | -- |
| Confusion matrix support, LangSmith provider abstraction | DONE | -- |
| **RAG Retrieval Precision@K / Recall@K** -- independent retrieval quality | TODO | Q2 |
| **Context Relevance** -- are retrieved chunks useful for the question? | TODO | Q2 |
| **Faithfulness** -- is the answer grounded in retrieved context? | TODO | Q2 |
| **Pipeline step latency** -- time per node in retrieval vs generation | TODO | Q3 |
| Ragas / DeepEval integration or custom evaluators in `sta-eval` | TODO | Q3 |

### 4c. Guardrails and Observability

| Objective | Status | Target |
|-----------|--------|--------|
| `GenerationRetryMiddleware` with fallback models | DONE | -- |
| `ToolCallLimitMiddleware`, `SummarizationMiddleware` | DONE | -- |
| Input/output guardrails: PII detection, content filtering | TODO | Q3 |
| Circuit breakers: auto-shutdown after repeated failures | TODO | Q3 |
| Orchestration KPI dashboard (LangSmith or Grafana) | TODO | Q3-Q4 |
| Config-driven orchestration (YAML graph definitions) | TODO | Q4 |
