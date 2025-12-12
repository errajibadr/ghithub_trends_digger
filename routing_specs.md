# Router 4 Twin - Technical Specification

**Version:** 1.0  
**Date:** December 10, 2025  
**Status:** Draft for Implementation  


---

## 1. Scope & Feature Roadmap

### 1.1 Version Scope

| Feature | v1.0 | v1.1 | Later |
|---------|:----:|:----:|:-----:|
| **Semantic Router (BGE-m3)** | ✅ | | |
| **LLM Router fallback** | ✅ | | |
| **Previous route weighting** | ✅ | | |
| **Clarification flow** | ✅ | | |
| **FastMode middleware (user toggle)** | ✅ | | |
| **RAG tool wrapper** | ✅ | | |
| **Incident Agent tool wrapper** | ✅ | | |
| **General Knowledge agent** | ✅ | | |
| **Evaluation framework** | ✅ | | |
| **Base dataset creation** | ✅ | | |
| **Skeleton architecture with mock calls** | ✅ | | |
| Semantic Caching | | | ✅ |
| GK internal routing (Code ↔ TextGen) | | ✅ | |
| Fine-tuned BERT/SetFit router | | | ✅ |
| FastMode intent detection (auto) | | ✅ | |
| Elastic Retriever RAG | | | ✅ |
| Rust Candle integration | | | ✅ |

### 1.2 Implementation Priority (v1.0)

| Priority | Task | Dependencies |
|:--------:|------|--------------|
| **P0** | Create skeleton architecture with mock calls | None |
| **P1** | Implement Semantic Router with BGE-m3 | Skeleton |
| **P1** | Implement routing middlewares | Skeleton |
| **P2** | Create tool wrappers (RAG, Incident Agent) | Skeleton |
| **P2** | Build General Knowledge agent | Skeleton |
| **P3** | Implement LLM Router fallback | Semantic Router |
| **P3** | Implement Clarification flow | LLM Router |
| **P4** | Build evaluation framework | All routes working |
| **P4** | Create & label evaluation dataset | Framework |

---

## 2. Architecture Overview

### 2.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TWIN ROUTER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────┐     ┌──────────────────────────────────────┐                 │
│   │  User   │     │     LAYER 1: SEMANTIC ROUTER         │                 │
│   │  Query  │────▶│     (BGE-m3, <150ms)                 │                 │
│   └─────────┘     │                                      │                 │
│        │          │  Inputs:                             │                 │
│        │          │   - Query embedding                  │                 │
│        │          │   - Previous route (weighted)        │                 │
│        │          │                                      │                 │
│        │          │  Output: Softmax scores per route    │                 │
│        │          └──────────────┬───────────────────────┘                 │
│        │                         │                                          │
│        │          ┌──────────────┴───────────────┐                         │
│        │          │                              │                          │
│        │     HIGH CONFIDENCE              LOW CONFIDENCE                    │
│        │     (threshold TBD)              (threshold TBD)                   │
│        │          │                              │                          │
│        │          │                              ▼                          │
│        │          │              ┌───────────────────────────┐             │
│        │          │              │  LAYER 2: LLM ROUTER      │             │
│        │          │              │  (Fallback,        )       │             │
│        │          │              │                           │             │
│        │          │              │  Decisions:               │             │
│        │          │              │   - Clear → Route         │             │
│        │          │              │   - Ambiguous → Clarify   │             │
│        │          │              └─────────────┬─────────────┘             │
│        │          │                            │                            │
│        │          ▼                            ▼                            │
│        │    ┌─────────────────────────────────────────────────┐            │
│        │    │              AGENT CLUSTER                       │            │
│        │    │  ┌─────────┐ ┌─────────┐ ┌─────┐ ┌─────────┐   │            │
│        │    │  │Incident │ │   RAG   │ │ GK  │ │ Clarify │   │            │
│        │    │  │ Agent   │ │         │ │     │ │         │   │            │
│        │    │  └─────────┘ └─────────┘ └─────┘ └─────────┘   │            │
│        │    └─────────────────────────────────────────────────┘            │
│        │                           │                                        │
│        │                           ▼                                        │
│        │                    ┌─────────────┐                                │
│        └───────────────────▶│   Response  │                                │
│          (next turn)        └─────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Base Architecture** | ReAct Agent + Middlewares | Leverage existing middleware library; flexibility with controlled determinism |
| **Routing Strategy** | Hybrid (Semantic → LLM fallback) | Balance speed (<150ms for clear routes) with accuracy (LLM for ambiguous) |
| **Route Continuity** | Re-route every query with previous route weighting | Maintains context without hard-locking; natural conversation flow |
| **Clarification Handling** | Full workflow re-entry | User response goes through complete routing logic |
| **Error Fallback** | Default to RAG | Grounded answers safer than ungrounded GK |
| **Streaming** | Native LangGraph capabilities | No custom implementation needed |

---

## 3. Routes Specification

### 3.1 Route Definitions

| Route ID | Name | Handler | Status | Description |
|----------|------|---------|--------|-------------|
| `INCIDENT` | Incident Agent | Existing LangGraph Graph | ✅ Exists (needs wrapper) | ServiceNow ticket queries, incident status, priorities |
| `RAG` | RAG Q&A | Existing RAG API | ✅ Exists (needs wrapper) | Internal documentation, procedures, application info |
| `GK` | General Knowledge | New Agent | 🔨 To build | General questions, code help, translation, text generation |
| `CLARIFY` | Clarification | New Flow | 🔨 To build | Ambiguous query resolution |

### 3.2 Route Characteristics

```python
ROUTE_CONFIG = {
    "INCIDENT": {
        "description": "ServiceNow incident and ticket management",
        "handler_type": "langgraph_subgraph",
        "api_endpoint": None,  # Direct graph invocation
        "typical_latency": >2s,
        "fallback_on_error": "RAG",
    },
    "RAG": {
        "description": "Internal documentation retrieval and Q&A",
        "handler_type": None, # Particular case - handof + force end Graph 
        "typical_latency": TBD, # Unknow, for this V.0 special architecture, we don't care about this latency.
        "fallback_on_error": "GK",  # Last resort: answer without grounding
    },
    "GK": {
        "description": "General knowledge, coding, translation, text generation",
        "handler_type": "agent/model",
        "api_endpoint": None,  # Local agent
        "typical_latency": >2s,
        "fallback_on_error": None,  # Terminal
    },
    "CLARIFY": {
        "description": "Request clarification for ambiguous queries",
        "handler_type": "clarification_flow",
        "api_endpoint": None,
        "typical_latency": >2s,
        "fallback_on_error": "RAG",
    },
}
```

### 3.3 Route Bias Rules

| Scenario | Bias | Rationale |
|----------|------|-----------|
| RAG vs GK (close scores) | **RAG** | Prefer grounded answers over hallucination risk |
| RAG vs INCIDENT (close scores) | **CLARIFY** | User intent matters; ticket vs docs is significant |
| INCIDENT vs GK (close scores) | **INCIDENT** | Specific agent better than general |
| Any three-way tie | **CLARIFY** | Too ambiguous to guess |

---

## 4. Semantic Router Specification

### 4.1 Overview

The semantic router is the first-layer classification system providing fast (ms order of magnitude) routing decisions based on query embeddings.

### 4.2 Technical Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| **Embedding Model** | BGE-m3 | Multilingual, good for French/English mix |
| **Similarity Metric** | Cosine similarity | Standard for embeddings |
| **Score Normalization** | Softmax | Converts to probability-like scores (sum = 1) |
| **Library** | `semantic-router` or custom | TBD based on testing |

### 4.3 Route Exemplars Strategy

**Initial Creation (per route):**
- Collect 10-50 real queries from production logs
- Ensure queries are clearly in-class (not borderline)
- Cover variety of phrasings and intents
- Add short route descriptions as extra exemplars
- Normalize (remove PII, avoid very long context)

**Optimization Loop:**
1. Evaluate on labeled validation set
2. Inspect systematic confusions and high-entropy samples
3. Fix by:
   - Adding exemplars for under-represented phrasings
   - Removing overly generic exemplars
   - Adjusting per-route thresholds

### 4.4 Scoring Logic

```python
def compute_route_scores(query: str, previous_route: Optional[str] = None) -> dict[str, float]:
    """
    Compute routing scores using semantic similarity + softmax normalization.
    
    Returns:
        Dict mapping route_id -> probability score (sum = 1.0)
    """
    # Step 1: Embed query
    query_embedding = embed(query)  # BGE-m3
    
    # Step 2: Compute similarity to each route's exemplars
    raw_scores = {}
    for route_id, exemplars in ROUTE_EXEMPLARS.items():
        exemplar_embeddings = [embed(ex) for ex in exemplars]  # Can be pre-computed
        similarities = [cosine_similarity(query_embedding, ex_emb) for ex_emb in exemplar_embeddings]
        raw_scores[route_id] = max(similarities)  # Or mean, TBD
    
    # Step 3: Apply softmax normalization
    softmax_scores = softmax(raw_scores)
    
    # Step 4: Apply previous route weighting (if applicable)
    if previous_route and previous_route in softmax_scores:
        softmax_scores = apply_previous_route_weight(softmax_scores, previous_route)
    
    return softmax_scores
```

### 4.5 Previous Route Weighting Strategies

two approaches documented; **Strategy A** is the initial implementation choice.

#### Strategy A: Lower Threshold for Previous Route (Initial Choice)

```python
def should_route_directly(scores: dict, previous_route: Optional[str]) -> tuple[bool, str]:
    """
    Lower the confidence threshold required for the previous route.
    """
    best_route = max(scores, key=scores.get)
    best_score = scores[best_route]
    second_best_score = sorted(scores.values(), reverse=True)[1]
    margin = best_score - second_best_score
    
    # Standard thresholds
    CONFIDENCE_THRESHOLD = 0.85  # TBD
    MARGIN_THRESHOLD = 0.15      # TBD
    
    # Lowered thresholds for previous route
    PREV_ROUTE_CONFIDENCE_THRESHOLD = 0.70  # TBD
    PREV_ROUTE_MARGIN_THRESHOLD = 0.10      # TBD
    
    if best_route == previous_route:
        # More lenient for continuing same route
        if best_score >= PREV_ROUTE_CONFIDENCE_THRESHOLD and margin >= PREV_ROUTE_MARGIN_THRESHOLD:
            return True, best_route
    else:
        # Standard thresholds for route change
        if best_score >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD:
            return True, best_route
    
    return False, None  # Fall back to LLM router
```

#### Strategy B: Weight Previous Route in Score Calculation (Alternative)

```python
def apply_previous_route_weight(scores: dict, previous_route: str, weight: float = 0.1) -> dict:
    """
    Boost the previous route's score before threshold comparison.
    """
    adjusted_scores = scores.copy()
    if previous_route in adjusted_scores:
        adjusted_scores[previous_route] += weight
    
    # Re-normalize to sum = 1
    total = sum(adjusted_scores.values())
    return {k: v / total for k, v in adjusted_scores.items()}
```


### 4.6 Confidence Thresholds

> ⚠️ **All thresholds are TBD** - require empirical testing with real data

| Threshold | Initial Value | Purpose |
|-----------|---------------|---------|
| `HIGH_CONFIDENCE` | 0.85 | Route directly without LLM fallback |
| `MARGIN_THRESHOLD` | 0.15 | Minimum gap to second-best route |
| `PREV_ROUTE_CONFIDENCE` | 0.70 | Lower threshold when continuing same route |
| `PREV_ROUTE_MARGIN` | 0.10 | Lower margin when continuing same route |
| `RAG_GK_BIAS_THRESHOLD` | 0.05 | If RAG and GK within this margin, choose RAG |

### 4.7 Multi-Route Limitation

**Current Limitation:** Semantic router produces single-route classification. Multi-intent queries (e.g., "Create a ticket and find the documentation") are not decomposed.

**Behavior:**
- Softmax scores will show two high values
- This triggers LLM Router fallback (low margin)
- LLM Router may clarify or pick primary intent

**Future Enhancement (v1.1+):** Query decomposition for explicit multi-route requests.

---

## 5. LLM Router Specification (Fallback)

### 5.1 Overview

The LLM router is invoked when the semantic router has low confidence. It provides more nuanced classification and can trigger clarification.

### 5.2 Trigger Conditions

LLM Router is called when ANY of:
- Best route confidence < `HIGH_CONFIDENCE` threshold
- Margin to second-best < `MARGIN_THRESHOLD`
- Explicit bias rules require clarification (e.g., RAG vs INCIDENT tie)

### 5.3 Prompt Template

> **Note:** Routes are presented as **tools** to leverage LLM native tool-calling capabilities.

```python
LLM_ROUTER_PROMPT = """You are a routing assistant for an enterprise IT operations chatbot.

## AVAILABLE TOOLS

**incident_agent** - ServiceNow ticket and incident management
Call this for: incident lookups (INC numbers), ticket status, priorities, creating/updating tickets
Examples: "What's the status of INC10557452?", "Quelle est la priorité de cet incident?"

**rag_search** - Internal documentation retrieval
Call this for: company procedures, application info, configs, team/role information
Examples: "Quelle est la procédure pour déclarer un incident?", "Qui est référent monitoring?"

**general_knowledge** - Direct LLM response (NO internal context needed)
Call this for: code help, translation, text generation, general explanations
Examples: "Translate to English...", "Fix this Python code...", "Explain Kubernetes"

**clarify_user** - Request clarification
Only call when you cannot reasonably determine the appropriate tool
Prefer making a decision over clarifying

## BIAS RULES
- If torn between rag_search and general_knowledge → Choose rag_search (prefer grounded answers)
- If torn between incident_agent and rag_search → Choose clarify_user (significant difference)
- If torn between incident_agent and general_knowledge → Choose incident_agent

## CURRENT CONTEXT
- Previous tool: {previous_route}
- Semantic router scores: {semantic_scores}

## USER QUERY
{query}

## YOUR TASK
Select the most appropriate tool for the user's query.
Call EXACTLY ONE tool based on your analysis.
"""
```

### 5.4 LLM Router Output Schema

> **Note:** With tools semantic, the LLM Router uses native tool calling. 
> The router agent is a "react" agent configured with tools that trigger the appropriate route and calls them based on the query.

```python

@tool
def rag_search(query: str) -> Command:
    pass

@tool
def incident_agent(query: str) -> Command:
    pass

...

agent = create_agent(
    tools=[rag_search, incident_agent, general_knowledge, clarify_user],
    prompt=LLM_ROUTER_PROMPT,
)

result = await agent.ainvoke(
    {
        "messages": [HumanMessage("What's the status of INC10557452?")],
    }
)

print(result)


```

---

## 6. State Schema

### 6.1 Router State Definition

> **Design Principle:** Start minimal. Add fields only when needed.

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

class TwinRouterState(TypedDict):
    """Minimal router state for v1.0."""
    
    # Core conversation
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Routing
    current_route: Literal["INCIDENT", "RAG", "GK", "CLARIFY"] | None
    previous_route: Literal["INCIDENT", "RAG", "GK", "CLARIFY"] | None
    semantic_scores: dict[str, float]  # Route -> softmax score
    
    # Mode control
    fast_mode: bool  # User toggle: True = single tool call, skip reasoning loops
```

**Removed fields** (can be added later if needed):
- `route_confidence`, `routing_method` - implicit from semantic_scores
- `route_history` - previous_route sufficient for v1.0
- `iteration_count`, `max_iterations` - handled by FastModeMiddleware
- `handler_input`, `handler_output` - tools handle their own I/O
- `error_count`, `last_error`, `used_fallback` - deferred to v1.1 ErrorFallbackMiddleware

### 6.2 State Initialization

```python
def create_initial_state() -> TwinRouterState:
    return {
        "messages": [],
        "current_route": None,
        "previous_route": None,
        "semantic_scores": {},
        "fast_mode": True,  # Default to fast mode
    }
```

---

## 7. Middleware Specifications

### 7.1 Middleware Overview

| Middleware | Type | Hook | Purpose | v1.0 |
|------------|------|------|---------|:----:|
| `SemanticRoutingMiddleware` | Pre-Agent | `before_agent` | Fast first-layer routing (runs ONCE) | ✅ |
| `FastModeMiddleware` | Post-Tool | `after_tool` | Force graph END after tool call | ✅ |

**Future Middleware (v1.1+):**
- `ToolMaxUseMiddleware` - Circuit breaker for loop search in RAG tool calls

### 7.2 SemanticRoutingMiddleware

**Hook:** `before_agent` - runs **ONCE** per user request, not per ReAct loop iteration.

**Behavior:**
1. Extract latest user message
2. Compute semantic router scores
3. Apply previous route weighting (Strategy A)
4. If high confidence: set route, signal skip LLM router
5. If low confidence: store scores for LLM router

```python
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime


class SemanticRoutingMiddleware(AgentMiddleware):
    """First-layer semantic routing - runs ONCE at agent start.
    
    Uses before_agent hook to compute semantic scores before the ReAct loop begins.
    If confident, sets route directly; otherwise stores scores for LLM router.
    """
    
    def __init__(self, embedding_model: str = "bge-m3"):
        super().__init__()
        self.router = SemanticRouter(embedding_model)
        self.thresholds = RoutingThresholds()
    
    def before_agent(self, state: dict, runtime: Runtime) -> dict | None:
        """Compute semantic routing scores once at agent start.
        
        This hook runs ONCE per user request, before the ReAct loop begins.
        """
        messages = state.get("messages", [])
        if not messages:
            return None
            
        # Get latest user message
        query = messages[-1].content if hasattr(messages[-1], "content") else ""
        previous_route = state.get("previous_route")
        
        # Compute semantic scores
        scores = self.router.compute_scores(query)
        should_route, route = self._evaluate_confidence(scores, previous_route)
        
        if should_route:
            # High confidence - set route directly, skip LLM router
            return {
                "current_route": route,
                "semantic_scores": scores,
                "_skip_llm_router": True,  # Signal to skip LLM classification
            }
        
        # Low confidence - store scores for LLM router to use
        return {"semantic_scores": scores}
    
    def _evaluate_confidence(
        self, scores: dict[str, float], previous_route: str | None
    ) -> tuple[bool, str | None]:
        """Apply threshold logic with previous route consideration."""
        # Implementation per Section 4.5 Strategy A
        ...
```

### 7.3 FastModeMiddleware

**Hook:** `before_model` - checks last message type and jumps to end when appropriate.

**Behavior:**
1. If last message is AIMessage → jump to end
2. If fast_mode enabled AND last message is ToolMessage:
   - `rag_search`: jump to end (handoff to external RAG system)
   - `incident_agent`: append AIMessage with tool content, jump to end
   - `general_knowledge`: append AIMessage with tool content, jump to end
   - `clarify_user`: append AIMessage with clarification, jump to end

```python
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime


class FastModeMiddleware(AgentMiddleware):
    """Force graph termination after single tool call in fast mode.
    
    Uses before_model hook to check last message and jump to end.
    """
    
    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Check last message and decide if we should jump to end."""
        messages = state.get("messages", [])
        if not messages:
            return None

        last_message = messages[-1]

        # If last message is AIMessage, go to end
        if isinstance(last_message, AIMessage):
            return {"jump_to": "end"}

        # Handle ToolMessage in fast mode
        if state.get("fast_mode", False) and isinstance(last_message, ToolMessage):
            tool_name = last_message.name

            if tool_name == "rag_search":
                # RAG: handoff to external system
                return {"jump_to": "end"}

            if tool_name in ("incident_agent", "general_knowledge", "clarify_user"):
                # Append AI message with tool content, then end
                return {
                    "jump_to": "end",
                    "messages": [AIMessage(content=last_message.content)],
                }

        return None
```



---

## 8. Route Tools Specification

> **Design Principle:** Routes are implemented as LangGraph `@tool` functions that return `Command` objects.
> This leverages native LangGraph patterns and enables clean graph control flow.

### 8.1 Tool Interface

All route tools follow the same pattern:
- Accept query/parameters as input
- Use `ToolRuntime` for state access when needed
- Return simple dict for state updates or `Command(goto="__end__", update={...})` to terminate graph (currently only for RAG tool call).

For the rest the goto __end__ is forced by fast mode middleware.

```python
from langchain_core.tools import tool
from langchain.tools.tool_node import ToolRuntime
from langgraph.types import Command
```

### 8.2 RAG Tool (Special Case - Handoff to External System)

> **Special Case:** RAG is handled by an external system. This tool signals handoff
> by ending the graph with the query stored in state.

```python
@tool
def rag_search(query: str) -> Command:
    """Search internal documentation for company procedures and information.
    
    Call this for: company procedures, application info, configs, team/role information.
    
    Args:
        query: The search query for internal documentation
    
    Returns:
        Command to end graph (RAG handled externally)
    """
    # RAG is handled by external system - we just signal handoff
    return Command(
        goto="__end__",
        update={"current_route": "RAG", "rag_query": query}
    )
```

### 8.3 Incident Agent Tool (Invoke Subgraph)

```python
@tool
async def incident_agent(query: str, runtime: ToolRuntime) -> Command:
    """Query ServiceNow for incident information.
    
    Call this for: incident lookups (INC numbers), ticket status, priorities,
    creating or updating tickets.
    
    Args:
        query: The incident-related query
        runtime: Injected runtime for accessing state and incident graph
    """
    from somewhere import get_incident_graph  # Factory function
    
    # Invoke existing incident agent subgraph
    incident_graph = get_incident_graph()
    result = await incident_graph.ainvoke({
        "messages": runtime.state.get("messages", [])
    })
    
    return {
        "current_route": "INCIDENT",
        "messages": result.get("messages", []),
    }
    
```

### 8.4 Clarify Tool (ChatModel with Clarification Prompt)

```python
@tool
async def clarify_user(
    clarification_type: str,
    question: str,
    runtime: ToolRuntime,
) -> Command:
    """Request clarification from the user when intent is ambiguous.
    
    Only call when you cannot reasonably determine the appropriate tool.
    Prefer making a decision over clarifying.
    
    Args:
        clarification_type: Type of ambiguity (e.g., "RAG_vs_GK", "RAG_vs_INCIDENT", "generic")
        question: The clarification question to ask the user
        runtime: Injected runtime for state access
    """
    from langchain_core.messages import AIMessage
    
    clarification_response = AIMessage(content=question)
    
    return Command(
        goto="__end__",
        update={
            "current_route": "CLARIFY",
            "messages": [clarification_response],
            "awaiting_clarification": True,
        }
    )
```

### 8.5 General Knowledge Tool (Simple Agent with Prompt)

```python
@tool
async def general_knowledge(query: str, runtime: ToolRuntime) -> Command:
    """Answer general questions without internal context.
    
    Call this for: code help, translation, text generation, general explanations.
    Do NOT use for company-specific questions.
    
    Args:
        query: The general knowledge query
        runtime: Injected runtime for state access
    """
    from sta_agent_engine.models.custom_chat_model import create_chat_model
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    
    GK_PROMPT = """You are a helpful AI assistant.

You can help with:
- Programming and code questions
- Translation between languages
- Text generation and reformulation
- General explanations and knowledge

Respond in the same language as the user's query.
Be concise and helpful."""
    
    model = create_chat_model()
    response = await model.ainvoke([
        SystemMessage(content=GK_PROMPT),
        HumanMessage(content=query),
    ])
    
    return Command(
        goto="__end__",
        update={
            "current_route": "GK",
            "messages": [AIMessage(content=response.content)],
        }
    )
```

### 8.6 Tools Summary

| Tool | Purpose | Returns |
|------|---------|---------|
| `rag_search` | Handoff to external RAG system | `Command(goto="__end__")` with query |
| `incident_agent` | Invoke incident subgraph | state update with messages |
| `clarify_user` | Ask user for clarification | `Command(goto="__end__")` with question |
| `general_knowledge` | Direct LLM response | `Command(goto="__end__")` with response |

---

## 9. Error Handling

### 9.1 Error Fallback Chain

```
Primary Route (INCIDENT/RAG/GK)
    │
    ├── Success → Return response
    │
    └── Error → Fallback to RAG
                    │
                    ├── Success → Return response (note: used fallback)
                    │
                    └── Error → Return error message to user
```

### 9.2 Error Response Template

```python
ERROR_RESPONSE = """I apologize, but I encountered an issue processing your request.

**What happened:** {error_summary}

**What you can do:**
- Try rephrasing your question
- If asking about an incident, verify the incident number
- Contact support if the issue persists

I've logged this error for our team to investigate."""
```

### 9.3 Service-Specific Error Handling

| Service | Error Type | Behavior |
|---------|------------|----------|
| Semantic Router | Embedding service down | Skip to LLM Router |
| LLM Router | LLM timeout/error | Use highest semantic score |
| RAG API | API unavailable | Fallback to GK with disclaimer |
| Incident Agent | SNOW unavailable | Inform user, suggest retry |
| GK | LLM error | Return error template |

---

## 10. Evaluation Framework

### 10.1 Evaluation Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Routing Accuracy** | >90% | Correct route on clear queries |
| **Semantic Router Speed** | <150ms | P95 latency |
| **End-to-End Latency (fast)** | <2s | User query to first token |
| **Clarification Rate** | <15% | Queries requiring clarification |
| **Fallback Rate** | <5% | Queries using error fallback |
| **Route Continuation Accuracy** | >95% | Correct continuation decisions |

### 10.2 Evaluation Dataset Schema

```python
from pydantic import BaseModel
from typing import Literal, Optional

class EvaluationSample(BaseModel):
    # Query
    query: str
    language: Literal["fr", "en", "other"]
    
    # Labels
    primary_route: Literal["INCIDENT", "RAG", "GK"]
    secondary_route: Optional[Literal["INCIDENT", "RAG", "GK"]] = None
    
    # Metadata
    confidence: Literal["clear", "ambiguous", "multi_intent"]
    needs_clarification: bool
    
    # Context (optional)
    previous_route: Optional[str] = None
    conversation_context: Optional[str] = None
    
    # Notes
    notes: Optional[str] = None
    source: Literal["production", "synthetic", "manual"]
```

### 10.3 Dataset Categories (from Production Data)

| Category | Route | Example Count (Target) |
|----------|-------|------------------------|
| Incident queries | INCIDENT | 50+ |
| Procedure questions | RAG | 100+ |
| Application info | RAG | 100+ |
| Configuration guides | RAG | 50+ |
| Code questions | GK | 50+ |
| Translation | GK | 30+ |
| Text generation | GK | 30+ |
| Ambiguous | CLARIFY | 30+ |

---

## 11. Implementation Checklist

### 11.1 Phase 0: Skeleton Architecture

> **FIRST PRIORITY:** Create the complete skeleton with mock calls before implementing real logic.

- [ ] **Create project structure**
  ```
  router_4_twin/
  ├── __init__.py
  ├── graph.py              # Main LangGraph definition
  ├── state.py              # State schema
  ├── routes/
  │   ├── __init__.py
  │   ├── semantic_router.py   # Mock: returns random scores
  │   └── llm_router.py        # Mock: returns random decision
  ├── tools/
  │   ├── __init__.py
  │   ├── incident.py          # Mock: returns dict state update with messages
  │   ├── rag.py               # Mock: returns Command to END
  │   ├── gk.py                # Mock: returns Command to END
  │   └── clarify.py           # Mock: returns Command to END
  ├── middlewares/
  │   ├── __init__.py
  │   ├── semantic_routing.py  # before_agent hook
  │   └── fast_mode.py         # after_tool hook
  ├── config.py             # Thresholds, settings
  └── tests/
      └── test_skeleton.py  # Verify flow works end-to-end
  ```

- [ ] **Implement mock semantic router**
  ```python
  # Mock returns configurable scores for testing
  def mock_semantic_scores(query: str) -> dict[str, float]:
      return {"INCIDENT": 0.2, "RAG": 0.5, "GK": 0.3}
  ```

- [ ] **Implement mock tools**
  ```python
  # Each tool returns a mock Command
  @tool
  def mock_rag_search(query: str) -> Command:
      return Command(
          goto="__end__",
          update={"current_route": "RAG", "mock_response": f"[MOCK RAG] {query}"}
      )
  ```

- [ ] **Wire up LangGraph flow**
  - Router node → Handler nodes → Response node
  - Verify state flows correctly through all paths

- [ ] **Test all routes manually**
  - Force each route and verify correct tool is called
  - Verify Command(goto="__end__") terminates graph
  - Test clarification flow

### 11.2 Phase 1: Semantic Router Implementation

- [ ] Set up BGE-m3 embedding model
- [ ] Create initial route exemplars (10-20 per route)
- [ ] Implement softmax normalization
- [ ] Implement previous route weighting (Strategy A)
- [ ] Integrate with SemanticRoutingMiddleware
- [ ] Benchmark latency (<150ms target)

### 11.3 Phase 2: Route Tools Integration

- [ ] Create `rag_search` tool (Command to END with query)
- [ ] Create `incident_agent` tool (invoke subgraph)
- [ ] Implement `general_knowledge` tool (ChatModel with prompt)
- [ ] Implement `clarify_user` tool (return clarification question)
- [ ] Test each tool independently
- [ ] Test tool routing end-to-end

### 11.4 Phase 3: LLM Router & Refinement

- [ ] Implement LLM Router prompt
- [ ] Integrate as fallback from semantic router
- [ ] Implement clarification flow (response re-enters workflow)
- [ ] Test edge cases and ambiguous queries

### 11.5 Phase 4: Middlewares & Controls

- [ ] Implement SemanticRoutingMiddleware (`before_agent` hook)
- [ ] Implement FastModeMiddleware (`after_tool` hook)
- [ ] Test user toggle for fast mode
- [ ] Verify single tool call in fast mode

**Future (v1.1):**
- [ ] ToolMaxUseMiddleware (circuit breaker for RAG tool calls)

### 11.6 Phase 5: Evaluation

- [ ] Create evaluation dataset (500+ samples)
- [ ] Implement evaluation harness
- [ ] Run baseline evaluation
- [ ] Tune thresholds based on results
- [ ] Optimize route exemplars
- [ ] Document final performance metrics

---

## 12. Appendix: Production Dataset Examples

### 12.1 INCIDENT Route Examples

```
- What do you know about the incident INC10557452?
- Quelle la priorité de cet incident?
- Résume l'incident 'INC12291856'?
- Quel est le statut de l'incident?
```

### 12.2 RAG Route Examples

**Procedures:**
```
- Quelle est la procédure pour déclarer un incident?
- Quelle est la procédure de mise en surveillance d'une application?
- Comment configurer un ecosystème Vault?
- Quelles sont les procédures de bascule de l'application?
```

**Application Info:**
```
- Qui est référent monitoring pour l'APS BCEF?
- Qui est adoption leader Dynatrace au sein de l'APS?
- Quel est le groupe support de GMON?
- Quels sont les utilisateurs de l'application GPC?
- Quelles sont les horaires de services de l'application GPC?
- A quoi sert l'application GPC?
```

**Architecture:**
```
- Sur quel type d'architecture technique s'appuie l'application?
- Y-a-t-il des flux vers une application tierce?
- L'application utilise-t-elle le Mainframe?
```

### 12.3 GK Route Examples

**Translation:**
```
- traduire para ingles tudo o que escrever
- traduire en anglais: -[Fournisseurs consultés
- Support jährliches RCSA-Update in englisch
```

**Text Generation:**
```
- corrige "Note il faut pas oublier que avant de commencer..."
- Enhance this message: I hope to send you the updated version by EOD Monday
- reformuler La liste des erreurs a été fournie dans la ASAP
```

**Code:**
```
- je prefere pandas, mais es tu sur de df.at[2,'C']?
- Ecris ce calcul sous forme d'une fonction
- enlever les doublons d'une liste string dans angular
- df_pm_tuv_detail.iloc[: -1] est-ce sur python ceci prend bien la dernière colonne?
```

### 12.4 Ambiguous Examples (Potential CLARIFY)

```
- c'est quoi les différences entre OPS et APS?  
  → Could be RAG (internal definition) or GK (general concept)


- it may not be optimized but it's not wrong to not show level 1 and level 2?
  → Missing context, unclear domain
```

---

## 13. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Dec 10, 2025 | AI-OPS Team | Initial specification |

---

**END OF SPECIFICATION**