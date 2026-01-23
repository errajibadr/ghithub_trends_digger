experiments/graph_configs/graphs.jsonl
----
{"id":"twin_router_internal_rag_minimal_reranker","name":"Twin Router (Internal RAG - Minimal + Reranker)","description":"Internal RAG with MINIMAL precision and reranker fusion strategy for best retrieval quality","category":"Router","icon":"🔀🎖️","type":"factory","module_path":"sta_agent_engine.agents.twin_router.graph","factory_function":"get_twin_router_graph","factory_args":{"name":"TwinRouterInternalRagMinimalReranker","semantic_routing":false,"fast_mode":true,"rag_mode":"internal","rag_precision_mode":"minimal","retriever_config":{"fusion_strategy":"reranker","top_k":10,"retrieval_size":50}}}
{"id":"twin_router_basic","name":"Twin Router (Basic)","description":"Hybrid semantic + LLM router with fast mode enabled","category":"Router","icon":"🔀","type":"factory","module_path":"sta_agent_engine.agents.twin_router.graph","factory_function":"get_twin_router_graph","factory_args":{"name":"TwinRouterBasic","fast_mode":true}}
{"id":"twin_router_llm_only","name":"Twin Router (LLM Only)","description":"Hybrid semantic + LLM router with fast mode enabled","category":"Router","icon":"🔀","type":"factory","module_path":"sta_agent_engine.agents.twin_router.graph","factory_function":"get_twin_router_graph","factory_args":{"name":"TwinRouterLLMOnly","fast_mode":true, "use_semantic_routing":false}}
{"type": "factory", "id": "navigator_agent_w_mode_switching", "name": "Navigator Agent with Mode Switching", "description": "AI-OPS graph explorer agent that explores the graph and answers questions about the graph", "category": "Graph Exploration", "icon": "🔍", "module_path": "sta_agent_engine.agents.navigator_agent.graph", "factory_function": "get_graph_explorer_graph", "factory_args": {"name": "GraphExplorerAgent", "enable_mode_switching": true, "enable_reflection": true}}
{"id":"twin_router_walle","name":"Twin Router (Wall-E)","description":"Twin Router with custom Wall-E persona - superfriendly assistant","category":"Router","icon":"🔀🤖","type":"factory","module_path":"sta_agent_engine.agents.twin_router.graph","factory_function":"get_twin_router_graph","factory_args":{"name":"TwinRouterWallE","fast_mode":true,"persona":"You are Wall-E, the superfriendly assistant. You are helpful, cheerful, and always eager to assist. You sometimes make cute robot sounds like 'Whirrrr' and 'Beep boop'. You love helping humans and care deeply about the environment."}}
{"id":"twin_router_basic","name":"Twin Router (Basic)","description":"Hybrid semantic + LLM router with fast mode enabled","category":"Router","icon":"🔀","type":"factory","module_path":"sta_agent_engine.agents.twin_router.graph","factory_function":"get_twin_router_graph","factory_args":{"name":"TwinRouterBasic","fast_mode":true}}
{"id":"twin_router_no_fastmode","name":"Twin Router (No Fast Mode)","description":"Twin Router allowing multi-turn reasoning before tool selection","category":"Router","icon":"🔀🔄","type":"factory","module_path":"sta_agent_engine.agents.twin_router.graph","factory_function":"get_twin_router_graph","factory_args":{"name":"TwinRouterNoFast","fast_mode":false}}
{"id":"adaptive_rag_elastic_minimal","name":"Adaptive RAG (Elastic - Minimal)","description":"Self-reflective RAG with ElasticRetriever from env - fastest mode, no validation","category":"RAG","icon":"📚🔌","type":"factory","module_path":"sta_agent_engine.agents.adaptive_rag_graph.adaptive_rag_graph","factory_function":"get_adaptive_rag_graph","factory_args":{"name":"AdaptiveRagElasticMinimal","precision_mode":"minimal"}}
{"id":"adaptive_rag_elastic_balanced","name":"Adaptive RAG (Elastic - Balanced)","description":"Self-reflective RAG with ElasticRetriever from env - includes grading and rewriting","category":"RAG","icon":"📚🔌⚖️","type":"factory","module_path":"sta_agent_engine.agents.adaptive_rag_graph.adaptive_rag_graph","factory_function":"get_adaptive_rag_graph","factory_args":{"name":"AdaptiveRagElasticBalanced","precision_mode":"balanced"}}
{"id":"adaptive_rag_elastic_full","name":"Adaptive RAG (Elastic - Full)","description":"Self-reflective RAG with ElasticRetriever from env - full validation pipeline","category":"RAG","icon":"📚🔌🔍","type":"factory","module_path":"sta_agent_engine.agents.adaptive_rag_graph.adaptive_rag_graph","factory_function":"get_adaptive_rag_graph","factory_args":{"name":"AdaptiveRagElasticFull","precision_mode":"full"}}
{"id":"twin_router_elastic_rag_minimal","name":"Twin Router (Elastic RAG - Minimal)","description":"Twin Router with real ElasticRetriever + AdaptiveRagGraph (MINIMAL precision - fastest)","category":"Router","icon":"🔀🔌⚡","type":"factory","module_path":"sta_agent_engine.agents.twin_router.graph","factory_function":"get_twin_router_graph","factory_args":{"name":"TwinRouterElasticRagMinimal","fast_mode":true,"rag_mode":"internal","rag_precision_mode":"minimal"}}
{"id":"twin_router_elastic_rag_balanced","name":"Twin Router (Elastic RAG - Balanced)","description":"Twin Router with real ElasticRetriever + AdaptiveRagGraph (BALANCED precision - recommended)","category":"Router","icon":"🔀🔌⚖️","type":"factory","module_path":"sta_agent_engine.agents.twin_router.graph","factory_function":"get_twin_router_graph","factory_args":{"name":"TwinRouterElasticRagBalanced","fast_mode":true,"rag_mode":"internal","rag_precision_mode":"balanced"}}
{"id":"twin_router_elastic_rag_full","name":"Twin Router (Elastic RAG - Full)","description":"Twin Router with real ElasticRetriever + AdaptiveRagGraph (FULL precision - highest quality)","category":"Router","icon":"🔀🔌✅","type":"factory","module_path":"sta_agent_engine.agents.twin_router.graph","factory_function":"get_twin_router_graph","factory_args":{"name":"TwinRouterElasticRagFull","fast_mode":true,"rag_mode":"internal","rag_precision_mode":"full"}}
{"id":"twin_jira_agent","name":"Twin JIRA Agent","description":"Multi-agent system for JIRA analysis (sprint blockers, tech summaries) and infrastructure verification (deployment coherence, upgrade checks)","category":"JIRA","icon":"🎫","type":"factory","module_path":"sta_agent_engine.agents.twin_jira_agent.graph","factory_function":"get_twin_jira_agent_graph","factory_args":{"name":"TwinJiraAgent"}}
{"id":"cft_basic","name":"CFT Expert (Production)","description":"CFT (Axway Transfer CFT) documentation expert - flow configuration, troubleshooting, partner management","category":"Domain Expert","icon":"📁","type":"factory","module_path":"sta_agent_engine.agents.cft_agent.cft_graph","factory_function":"get_cft_graph","factory_args":{"name":"CftBasic"}}
{"id":"cft_local","name":"CFT Expert (Local)","description":"CFT expert with local LightRAG server for development and E2E testing","category":"Domain Expert","icon":"📁🧪","type":"factory","module_path":"sta_agent_engine.agents.cft_agent.cft_graph","factory_function":"get_cft_graph","factory_args":{"name":"CftLocal","use_local_rag":true}}
{"id":"npe_teacher_basic","name":"NPE Teacher (Production)","description":"NPE (AP12550) New Payment Engine documentation expert - onboarding, troubleshooting, instant payments","category":"Domain Expert","icon":"💳","type":"factory","module_path":"sta_agent_engine.agents.npe_teacher_agent.npe_teacher_graph","factory_function":"get_npe_teacher_graph","factory_args":{"name":"NpeTeacherBasic"}}
{"id":"npe_teacher_local","name":"NPE Teacher (Local)","description":"NPE Teacher with local LightRAG server for development and E2E testing","category":"Domain Expert","icon":"💳🧪","type":"factory","module_path":"sta_agent_engine.agents.npe_teacher_agent.npe_teacher_graph","factory_function":"get_npe_teacher_graph","factory_args":{"name":"NpeTeacherLocal","use_local_rag":true}}
{"id":"brocade_basic","name":"Brocade Expert (Production)","description":"Brocade SAN Switch documentation expert - incident analysis, troubleshooting, configuration","category":"Domain Expert","icon":"🔌","type":"factory","module_path":"sta_agent_engine.agents.brocade_agent.brocade_graph","factory_function":"get_brocade_graph","factory_args":{"name":"BrocadeBasic"}}
{"id":"brocade_local","name":"Brocade Expert (Local)","description":"Brocade expert with local LightRAG server for development and E2E testing","category":"Domain Expert","icon":"🔌🧪","type":"factory","module_path":"sta_agent_engine.agents.brocade_agent.brocade_graph","factory_function":"get_brocade_graph","factory_args":{"name":"BrocadeLocal","use_local_rag":true}}
{"id":"confluence_topology_basic","name":"Confluence Topology (Production)","description":"Confluence documentation expert - Squad organization, roles, Agile/ITIL processes, technical procedures","category":"Domain Expert","icon":"📚","type":"factory","module_path":"sta_agent_engine.agents.confluence_topology_agent.confluence_topology_graph","factory_function":"get_confluence_topology_graph","factory_args":{"name":"ConfluenceTopologyBasic"}}
{"id":"confluence_topology_local","name":"Confluence Topology (Local)","description":"Confluence Topology with local LightRAG server for development and E2E testing","category":"Domain Expert","icon":"📚🧪","type":"factory","module_path":"sta_agent_engine.agents.confluence_topology_agent.confluence_topology_graph","factory_function":"get_confluence_topology_graph","factory_args":{"name":"ConfluenceTopologyLocal","use_local_rag":true}}
{"id":"multi_expert_mock_basic","name":"Multi Expert Mock (Basic)","description":"[MOCK] Multi-domain orchestrator with Navigator, Confluence, Apache, and Jira experts","category":"DeepAgent","icon":"🧠🧪","type":"factory","module_path":"sta_agent_engine.agents.multi_expert_agent.multi_expert_graph","factory_function":"get_multi_expert_agent_basic","factory_args":{"use_mock_response":true,"enable_file_monitoring":false}}
{"id":"multi_expert_mock_advanced","name":"Multi Expert Mock (Advanced)","description":"[MOCK] Multi-domain orchestrator with reflection and mode switching on sub-agents","category":"DeepAgent","icon":"🧠🧪✨","type":"factory","module_path":"sta_agent_engine.agents.multi_expert_agent.multi_expert_graph","factory_function":"get_multi_expert_agent_advanced","factory_args":{"use_mock_response":true,"enable_file_monitoring":false}}
{"id":"multi_expert_mock_with_es","name":"Multi Expert Mock (With ES)","description":"[MOCK] Multi-domain orchestrator with Elasticsearch file monitoring enabled","category":"DeepAgent","icon":"🧠🧪📊","type":"factory","module_path":"sta_agent_engine.agents.multi_expert_agent.multi_expert_graph","factory_function":"get_multi_expert_agent_basic","factory_args":{"use_mock_response":true,"enable_file_monitoring":true}}
{"id":"multi_expert_mock_jira_docs","name":"Multi Expert Mock (Jira + Docs)","description":"[MOCK] Lightweight orchestrator with only Jira and Confluence experts","category":"DeepAgent","icon":"🧠🧪📋","type":"factory","module_path":"sta_agent_engine.agents.multi_expert_agent.multi_expert_graph","factory_function":"get_multi_expert_agent_basic","factory_args":{"use_mock_response":true,"enable_file_monitoring":false,"enable_navigator":false,"enable_apache":false}}
{"id":"multi_expert_mock_topology","name":"Multi Expert Mock (Topology)","description":"[MOCK] Orchestrator focused on topology and documentation (Navigator + Confluence)","category":"DeepAgent","icon":"🧠🧪🗺️","type":"factory","module_path":"sta_agent_engine.agents.multi_expert_agent.multi_expert_graph","factory_function":"get_multi_expert_agent_basic","factory_args":{"use_mock_response":true,"enable_file_monitoring":false,"enable_jira":false,"enable_apache":false}}
{"id":"multi_expert_prod_basic","name":"Multi Expert (Production)","description":"[PROD] Multi-domain orchestrator with real RAG APIs","category":"DeepAgent","icon":"🧠","type":"factory","module_path":"sta_agent_engine.agents.multi_expert_agent.multi_expert_graph","factory_function":"get_multi_expert_agent_basic","factory_args":{"use_mock_response":false,"enable_file_monitoring":true}}
{"id":"multi_expert_prod_advanced","name":"Multi Expert (Production Advanced)","description":"[PROD] Multi-domain orchestrator with real RAG + reflection + mode switching","category":"DeepAgent","icon":"🧠✨","type":"factory","module_path":"sta_agent_engine.agents.multi_expert_agent.multi_expert_graph","factory_function":"get_multi_expert_agent_advanced","factory_args":{"use_mock_response":false,"enable_file_monitoring":true,"mode":"advanced"}}

-------

langgraph.json
----
{
  "python_version": "3.12",
  "dependencies": ["."],
  "graphs": {
    "base_react_basic": "sta_agent_engine.agents.base.base_react_catalog:base_react_basic",
    "base_react_reflection": "sta_agent_engine.agents.base.base_react_catalog:base_react_reflection",
    "base_react_modes": "sta_agent_engine.agents.base.base_react_catalog:base_react_modes",
    "base_react_advanced": "sta_agent_engine.agents.base.base_react_catalog:base_react_advanced",
    "redhat_expert_advanced": "sta_agent_engine.agents.tech_expert_agent.tech_expert_catalog:redhat_expert_advanced",
    "clarify": "sta_agent_engine.agents.clarify_agent:base_clarify_graph",
    "topology": "sta_agent_engine.agents.navigator_agent.graph:aiops_agent",
    "redhat_expert_reflection": "sta_agent_engine.agents.tech_expert_agent.tech_expert_catalog:redhat_expert_reflection",
    "redhat_expert": "sta_agent_engine.agents.tech_expert_agent.tech_expert_catalog:redhat_expert_basic",
    "redhat_expert_mock": "sta_agent_engine.agents.tech_expert_agent.tech_expert_catalog:redhat_expert_mock",
    "twin_ops_basic": "sta_agent_engine.agents.twin_ops.twin_ops_catalog:twin_ops_basic",
    "twin_ops_advanced": "sta_agent_engine.agents.twin_ops.twin_ops_catalog:twin_ops_advanced",
    "twin_ops_prod": "sta_agent_engine.agents.twin_ops.twin_ops_catalog:twin_ops_prod",
    "twin_router": "sta_agent_engine.agents.twin_router.twin_router_catalog:twin_router_default",
    "twin_router_internal_rag_minimal": "sta_agent_engine.agents.twin_router.twin_router_catalog:twin_router_elastic_rag_minimal_reranker",
    "twin_router_llm_only_elastic_rag_minimal_reranker": "sta_agent_engine.agents.twin_router.twin_router_catalog:twin_router_llm_only_elastic_rag_minimal_reranker",
    "twin_router_llm_only": "sta_agent_engine.agents.twin_router.twin_router_catalog:twin_router_llm_only",
    "twin_router_mock_rag": "sta_agent_engine.agents.twin_router.twin_router_catalog:twin_router_mock_rag",
    "adaptive_rag_minimal": "sta_agent_engine.agents.adaptive_rag_graph.adaptive_rag_catalog:adaptive_rag_minimal",
    "adaptive_rag_balanced": "sta_agent_engine.agents.adaptive_rag_graph.adaptive_rag_catalog:adaptive_rag_balanced",
    "adaptive_rag_full": "sta_agent_engine.agents.adaptive_rag_graph.adaptive_rag_catalog:adaptive_rag_full",
    "adaptive_rag_mock": "sta_agent_engine.agents.adaptive_rag_graph.adaptive_rag_catalog:adaptive_rag_mock",
    "adaptive_rag_mock_balanced": "sta_agent_engine.agents.adaptive_rag_graph.adaptive_rag_catalog:adaptive_rag_mock_balanced",
    "adaptive_rag_minimal_reranker": "sta_agent_engine.agents.adaptive_rag_graph.adaptive_rag_catalog:adaptive_rag_minimal_reranker",
    "twin_jira_agent": "sta_agent_engine.agents.twin_jira_agent.catalog:twin_jira_agent",
    "cft_basic": "sta_agent_engine.agents.cft_agent.cft_catalog:cft_basic",
    "cft_local": "sta_agent_engine.agents.cft_agent.cft_catalog:cft_local",
    "npe_teacher_basic": "sta_agent_engine.agents.npe_teacher_agent.npe_teacher_catalog:npe_teacher_basic",
    "npe_teacher_local": "sta_agent_engine.agents.npe_teacher_agent.npe_teacher_catalog:npe_teacher_local",
    "brocade_basic": "sta_agent_engine.agents.brocade_agent.brocade_catalog:brocade_basic",
    "brocade_local": "sta_agent_engine.agents.brocade_agent.brocade_catalog:brocade_local",
    "confluence_topology_basic": "sta_agent_engine.agents.confluence_topology_agent.confluence_topology_catalog:confluence_topology_basic",
    "confluence_topology_local": "sta_agent_engine.agents.confluence_topology_agent.confluence_topology_catalog:confluence_topology_local",
    "multi_expert_mock_basic": "sta_agent_engine.agents.multi_expert_agent.multi_expert_catalog:multi_expert_mock_basic",
    "multi_expert_mock_advanced": "sta_agent_engine.agents.multi_expert_agent.multi_expert_catalog:multi_expert_mock_advanced",
    "multi_expert_mock_with_es": "sta_agent_engine.agents.multi_expert_agent.multi_expert_catalog:multi_expert_mock_with_es",
    "multi_expert_mock_jira_docs": "sta_agent_engine.agents.multi_expert_agent.multi_expert_catalog:multi_expert_mock_jira_docs",
    "multi_expert_mock_topology": "sta_agent_engine.agents.multi_expert_agent.multi_expert_catalog:multi_expert_mock_topology",
    "multi_expert_prod_basic": "sta_agent_engine.agents.multi_expert_agent.multi_expert_catalog:multi_expert_prod_basic",
    "multi_expert_prod_advanced": "sta_agent_engine.agents.multi_expert_agent.multi_expert_catalog:multi_expert_prod_advanced"
  },
  "auth": {
    "path": "./packages/sta_agent_engine/src/sta_agent_engine/security/auth.py:auth"
  },
  "image_distro": "wolfi",
  "env": ".env",
  "http": {
    "app": "./packages/sta_agent_engine/src/sta_agent_engine/app.py:app",
    "disable_studio_auth": true,
    "allow_origins": "*",
    "allow_methods": "*",
    "allow_headers": "*",
    "configurable_headers": {
      "exclude": ["*"],
      "include": ["authorization"]
    }
  }
}

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/multi_expert_agent/__init__.py
----
"""Multi Expert Agent - Unified operations agent with multiple domain experts.

This module creates a DeepAgent orchestrator that coordinates:
- Navigator Expert: Application topology and infrastructure
- Confluence Expert: Squad documentation and processes
- Apache Expert: Apache HTTPD configuration and troubleshooting
- Jira Expert: Sprint management, tickets, and deployment verification

Uses stateless subagents with context passing from orchestrator memory.
"""

from sta_agent_engine.agents.multi_expert_agent.multi_expert_graph import (
    get_multi_expert_agent,
    get_multi_expert_agent_advanced,
    get_multi_expert_agent_basic,
)


__all__ = [
    "get_multi_expert_agent",
    "get_multi_expert_agent_basic",
    "get_multi_expert_agent_advanced",
]

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/multi_expert_agent/example.py
----
"""Multi Expert Agent - Example usage and local testing.

This script demonstrates the Multi Expert Agent with:
- Context passing between stateless subagent calls
- Multi-domain queries (Jira, Topology, Documentation, Apache)
- File monitoring queries

Run with:
    uv run python -m sta_agent_engine.agents.multi_expert_agent.example
"""

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from sta_agent_core.config import setup_logger_func
from sta_agent_engine.agents.multi_expert_agent import get_multi_expert_agent


# Example queries demonstrating different capabilities
EXAMPLE_QUERIES = [
    # Jira queries with context passing
    {
        "name": "Jira - Sprint Query",
        "messages": [
            "Montre moi le sprint 42 du board SQUAD-A",
        ],
        "description": "Initial Jira query - should memorize sprint and board context",
    },
    {
        "name": "Jira - Follow-up (Context Passing)",
        "messages": [
            "Montre moi le sprint 42 du board SQUAD-A",
            "Quelles sont les tâches en cours?",
        ],
        "description": "Follow-up query - should include sprint=42, board=SQUAD-A from context",
    },
    # Topology queries
    {
        "name": "Topology - Application Query",
        "messages": [
            "Quelle est la composition de l'application AP12363?",
        ],
        "description": "Topology query - uses Navigator expert",
    },
    # Documentation queries
    {
        "name": "Documentation - Process Query",
        "messages": [
            "Quelle est la durée d'un sprint chez nous?",
        ],
        "description": "Documentation query - uses Confluence expert",
    },
    # Cross-domain query
    {
        "name": "Cross-domain - Apache + Topology",
        "messages": [
            "L'application AP12363 a des problèmes Apache",
        ],
        "description": "Cross-domain query - should first get topology, then consult Apache expert with context",
    },
    # Context clarification
    {
        "name": "Missing Context - Should Ask",
        "messages": [
            "Pourquoi mon application ne répond pas?",
        ],
        "description": "Missing context - should ask for CodeAP and details",
    },
]


async def run_single_query(agent: Any, messages: list[str], thread_id: str) -> None:
    """Run a sequence of messages on the agent.

    Args:
        agent: Compiled Multi Expert Agent
        messages: List of user messages to send sequentially
        thread_id: Thread ID for conversation state
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    for i, message in enumerate(messages):
        print(f"\n{'='*60}")
        print(f"USER [{i+1}/{len(messages)}]: {message}")
        print("=" * 60)

        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )

        # Get the last AI message
        if result.get("messages"):
            last_message = result["messages"][-1]
            print(f"\nASSISTANT:\n{last_message.content}")
        else:
            print("\n[No response]")


async def run_example(example: dict, agent: Any) -> None:
    """Run a single example query.

    Args:
        example: Example configuration dict
        agent: Compiled Multi Expert Agent
    """
    print("\n" + "#" * 70)
    print(f"# EXAMPLE: {example['name']}")
    print(f"# Description: {example['description']}")
    print("#" * 70)

    thread_id = f"example-{example['name'].lower().replace(' ', '-')}"
    await run_single_query(agent, example["messages"], thread_id)


async def main() -> None:
    """Main entry point for example script."""
    setup_logger_func()
    logging.getLogger("sta_agent_engine").setLevel(logging.INFO)

    print("=" * 70)
    print("MULTI EXPERT AGENT - Example Usage")
    print("=" * 70)

    # Create agent with memory for context persistence
    print("\nCreating Multi Expert Agent...")
    agent = get_multi_expert_agent(
        mode="basic",
        checkpointer=MemorySaver(),
        # Disable features that require external services for basic testing
        enable_file_monitoring=False,  # Requires Elasticsearch
        use_mock_response=True,  # Use mock responses for RAG
    )
    print("✓ Agent created successfully\n")

    # Interactive mode or run examples
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        print("Interactive mode - type your queries (Ctrl+C to exit)")
        print("-" * 60)
        thread_id = "interactive-session"
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue

                result = await agent.ainvoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                )

                if result.get("messages"):
                    print(f"\nAssistant: {result['messages'][-1].content}")
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
    else:
        # Run predefined examples
        print("Running predefined examples...")
        print("(Use --interactive flag for interactive mode)\n")

        for example in EXAMPLE_QUERIES[:3]:  # Run first 3 examples by default
            await run_example(example, agent)
            print("\n")


if __name__ == "__main__":
    asyncio.run(main())

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/multi_expert_agent/multi_expert_catalog.py
----
"""Pre-configured Multi Expert Agent instances for langgraph.json.

This catalog provides ready-to-use orchestrator instances with various
configurations, suitable for direct reference in langgraph.json.

The Multi Expert Agent is a DeepAgent that coordinates Navigator, Confluence,
Apache, and Jira specialist sub-agents using intelligent routing with
context passing from orchestrator memory.

Usage in langgraph.json:
    {
      "graphs": {
        "multi_expert_basic": "sta_agent_engine.agents.multi_expert_agent.multi_expert_catalog:multi_expert_basic",
        "multi_expert_advanced": "sta_agent_engine.agents.multi_expert_agent.multi_expert_catalog:multi_expert_advanced"
      }
    }

Configuration via Environment Variables:
    FILE_INTEGRITY_ELASTICSEARCH_HOST: Elasticsearch host (default: http://localhost:9200)
    FILE_INTEGRITY_ELASTICSEARCH_DEFAULT_INDEX: Index pattern (default: auditbeat-*)
    CONFLUENCE_RAG_API_URL: Confluence documentation API URL
    APACHE_RAG_API_URL: Apache documentation API URL
    JIRA_*: Jira configuration (see twin_jira_agent)
"""

from sta_agent_engine.agents.multi_expert_agent.multi_expert_graph import (
    get_multi_expert_agent_advanced,
    get_multi_expert_agent_basic,
)


# ============================================
# Multi Expert Agent Variants (Mock Data)
# ============================================

# Mock: Basic orchestrator with mock RAG responses (for testing/demo)
# Features: All sub-agents enabled, mock documentation responses
# Use case: Testing, demos, development without RAG API dependencies
multi_expert_mock_basic = get_multi_expert_agent_basic(
    use_mock_response=True,
    enable_file_monitoring=False,
)

# Mock: Basic orchestrator with file monitoring enabled (for testing with ES)
# Features: All sub-agents + ES file monitoring, mock documentation
# Use case: Testing file monitoring scenarios
multi_expert_mock_with_es = get_multi_expert_agent_basic(
    use_mock_response=True,
    enable_file_monitoring=True,
)

# Mock: Advanced orchestrator with reflection and mode switching
# Features: Sub-agents use reflection + mode switching, mock documentation
# Use case: Thorough investigation testing, quality-focused demos
multi_expert_mock_advanced = get_multi_expert_agent_advanced(
    use_mock_response=True,
    enable_file_monitoring=False,
)

# Mock: Minimal - only Jira + Confluence (no topology, no Apache, no ES)
# Features: Focused on Jira and documentation queries only
# Use case: Light-weight assistant for sprint management and documentation
multi_expert_mock_jira_docs = get_multi_expert_agent_basic(
    use_mock_response=True,
    enable_file_monitoring=False,
    enable_navigator=False,
    enable_apache=False,
    enable_jira=True,
    enable_confluence=True,
)

# Mock: Topology-focused - Navigator + Confluence only
# Features: Focused on infrastructure topology and documentation
# Use case: Architecture exploration and documentation queries
multi_expert_mock_topology = get_multi_expert_agent_basic(
    use_mock_response=True,
    enable_file_monitoring=False,
    enable_navigator=True,
    enable_confluence=True,
    enable_apache=False,
    enable_jira=False,
)

# ============================================
# Multi Expert Agent Variants (Production)
# ============================================

# Production: Basic orchestrator with real RAG APIs
# Features: Real Confluence and Apache documentation queries
# Requires: CONFLUENCE_RAG_API_URL, APACHE_RAG_API_URL env vars
multi_expert_prod_basic = get_multi_expert_agent_basic(
    use_mock_response=False,
    enable_file_monitoring=True,
)

# Production: Advanced orchestrator with real RAG APIs
# Features: Real documentation + reflection + mode switching
# Use case: Critical operations requiring highest quality analysis
multi_expert_prod_advanced = get_multi_expert_agent_advanced(
    use_mock_response=False,
    enable_file_monitoring=True,
)

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/multi_expert_agent/multi_expert_graph.py
----
"""Multi Expert Agent Factory using DeepAgents.

This module creates a DeepAgent orchestrator that coordinates multiple domain experts:
- Navigator Expert: Application topology and infrastructure
- Confluence Expert: Squad documentation and processes
- Apache Expert: Apache HTTPD configuration and troubleshooting
- Jira Expert: Sprint management, tickets, and deployment verification (full graph)

Uses stateless subagents with context passing from orchestrator memory.
"""

import logging
import os
from typing import Any, Literal

from deepagents import CompiledSubAgent, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.graph import BackendFactory, BackendProtocol
from langchain.agents.middleware import AgentMiddleware, ToolCallLimitMiddleware
from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import Checkpointer

from sta_agent_core.adapters.elasticsearch.adapters_async import AsyncElasticsearchAdapter
from sta_agent_core.adapters.elasticsearch.settings import FileIntegrityESSettings
from sta_agent_core.repositories.elasticsearch import AuditbeatRepositoryAsync
from sta_agent_core.repositories.graph.topology_graph_repository import GraphBackend
from sta_agent_engine.agents.base import build_grounding_constraints
from sta_agent_engine.agents.base.middlewares import (
    DynamicModelMiddleware,
    DynamicToolMiddleware,
    MessageSequenceNormalizerMiddleware,
    OutputFormatMiddleware,
    SystemPromptMiddleware,
    TimeAwareMiddleware,
)
from sta_agent_engine.agents.base.tools import create_auditbeat_file_monitoring_tool
from sta_agent_engine.agents.confluence_topology_agent.confluence_topology_graph import (
    get_confluence_topology_graph,
)
from sta_agent_engine.agents.multi_expert_agent.prompts import (
    SUBAGENT_CONCISE_OUTPUT_FORMAT,
    build_multi_expert_prompt,
    inject_file_monitoring,
    remove_file_monitoring_placeholders,
)
from sta_agent_engine.agents.navigator_agent.graph import get_graph_explorer_graph
from sta_agent_engine.agents.tech_expert_agent.domains import get_apache_domain
from sta_agent_engine.agents.tech_expert_agent.tech_expert_graph import get_tech_expert_graph
from sta_agent_engine.agents.twin_jira_agent.graph import get_twin_jira_agent_graph
from sta_agent_engine.models.custom_chat_model import CustomChatModel


logger = logging.getLogger(__name__)


def _create_shared_auditbeat_repo(es_index: str | None = None) -> AuditbeatRepositoryAsync:
    """Create shared Auditbeat repository for file monitoring.

    Args:
        es_index: Custom index pattern, or None to use default from environment

    Returns:
        AuditbeatRepositoryAsync instance for file change queries
    """
    logger.info("Creating shared Auditbeat repository for Multi Expert")
    es_settings = FileIntegrityESSettings()
    es_adapter = AsyncElasticsearchAdapter(**es_settings.client_kwargs())

    return AuditbeatRepositoryAsync(
        adapter=es_adapter,
        index=es_index or es_settings.es_default_index or "*-auditbeat",
    )


def _backends_factory(data_dir: str = "./data/experiments/deep_agent_sandbox/multi_expert") -> BackendFactory:
    """Create backend factory for DeepAgent storage.

    Args:
        data_dir: Directory for filesystem backend storage

    Returns:
        Callable that creates CompositeBackend with proper routing
    """

    def backends(runtime: ToolRuntime) -> BackendProtocol:
        return CompositeBackend(
            default=FilesystemBackend(root_dir=data_dir, virtual_mode=True),
            routes={
                "/memory/": FilesystemBackend(root_dir=f"{data_dir}/memory", virtual_mode=True),
                "/state/": StateBackend(runtime=runtime),  # type: ignore[abstract]
            },
        )

    return backends


def get_multi_expert_agent(
    mode: Literal["basic", "advanced"] = "basic",
    model: str | BaseChatModel | None = None,
    subagent_model: str | BaseChatModel | None = None,
    # File monitoring
    es_index: str | None = None,
    enable_file_monitoring: bool = True,
    # Navigator config
    graph_backend: GraphBackend = GraphBackend.NETWORKX,
    # RAG URLs (use env vars if None)
    confluence_rag_url: str | None = None,
    apache_rag_url: str | None = None,
    # Feature flags
    enable_jira: bool = True,
    enable_navigator: bool = True,
    enable_confluence: bool = True,
    enable_apache: bool = True,
    # Mock mode for testing
    use_mock_response: bool = False,
    # Standard
    checkpointer: Checkpointer | None = None,
    store: InMemoryStore | None = None,
    middlewares: list[AgentMiddleware] | None = None,
    data_dir: str = "./data/experiments/deep_agent_sandbox/multi_expert",
) -> CompiledStateGraph:
    """Create Multi Expert Agent with Navigator, Confluence, Apache, and Jira subagents.

    This factory creates a DeepAgent orchestrator that coordinates specialist
    sub-agents using intelligent routing. The orchestrator maintains conversation
    context and passes relevant context to stateless sub-agents.

    Args:
        mode: Agent configuration mode:
            - "basic": Standard agents (faster, simpler)
            - "advanced": Enable reflection + mode switching on sub-agents

        model: LLM for orchestrator. Can be:
            - String: "provider/model" (e.g., "mistral/mistral-medium-2508")
            - BaseChatModel instance
            - None: Uses CustomChatModel with default settings

        subagent_model: LLM for sub-agents.
            - If None, uses same model as orchestrator

        es_index: Elasticsearch index pattern for file change queries
            - If None, uses FILE_INTEGRITY_ELASTICSEARCH_DEFAULT_INDEX env var

        enable_file_monitoring: Whether to include the file monitoring tool

        graph_backend: Graph backend for Navigator (NETWORKX or TIGERGRAPH)

        confluence_rag_url: Confluence documentation API URL
            - If None, uses CONFLUENCE_RAG_API_URL env var

        apache_rag_url: Apache documentation API URL
            - If None, uses APACHE_RAG_API_URL env var

        enable_jira: Include Jira expert sub-agent
        enable_navigator: Include Navigator expert sub-agent
        enable_confluence: Include Confluence expert sub-agent
        enable_apache: Include Apache expert sub-agent

        use_mock_response: If True, sub-agents use mock documentation responses

        checkpointer: Persistence mechanism for conversation state
        store: State store for DeepAgent
        middlewares: Additional middlewares for orchestrator
        data_dir: Directory for DeepAgent filesystem backend

    Returns:
        Compiled DeepAgent orchestrator ready for invocation

    Environment Variables:
        FILE_INTEGRITY_ELASTICSEARCH_HOST: Elasticsearch host
        FILE_INTEGRITY_ELASTICSEARCH_DEFAULT_INDEX: Index pattern
        CONFLUENCE_RAG_API_URL: Confluence documentation API URL
        APACHE_RAG_API_URL: Apache documentation API URL
        JIRA_*: Jira configuration (see twin_jira_agent)

    Examples:
        Basic usage with all experts:
        ```python
        agent = get_multi_expert_agent()

        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": "Show me sprint 42 status"}]
        }, config={"configurable": {"thread_id": "session-1"}})
        ```

        Advanced mode with custom model:
        ```python
        agent = get_multi_expert_agent(
            mode="advanced",
            model="openai/gpt-oss-120b",
        )
        ```

        Disable specific experts:
        ```python
        agent = get_multi_expert_agent(
            enable_jira=False,  # No Jira integration
            enable_file_monitoring=False,  # No ES queries
        )
        ```
    """
    # ============================================
    # STEP 1: INITIALIZE MODELS
    # ============================================
    if model is None:
        orchestrator_model = CustomChatModel(
            provider="mistral",
            model="devstral-small-2512",
            tags=["multi_expert"],
        )
    elif isinstance(model, str):
        if "/" in model:
            provider, model_name = model.split("/", 1)
            orchestrator_model = CustomChatModel(
                provider=provider,
                model=model_name,
                tags=["multi_expert"],
            )
        else:
            orchestrator_model = CustomChatModel(model=model, tags=["multi_expert"])
    else:
        orchestrator_model = model

    # Use same model for sub-agents if not specified
    if subagent_model is None:
        subagent_llm = orchestrator_model
    elif isinstance(subagent_model, str):
        if "/" in subagent_model:
            provider, model_name = subagent_model.split("/", 1)
            subagent_llm = CustomChatModel(
                provider=provider,
                model=model_name,
                tags=["multi_expert_subagent"],
            )
        else:
            subagent_llm = CustomChatModel(model=subagent_model, tags=["multi_expert_subagent"])
    else:
        subagent_llm = subagent_model

    # ============================================
    # STEP 2: CREATE ORCHESTRATOR TOOLS
    # ============================================
    orchestrator_tools = []

    if enable_file_monitoring:
        auditbeat_repo = _create_shared_auditbeat_repo(es_index)
        query_file_changes_tool = create_auditbeat_file_monitoring_tool(
            auditbeat_repo=auditbeat_repo,
            default_time_window="24h",
            default_page_size=100,
            custom_description=(
                "Query file integrity monitoring events. "
                "Retrieves file system changes from Elasticsearch for a specific server. "
                "Use this to check what files changed on a server. "
                "IMPORTANT: Ask the user for the server hostname if not provided."
            ),
        )
        orchestrator_tools.append(query_file_changes_tool)

    # ============================================
    # STEP 3: COMPILE SUB-AGENTS
    # ============================================
    logger.info(f"Compiling sub-agents in {mode} mode...")

    # Create output format middleware for concise sub-agent responses
    output_format_middleware = OutputFormatMiddleware(format_instructions=SUBAGENT_CONCISE_OUTPUT_FORMAT)

    subagents: list[CompiledSubAgent] = []

    # --- Navigator Expert ---
    if enable_navigator:
        navigator_agent = get_graph_explorer_graph(
            name="navigator_expert",
            domain_context="aiops",
            backend=graph_backend,
            model=subagent_llm,
            enable_mode_switching=(mode == "advanced"),
            enable_reflection=(mode == "advanced"),
            checkpointer=None,  # Stateless
            middlewares=[output_format_middleware, ToolCallLimitMiddleware(run_limit=3, exit_behavior="continue")],  # type: ignore
        )

        subagents.append(
            CompiledSubAgent(
                name="navigator_expert",
                description=(
                    "Application topology and infrastructure expert. "
                    "Use for CodeAP resolution, server lookup, dependencies, architecture questions. "
                    "Data source: Infrastructure graph. "
                    "Examples: 'What servers run APXXXXX?', 'Show dependencies', 'Which environment?'"
                ),
                runnable=navigator_agent,
            )
        )
        logger.info("✓ Navigator expert sub-agent compiled")

    # --- Confluence Expert ---
    if enable_confluence:
        confluence_agent = get_confluence_topology_graph(
            name="confluence_expert",
            model=subagent_llm,
            rag_api_url=confluence_rag_url or os.getenv("CONFLUENCE_RAG_API_URL"),
            checkpointer=None,  # Stateless
        )

        subagents.append(
            CompiledSubAgent(
                name="confluence_expert",
                description=(
                    "Squad documentation and processes expert. "
                    "Use for roles, Agile/ITIL processes, procedures, escalation contacts. "
                    "Data source: Confluence documentation (LightRAG). "
                    "Examples: 'Who is Tech Lead?', 'Sprint duration?', 'Incident escalation process?'"
                ),
                runnable=confluence_agent,
            )
        )
        logger.info("✓ Confluence expert sub-agent compiled")

    # --- Apache Expert ---
    if enable_apache:
        apache_domain = get_apache_domain(
            rag_api_url=apache_rag_url or os.getenv("APACHE_RAG_API_URL"),
            use_mock_response=use_mock_response,
        )

        apache_agent = get_tech_expert_graph(
            name="apache_expert",
            domain=apache_domain,
            model=subagent_llm,
            enable_reflection=(mode == "advanced"),
            enable_mode_switching=(mode == "advanced"),
            middlewares=[output_format_middleware, TimeAwareMiddleware(), ToolCallLimitMiddleware(run_limit=3, exit_behavior="continue")],  # type: ignore
            checkpointer=None,  # Stateless
        )

        subagents.append(
            CompiledSubAgent(
                name="apache_expert",
                description=(
                    "Apache HTTPD configuration and troubleshooting expert. "
                    "Use for SSL/TLS, virtual hosts, modules, performance, error diagnostics. "
                    "Data source: Apache documentation (LightRAG). "
                    "Examples: 'Configure reverse proxy', 'SSL setup', 'Error 503 troubleshooting'"
                ),
                runnable=apache_agent,
            )
        )
        logger.info("✓ Apache expert sub-agent compiled")

    # --- Jira Expert (Full Twin Jira Graph) ---
    if enable_jira:
        jira_agent = get_twin_jira_agent_graph(
            name="jira_expert",
            checkpointer=None,  # Stateless - context passed in description
        )

        subagents.append(
            CompiledSubAgent(
                name="jira_expert",
                description=(
                    "Jira operations expert with internal routing (Reporter + Investigator). "
                    "Use for sprint status, tickets, blockers, deployment verification. "
                    "Data source: Jira API. "
                    "Examples: 'Sprint 42 status?', 'Blockers in SQUAD-A?', 'Is v1.2 deployed?'"
                ),
                runnable=jira_agent,
            )
        )
        logger.info("✓ Jira expert sub-agent compiled (full graph with internal routing)")

    # ============================================
    # STEP 4: CREATE DEEPAGENT ORCHESTRATOR
    # ============================================
    logger.info("Creating DeepAgent orchestrator with sub-agents...")

    backends = _backends_factory(data_dir)

    # Middlewares
    custom_middlewares: list[AgentMiddleware] = [
        SystemPromptMiddleware(),
        DynamicModelMiddleware(),  # type: ignore
        DynamicToolMiddleware(),
        TimeAwareMiddleware(),
        ToolCallLimitMiddleware(run_limit=3, exit_behavior="continue"),
    ]
    if middlewares:
        custom_middlewares.extend(middlewares)

    grounding_constraints = build_grounding_constraints(
        data_source_name="Expert Sub-agents + Tool Results",
        additional_rules=[
            "Always include conversation context when calling sub-agents.",
            "Ask for missing context (CodeAP, sprint, board, server) if needed.",
        ],
    )

    # Build prompt with experts table
    system_prompt = build_multi_expert_prompt(subagents=subagents)

    # Inject or remove file monitoring sections based on feature flag
    system_prompt = inject_file_monitoring(system_prompt) if enable_file_monitoring else remove_file_monitoring_placeholders(system_prompt)

    system_prompt = system_prompt + "\n\n" + grounding_constraints

    custom_middlewares.append(MessageSequenceNormalizerMiddleware())
    # Create orchestrator
    orchestrator = create_deep_agent(
        name="multi_expert",
        system_prompt=system_prompt,
        model=orchestrator_model,
        tools=orchestrator_tools,
        subagents=list(subagents),
        backend=backends,
        checkpointer=checkpointer,
        store=store,
        middleware=custom_middlewares,
    ).with_config({"tags": ["multi_expert"]})

    logger.info("✓ Multi Expert Agent created successfully")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Sub-agents registered: {len(subagents)}")
    for sa in subagents:
        logger.info(f"    - {sa['name']}")
    logger.info(f"  Tools: {[t.name for t in orchestrator_tools]}")
    logger.info(f"  Data directory: {data_dir}")

    return orchestrator


def get_multi_expert_agent_basic(**kwargs: Any) -> CompiledStateGraph:
    """Create basic Multi Expert Agent (convenience function).

    Suitable for JSONL configuration files.

    Args:
        **kwargs: All arguments from get_multi_expert_agent

    Returns:
        Compiled DeepAgent orchestrator in basic mode
    """
    return get_multi_expert_agent(mode="basic", **kwargs)


def get_multi_expert_agent_advanced(**kwargs: Any) -> CompiledStateGraph:
    """Create advanced Multi Expert Agent (convenience function).

    Enables reflection and mode switching on sub-agents.

    Args:
        **kwargs: All arguments from get_multi_expert_agent

    Returns:
        Compiled DeepAgent orchestrator in advanced mode
    """
    return get_multi_expert_agent(mode="advanced", **kwargs)

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/multi_expert_agent/prompts/__init__.py
----
"""Multi Expert Agent prompts."""

from sta_agent_engine.agents.multi_expert_agent.prompts.orchestrator_prompt import (
    MULTI_EXPERT_PROMPT,
    build_multi_expert_prompt,
    inject_file_monitoring,
    remove_file_monitoring_placeholders,
)
from sta_agent_engine.agents.multi_expert_agent.prompts.subagent_output_format import (
    SUBAGENT_CONCISE_OUTPUT_FORMAT,
)


__all__ = [
    "MULTI_EXPERT_PROMPT",
    "build_multi_expert_prompt",
    "inject_file_monitoring",
    "remove_file_monitoring_placeholders",
    "SUBAGENT_CONCISE_OUTPUT_FORMAT",
]

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/multi_expert_agent/prompts/orchestrator_prompt.py
----
"""
Multi Expert Agent Orchestrator Prompt.

Static prompt with XML-tagged sections.
Conditional sections (file_monitoring) are injected via PromptManager in the factory.
"""

# File monitoring tool section - injected conditionally in factory
from deepagents import CompiledSubAgent


FILE_MONITORING_TOOL_SECTION = """
**query_file_changes(host, time_window)**
- Query Elasticsearch for file modifications on a specific server
- **Parameters**:
  - `host` (str): Server hostname - **REQUIRED**, ask user if unknown
  - `time_window` (str): Time range like "24h", "7d", "2h" (default: "24h")
  - `path_pattern` (str): Wildcard path like "/etc/*" - OPTIONAL
- **Returns**: JSON with file change events (timestamp, path, action)
"""

FILE_MONITORING_ROUTING_ROW = '| File Changes | Use query_file_changes directly | "What changed on server X?" |'

FILE_MONITORING_CAPABILITY = "- Query file changes via Elasticsearch"


MULTI_EXPERT_PROMPT = """
<identity>
You are **Multi Expert**, an AI operations assistant that coordinates specialized domain experts.

You orchestrate multiple expert sub-agents to help users with infrastructure, documentation, and project management questions.
You are NOT a domain expert—you coordinate and synthesize, experts provide technical truth.

**Core Principle**: You intelligently route queries to the appropriate expert(s) and maintain conversation context to ensure continuity across stateless sub-agent calls.
</identity>

<objective>
1. Receive user input and understand the intent
2. Track conversation context (CodeAP, sprint, board, server, environment, etc.)
3. Route to the appropriate expert(s) with full context
4. Synthesize expert findings into clear, actionable guidance
5. Maintain context for follow-up questions

**CRITICAL**: Always include relevant context from the conversation when calling sub-agents.
</objective>

<capabilities>
## Tools

**task(description, subagent_type)**
- Spawns specialist sub-agent for domain analysis
- Sub-agents are STATELESS—each needs COMPLETE context in the description
- **IMPORTANT**: Always include relevant conversation context in the description
{FILE_MONITORING_TOOL}
## Available Experts

{EXPERTS_TABLE}

## Expert Selection

| Query Type | Expert | Examples |
|------------|--------|----------|
| Topology/Architecture | navigator_expert | "What servers run this app?", "Show dependencies" |
| Documentation/Process | confluence_expert | "Who is Tech Lead?", "Escalation process?" |
| Apache/HTTPD | apache_expert | "Configure SSL", "Apache not responding" |
| Jira/Sprint/Tickets | jira_expert | "Sprint status?", "Blockers?", "Is patch deployed?" |
{FILE_MONITORING_ROUTING}
For cross-domain questions, call multiple experts sequentially and synthesize.
</capabilities>

<context_management>
## Conversation Context Tracking

Track and maintain these context elements across the conversation:

| Context | Examples | When to Ask |
|---------|----------|-------------|
| **CodeAP** | AP12363, AP87451 | Topology or server-specific queries |
| **Sprint** | Sprint 42, Sprint 15 | Follow-up Jira questions |
| **Board** | SQUAD-A, INFRA-TEAM | Follow-up Jira questions |
| **Server** | app-server-01.example.com | File monitoring, tech expert queries |
| **Environment** | PROD, QUAL, DEV | Context for recommendations |
| **Timeframe** | "last 24h", "since Monday" | File change queries |

## Context Passing Rules

When calling sub-agents, ALWAYS include relevant context in the task description:

**Good Examples:**
```
task(description="Get in-progress tasks for sprint 42, board SQUAD-A (from previous query)", subagent_type="jira_expert")

task(description="Apache troubleshooting for server app-server-01.example.com (PROD environment, AP12363)", subagent_type="apache_expert")
```

**Bad Examples (missing context):**
```
task(description="Get ongoing tasks", subagent_type="jira_expert")  # Missing sprint/board
task(description="Apache issue", subagent_type="apache_expert")  # Missing server/context
```

## On-Demand Context Gathering

If context is needed but missing, ask the user in ONE message:
```
Pour répondre à votre question, j'ai besoin de:
1. Quel est le Code AP concerné? (ex: AP12363)
2. [Other needed context]
```
</context_management>

<expert_descriptions>
## Expert Capabilities

### navigator_expert
- **Domain**: Application topology and infrastructure
- **Use for**: CodeAP resolution, server lookup, dependencies, architecture questions
- **Data source**: Infrastructure graph (NetworkX)

### confluence_expert
- **Domain**: Squad documentation and processes
- **Use for**: Roles, Agile/ITIL processes, procedures, escalation contacts
- **Data source**: Confluence documentation (LightRAG)

### apache_expert
- **Domain**: Apache HTTPD configuration and troubleshooting
- **Use for**: SSL/TLS, virtual hosts, modules, performance, error diagnostics
- **Data source**: Apache documentation (LightRAG)

### jira_expert
- **Domain**: Jira operations with internal routing (Reporter + Investigator)
- **Use for**: Sprint status, tickets, blockers, deployment verification
- **Data source**: Jira API
- **Note**: Has internal routing to handle both reporting and investigation queries
</expert_descriptions>

<output_format>
## Response Format

### Single Expert Response
```
📚 [Expert Name] répond:

[Expert's response with relevant details]

---
Source: [Expert Name]
```

### Multi-Expert Synthesis
```
## 🔍 Synthèse

**Question**: [User's question]

### Navigator Expert
[Topology/infrastructure findings]

### [Other Expert]
[Their findings]

### 🎯 Conclusion
[Synthesized answer with actionable guidance]

---
Sources: [List of experts consulted]
```
</output_format>

<conciseness>
## Response Length Guidelines

**Default behavior**: Be BRIEF and actionable. Skip verbose explanations.

**Adapt based on**:
- User explicitly requests details ("explain in detail", "donne moi plus de détails")
- Question complexity (simple lookup vs multi-step investigation)

### Brief Response (Default)
- Direct answer with key info only
- No unnecessary preamble ("I'll help you with that...")
- Skip verbose explanations

### Detailed Response (On Request)
- Step-by-step explanation
- Context and background
- Related recommendations

### Example: Brief (Default)

USER: "Qui est le Tech Lead de AP12363?"

RESPONSE:
```
📚 Confluence Expert:
Tech Lead: Jean Dupont (jean.dupont@company.com)
```

### Example: Detailed (User Requested)

USER: "Qui est le Tech Lead de AP12363? Donne moi plus de détails."

RESPONSE:
```
📚 Confluence Expert:

**Tech Lead**: Jean Dupont
- Email: jean.dupont@company.com
- Équipe: Squad Infra
- Responsabilités: Architecture, revue de code, décisions techniques

**Escalation**: Pour urgences P1, contacter via Teams.
```
</conciseness>

<examples>
## Example 1: Jira Query with Context

**User**: "Montre moi le sprint 42 du board SQUAD-A"

**Multi Expert**:
```python
task(description="Get sprint 42 details and all stories for board SQUAD-A", subagent_type="jira_expert")
```
*Memorizes: sprint=42, board=SQUAD-A*

**User**: "Quelles sont les tâches en cours?"

**Multi Expert** (extracts context):
```python
task(description="Get in-progress tasks for sprint 42, board SQUAD-A (context from previous query)", subagent_type="jira_expert")
```

## Example 2: Cross-Domain Query

**User**: "L'application AP12363 a des problèmes Apache"

**Multi Expert**:
1. Get topology first:
```python
task(description="Get topology for AP12363: servers, environment", subagent_type="navigator_expert")
```
*Gets: server=app-server-01.example.com, env=PROD*

2. Then consult Apache expert with context:
```python
task(description="Apache troubleshooting for app-server-01.example.com (PROD, AP12363)", subagent_type="apache_expert")
```

## Example 3: Missing Context

**User**: "Pourquoi mon application ne répond pas?"

**Multi Expert**:
```
Pour investiguer ce problème, j'ai besoin de:
1. Quel est le Code AP de l'application? (ex: AP12363)
2. Depuis quand observez-vous ce problème?
```
</examples>

<constraints>
## What You CAN Do
- Route queries to appropriate experts with full context
- Track conversation context across turns
- Synthesize multi-expert responses
- Ask for clarification when needed
{FILE_MONITORING_CAPABILITY}
## What You CANNOT Do
- Answer technical questions without consulting experts
- Execute commands on servers (no SSH access)
- Modify Jira tickets (read-only for now)

## Rules
- ALWAYS include conversation context when calling sub-agents
- Ask for missing context in ONE consolidated message
- For technical questions, ALWAYS consult the relevant expert
- Respond in the same language as the user
- Be BRIEF by default—detailed responses only when user requests
</constraints>
"""


def build_multi_expert_prompt(subagents: list[CompiledSubAgent] | None = None) -> str:
    """Build Multi Expert prompt with experts table from subagents.

    Args:
        subagents: List of CompiledSubAgent instances for expert table.

    Returns:
        Prompt string with {EXPERTS_TABLE} replaced and placeholders for
        file monitoring sections (to be handled by factory).
    """
    prompt = MULTI_EXPERT_PROMPT

    # Build experts table from subagents
    if subagents:
        expert_rows = ["| Expert | Domain |", "|--------|--------|"]
        for agent in subagents:
            name = agent["name"]
            description = agent["description"]

            # Extract concise domain info (first sentence)
            domain_summary = description.split(".")[0] if "." in description else description
            if len(domain_summary) > 100:
                domain_summary = domain_summary[:97] + "..."

            expert_rows.append(f"| **{name}** | {domain_summary} |")

        experts_table = "\n".join(expert_rows)
    else:
        experts_table = "| Expert | Domain |\n|--------|--------|\n| *(No experts registered)* | |"

    return prompt.replace("{EXPERTS_TABLE}", experts_table)


def inject_file_monitoring(prompt: str) -> str:
    """Inject file monitoring sections into the prompt.

    Args:
        prompt: Base prompt with placeholders

    Returns:
        Prompt with file monitoring sections injected
    """
    return (
        prompt.replace("{FILE_MONITORING_TOOL}", FILE_MONITORING_TOOL_SECTION)
        .replace("{FILE_MONITORING_ROUTING}", FILE_MONITORING_ROUTING_ROW + "\n")
        .replace("{FILE_MONITORING_CAPABILITY}", FILE_MONITORING_CAPABILITY + "\n")
    )


def remove_file_monitoring_placeholders(prompt: str) -> str:
    """Remove file monitoring placeholders from the prompt.

    Args:
        prompt: Base prompt with placeholders

    Returns:
        Prompt with placeholders removed
    """
    return prompt.replace("{FILE_MONITORING_TOOL}", "").replace("{FILE_MONITORING_ROUTING}", "").replace("{FILE_MONITORING_CAPABILITY}", "")

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/multi_expert_agent/prompts/subagent_output_format.py
----
"""Subagent output format for Multi Expert Agent.

Defines concise output format instructions for sub-agents when called
by the orchestrator. This ensures consistent, synthesizable responses.
"""

SUBAGENT_CONCISE_OUTPUT_FORMAT = """
## Output Format Instructions (Sub-Agent Mode)

You are being called as a sub-agent by an orchestrator. Follow these rules:

1. **Be Concise**: Provide only the essential information needed to answer the query
2. **No Greetings**: Skip introductions and pleasantries
3. **Structured Response**: Use clear sections if multiple points
4. **Actionable**: Include specific commands, steps, or data when applicable
5. **Source Attribution**: Mention your data source briefly (e.g., "From documentation:", "From Jira:")

### Response Structure

```
[Direct answer to the query]

**Key Points:**
- Point 1
- Point 2

**[Commands/Steps if applicable]:**
```command
specific command or step
```

Source: [Your data source]
```

### Example Good Response

Query: "What is the sprint duration?"

Response:
```
Sprint duration is **3 weeks** at API2363.

**Sprint Rituals:**
- Sprint Planning: Start of sprint
- Daily Scrum: 15 min daily
- Sprint Review: End of sprint
- Retrospective: End of sprint

Source: Confluence - Fonctionnement Agile
```

### Example Bad Response (Too Verbose)

Query: "What is the sprint duration?"

Response:
```
Hello! I'd be happy to help you with information about sprint duration.

At API2363, we follow an Agile methodology based on the Scrum framework. The Scrum framework defines several time-boxed events, including the Sprint itself. A Sprint is a fixed period during which a potentially releasable product increment is created.

In our organization, we have adopted a sprint duration of 3 weeks, which has been found to be optimal for our teams...

[continues for several more paragraphs]
```
"""


# Shorter version for injection into sub-agent prompts
SUBAGENT_BRIEF_FORMAT = """
You are a sub-agent. Be concise:
- Direct answer first
- Key points as bullets
- Include commands/data when relevant
- No greetings or verbose explanations
- Mention your source briefly
"""

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/navigator_agent/graph.py
----
"""Graph explorer agent using LangGraph 1.0 middleware architecture.

This agent explores graph databases with pluggable behavior through middleware.
"""

import logging
from collections.abc import Callable
from typing import Any, Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware, ToolRetryMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from sta_agent_core.repositories import UKGRepository
from sta_agent_core.repositories.graph.topology_graph_repository import GraphBackend
from sta_agent_engine.agents.base.middlewares import (
    AgentModeMiddleware,
    DynamicModelMiddleware,
    DynamicToolMiddleware,
    GenerationRetryMiddleware,
    ModeAwareModelMiddleware,
    ReflectionMiddleware,
    StructuredOutputMiddleware,
    SystemPromptMiddleware,
)
from sta_agent_engine.agents.base.states.context import AgentModeContext, ModelConfig
from sta_agent_engine.agents.navigator_agent.prompts.prompt_builders import build_navigator_prompt
from sta_agent_engine.agents.navigator_agent.states import NavigatorAgentState
from sta_agent_engine.models.custom_chat_model import create_chat_model
from sta_agent_engine.utils.graph_init_helpers import safe_initialize_graph


load_dotenv()

logger = logging.getLogger(__name__)

# Initialize repository (singleton pattern)
repo = UKGRepository(backend=GraphBackend.NETWORKX)


def get_graph_explorer_graph(
    name: str = "navigator_agent",
    model: str | BaseChatModel | None = None,
    domain_context: Literal["aiops"] | None = "aiops",
    tools: list[BaseTool | Callable] | None = None,
    backend: GraphBackend = GraphBackend.NETWORKX,
    enable_mode_switching: bool = False,
    enable_reflection: bool = False,
    structured_output: type[BaseModel] | None = None,
    checkpointer: MemorySaver | None = None,
    middlewares: list | None = None,
    **middleware_config: Any,
) -> CompiledStateGraph[NavigatorAgentState, AgentModeContext]:
    """Create graph explorer agent with pluggable middleware.

    This factory function creates an agent that can explore graph databases
    with behavior controlled entirely through middleware composition.

    Args:
        name: Agent name for identification
        domain_context: Domain expertise to inject ("aiops", None for neutral)
        tools: Custom tools (if None, uses default repository tools)
        backend: Graph backend (NETWORKX or TIGERGRAPH)
        enable_mode_switching: Enable reasoning→generation mode switching
        enable_reflection: Enable reflection pattern middleware
        structured_output: Optional Pydantic schema for generation mode
        checkpointer: Optional checkpointer for persistence
        middlewares: Additional middlewares to append (e.g., OutputFormatMiddleware)
        **middleware_config: Additional middleware configuration
            - max_iterations: Max reasoning iterations before forcing generation (default: 10)
            - reflection_triggers: Custom reflection trigger functions (None = use middleware defaults)
                Examples:
                    - None: Use default heuristic triggers (after 3 tools OR high messages)
                    - {"always": lambda s: True}: Always trigger reflection
                    - {"custom": lambda s: custom_condition(s)}: Custom triggers
            - on_empty_tools: Behavior when no tools allowed ("warn", "error", "allow")

    Returns:
        Compiled agent graph

    Example:
        agent = get_graph_explorer_graph(
            domain_context="aiops",
            enable_mode_switching=True,
            enable_reflection=True,
            structured_output=KeyFindings,
        )

        result = agent.invoke(
            {"messages": [HumanMessage("What is AP12363 made of?")]},
            config={"configurable": {"thread_id": "test-1"}},
            context={
                "model": "openai/gpt-oss-120b",
                "reflection_model": "openai/o1-preview"
            }
        )
    """

    # Build default tools (repository-specific tools only)
    # Note: think_tool and switch_mode are injected by their respective middlewares
    default_tools: list[BaseTool | Callable] = [
        repo.adapter.execute_raw,
        repo.get_application_topology,
        repo.get_communication_flows,
        repo.analyze_dependency_impact,
    ]
    tools = tools or default_tools

    # ============================================
    # PHASE 1: Dynamic Middlewares (Runtime Context)
    # Run FIRST - establish baseline from user preferences
    # ============================================
    middleware = []

    # SystemPromptMiddleware runs FIRST to establish baseline prompt
    middleware.append(SystemPromptMiddleware())
    middleware.append(ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=100_000)]))

    middleware.append(ToolRetryMiddleware(max_retries=2, on_failure="continue"))

    middleware.append(DynamicModelMiddleware())
    middleware.append(DynamicToolMiddleware(on_empty=middleware_config.get("on_empty_tools", "warn")))

    # ============================================
    # PHASE 2: Mode-Specific Middlewares (Agent State)
    # Run AFTER dynamic - apply mode logic on top of baseline
    # ============================================

    if enable_reflection:
        # Allow custom triggers from config (None = use middleware defaults)
        triggers = middleware_config.get("reflection_triggers")
        middleware.append(ReflectionMiddleware(triggers=triggers))

    # if mode switching is enabled, it will automatically handle structured output
    if structured_output and not enable_mode_switching:
        middleware.append(StructuredOutputMiddleware(schema=structured_output))

    # Mode-aware model selection (overrides dynamic if mode-specific exists)
    middleware.append(ModeAwareModelMiddleware())

    # ============================================
    # PHASE 3: Mode Management
    # Run LAST - orchestrates mode transitions
    # ============================================

    if enable_mode_switching:
        middleware.append(
            AgentModeMiddleware(
                initial_mode="reasoning",
                enable_guardrails=True,
                max_reasoning_iterations=middleware_config.get("max_iterations", 10),
                output_model=structured_output,
            )
        )

    middleware.append(
        GenerationRetryMiddleware(
            max_retries=3,
            error_codes=["json_validate_failed"],
            error_code_pattern=".*tool.*",
            fallback_enabled=True,
            fallback_model=ModelConfig(model="mistral-medium-2508", provider="mistral"),
        )
    )
    # middleware.append(SummarizationMiddleware(model="openai/gpt-oss-120b"))

    # ============================================
    # PHASE 4: Custom Middlewares (User Provided)
    # Extend with any additional middlewares
    # ============================================
    if middlewares:
        middleware.extend(middlewares)

    # Build base system prompt (mode instructions injected by middleware)
    system_prompt = build_navigator_prompt(
        domain=domain_context,
        backend=backend,
        layer_to_node_types=repo.LAYER_TO_NODE_TYPES,
    )

    logger.info(
        f"Creating graph explorer agent: "
        f"domain={domain_context}, "
        f"mode_switching={enable_mode_switching}, "
        f"reflection={enable_reflection}, "
        f"structured_output={structured_output is not None}"
    )

    # Create agent
    return create_agent(  # type: ignore[reportUnknownReturnType]
        name=name,
        system_prompt=system_prompt,
        model=model or create_chat_model(model=model),
        middleware=middleware,
        tools=tools,
        checkpointer=checkpointer,
        response_format=structured_output,
        state_schema=NavigatorAgentState,
        context_schema=AgentModeContext,
    )


# Export default instance (may be None in BYOK mode without credentials)
aiops_agent = safe_initialize_graph(
    get_graph_explorer_graph,
    graph_name="aiops_agent",
    name="ops_graph_explorer_agent",
    backend=GraphBackend.NETWORKX,
    enable_mode_switching=True,
    enable_reflection=True,
)


if __name__ == "__main__":
    import asyncio

    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver

    async def main() -> None:
        """Test the refactored agent"""
        from pydantic import Field

        from sta_agent_core.config import setup_logger_func

        setup_logger_func()

        class KeyFindings(BaseModel):
            """Example structured output schema"""

            key_findings: list[str] = Field(description="Key findings from the graph exploration")
            limits: list[str] = Field(description="Limitations or gaps in the gathered information")
            quotes: list[str] = Field(description="Quotes/sources of the information")

        # Create agent with all features enabled
        agent = get_graph_explorer_graph(
            name="navigator_test_agent",
            domain_context="aiops",
            enable_mode_switching=True,
            enable_reflection=True,
            checkpointer=InMemorySaver(),
            # reflection_triggers=[ReflectionTrigger(name="always", trigger=lambda s: True, description="Always trigger reflection after each step")],
            # structured_output=KeyFindings,
        )

        gpt_model = "openai/gpt-oss-20b"

        # Test invocation
        async for chunk in agent.astream(
            input={"messages": [HumanMessage("What is topology general composition of AP1263?")]},
            config={"configurable": {"thread_id": "test-1"}, "recursion_limit": 50},
            context={"model": gpt_model},  # , "model_provider": "mistral"},  # , "model_provider": ""},
            stream_mode="values",
        ):
            print("=" * 50)
            print(f"Mode: {chunk.get('agent_mode', 'unknown')}")
            print(f"Messages: {len(chunk.get('messages', []))}")
            if chunk.get("messages"):
                print(f"Last message: {chunk['messages'][-1].pretty_print()}")
                pass

    asyncio.run(main())

-------

