# __Demo plan 24/10__
1. **__Agents Philosophy__** Context/System engineering + Evaluation driven developpment 

Base layers : LLM —> techno incroyable (Transformers - Generalization + Reasoning incroyable etc … ) 

—> Hallucination, context management etc …

Un Agent est aussi bon que le context qu’on lui injecte.
TROP ? ça le tue et il se perd , Pas assez ? il hallucine.

Context Engineering → Repose sur le system Autour ( RAG —> Global sense → Vector RAG Hybrid RAG + GraphRAG ( PathRAG) 

ReAct Agent + Workflow Agents + Multi Agent

EDD concept : Quick proto + Define evaluation dataset + Define metrics + iterate on Offline Metrics ( Experiments ) 

Metrics —> Rule Based ( e.g. on sait exactement ) 
Mais souvent c’est subjectif —> LLM as a Judge Sauf que ça incorpore encre plus de stochasticité dans le system ( + Souvent il faut itérer plusieurs fois pour aligner le jugement du judge avec Nous 

Best Practice LLM as A Judge : 

> [langgraph-agent-repo/docs/learning/evals/Langsmith_Evaluation_doc.md]

[langgraph-agent-repo/docs/learning/evals/llm_as_a_judge_prompting.md]  ) 

Mon rêve = 

Framework où les agents s’améliorent tout seul.

( Online execution → Flag non Efficient Calls + UerFeedback —> Annotation Queues → We define expected output + Add them to dataset —> Retrain Prompt OR Model ) 

Dspy —> auto train prompts 

PEFT ( ) —> Ultra Specialized LLMs on BNP ecosystem 

kg-adapter ( MODEL ) —> Model Finuted on BNP KG [https://aclanthology.org/2024.findings-acl.229.pdf](https://aclanthology.org/2024.findings-acl.229.pdf)

Llama2-7B 89.2 78.1 76.6 Llama2(7B)
1. **__Show case - package structure - Layers __**

—> Core = adapters to Various dbs ( graph, elastic etc ) 

Repositories = Business Layers 

providers = LLMs + Embbeders etc … 

—> agent Engine : Agents = Tools Prompts 

( Langgraph 1.0 —> MiddleWare structure )

React Agent 
+ 

Custom React Agent ( Upgraded to Supervisor ) 

Custom Workflow Agents ( Clarify ) 

Streaming Wrapper — Handle Multi Agent Streaming — 

—> frontEnd [DEMOS] ( Streamlit + Consumer for Streaming ) 

(Doc Streaming processor )[/langgraph-agent-repo/docs/agent_engine:streaming/Stream_Processor_Architecture.md[]](http://Architecture.md)
1. **__Demo of agents W/ Streamlit + Langsmith Platfrom ( Tracing / evals etc ) __**
- Topology Agent 
- Supervisor 
- Clarification Agent 

Demo of TOPOLOGY Evaluation framework