**Objet : RE: Détail des évaluateurs et métriques**

Bonjour,

Oui, bonne remarque. Il faut effectivement distinguer trois notions :

```text
Evaluator              Metric                 Threshold
méthode de mesure  →   score produit      →   critère de passage
```

Par exemple, un **Completeness Evaluator** compare la réponse aux `key facts` du Golden Dataset. Il produit une metric `completeness_recall`, ensuite comparée à un threshold, par exemple ≥ 75 %.

Les principales méthodes envisagées sont :

- **Deterministic Evaluators** : vérification du format, de l’output schema, des sources citées, des tools appelés et du routing. Ils sont reproductibles et peuvent rapidement devenir bloquants dans la CI/CD.
- **LLM-as-Judge Evaluators** : évaluation sémantique de la Completeness, de la Faithfulness, des hallucinations et du respect des guidelines métier. Étant probabilistes, ils sont d’abord utilisés en reporting, puis transformés en Quality Gates après calibration.
- **Trajectory Evaluators** : comparaison entre les tools/routes attendus et ceux réellement utilisés par l’agent.
- **RAG Evaluators** : mesure de la qualité du retrieval avec Precision@K, Recall@K, MRR, puis de la génération avec Completeness et Faithfulness.

Quelques exemples :

| Evaluator | Méthode | Metric |
|---|---|---|
| Completeness | Présence des `key facts`, via LLM-as-Judge | `completeness_recall` |
| Faithfulness | Comparaison des affirmations aux sources | `hallucination_free` |
| Guideline Adherence | Vérification de chaque règle métier attendue | guideline pass rate |
| Tool Routing | Comparaison aux `reference_tool_calls` | tool recall/precision |
| Output Schema | Validation JSON/Pydantic | pass/fail |
| RAG Retrieval | Comparaison des documents attendus et retrouvés | Precision@K, Recall@K, MRR |

Certaines metrics sont ensuite dérivées. Par exemple, la Confusion Matrix combine Completeness et Faithfulness :

```text
Complete + Faithful     → IDEAL
Complete + Unfaithful   → RISKY
Incomplete + Faithful   → INCOMPLETE
Incomplete + Unfaithful → FAILURE
```

Côté outillage, l’approche proposée s’appuie sur :

- `sta-eval` pour exécuter les Eval Suites ;
- LangSmith pour les Golden Datasets, traces, experiments, metrics et comparisons ;
- pytest/CI/CD pour les tests déterministes et les Quality Gates ;
- l’Annotation Queue pour la Human Review et l’enrichissement continu.

Je vais intégrer ces précisions dans la section dédiée aux Evaluators, avec pour chacun la méthode, la metric produite et le threshold associé.

Cordialement,