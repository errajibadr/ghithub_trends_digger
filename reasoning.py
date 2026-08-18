Bonjour,

Voici une proposition de fonctionnement pour les différentes étapes.

### Étape 1 — Construire le Golden Dataset

Le **Golden Dataset** est un ensemble versionné de questions représentatives, avec les **expected outputs** et **expected behaviors** associés.

```text
Business Question
       +
Expected Output
       +
Sources / Guidelines / Expected Tools
       =
Golden Dataset
```

- **Content Owner :** Product Owner/SPOC de la franchise, avec les **Subject Matter Experts**.
- **Methodology & Tooling Owner :** équipe IA.
- **Validation de la représentativité :** conjointe Métier + IA, à partir d’une **coverage matrix** couvrant les happy paths, cas critiques, cas complexes, edge cases, demandes out-of-scope et cas adversariaux.
- **Taille indicative :** 10–20 exemples au prototype, 20–40 en development, 50+ avant production.
- **Maintenance :** enrichissement à chaque défaut significatif, monthly review des nouveaux cas et full quarterly review.

### Étape 2 — Développer et évaluer l’agent

Un socle commun d’**evaluators** peut être mutualisé entre les franchises :

- **Completeness** ;
- **Faithfulness / Hallucination detection** ;
- **Guideline adherence** ;
- **Refusal correctness** ;
- respect de l’**output schema** ;
- **Tool trajectory**, routing et efficiency ;
- pour le RAG : **Retrieval Precision/Recall**, context relevance et source quality.

La combinaison de **Completeness** et **Faithfulness** permet de classer les réponses :

```text
                       Faithful           Unfaithful
                   +----------------+----------------+
Complete           | IDEAL          | RISKY          |
                   | complete and   | fluent but     |
                   | correct        | potentially    |
                   |                | misleading     |
                   +----------------+----------------+
Incomplete         | INCOMPLETE     | FAILURE        |
                   | correct but    | incomplete and |
                   | partial        | incorrect      |
                   +----------------+----------------+
```

**Initial quality gates** proposés avant l’Integration Testing :

- average Completeness ≥ 75 % ;
- Hallucination-free rate ≥ 95 %, sans critical hallucination ;
- `RISKY rate` < 5 % ;
- Routing/Tool accuracy ≥ 90 % ;
- Security, permissions et critical output rules respectés à 100 % ;
- aucune régression significative par rapport à la validated baseline.

Ces thresholds devront être calibrés après plusieurs evaluation runs et adaptés à la criticité de chaque agent.

### Étape 3 — Integration & Regression Testing

Il faut combiner deux approches complémentaires :

```text
Standard CI/CD                        AI Evaluation / MLOps
----------------                      ----------------------
Unit Tests                            Completeness
Integration Tests                     Faithfulness
API & Schema Tests                    Hallucination Detection
Routing & Tool Tests                  Response Quality
Security Tests                        Baseline Comparison
Regression Tests                      Statistical Analysis
         |                                      |
         +------------ Quality Report ----------+
```

**Standard CI/CD**

Les tests déterministes sont automatisés et blocking : Unit Tests, Integration Tests, API contracts, schemas, routing, error handling, security et technical regression tests.

**AI Evaluation / MLOps**

- Dans un premier temps, les probabilistic evaluations génèrent un **non-blocking quality report** dans la CI/CD.
- Les résultats sont comparés à la **validated baseline** et à la previous version.
- Lorsque les metrics et thresholds deviennent suffisamment stables, certaines evaluations sont transformées en **release gates**.
- À terme, la release est bloquée si les performances passent sous les agreed thresholds ou sont significativement inférieures à la previous version.

Les developers maintiennent les technical tests, QA pilote l’Integration Testing et l’équipe IA analyse les probabilistic evaluations.

### Étape 4 — User Acceptance Testing

Les **UAT** sont réalisés par un panel de SPOC et d’utilisateurs représentatifs : Product Owner, Subject Matter Experts et operational users.

**Proposed Acceptance Criteria :**

- 100 % des critical scenarios validés ;
- UAT pass rate ≥ 90 % ;
- aucune critical hallucination, data leakage ou permission issue ;
- offline evaluation thresholds respectés ;
- aucun blocking defect ouvert ;
- Monitoring, Support et Rollback Plan prêts.

Le **business sign-off** est donné par le Product Owner ou Business Sponsor. Le **technical sign-off** est donné par le Release Owner. L’Operating Model confirme ensuite la production readiness.

### Étape 5 — Production & Continuous Improvement

L’Operating Model doit organiser le monitoring, le user feedback, l’annotation et l’enrichissement continu du Golden Dataset.

```text
       Production Agent
              |
              v
    Traces + User Feedback
              |
              v
       Annotation Queue
   (weak or suspicious cases)
              |
              v
      Human/SME Review
              |
       +------+------+
       |             |
    Valid case    Correction
       |             |
       +------+------+
              |
              v
   Golden Dataset Enrichment
              |
              v
  New Evaluation / Agent Version
              |
              +---------> Production
```

Les cas prioritaires à envoyer dans l’**Annotation Queue** sont :

- negative user feedback ;
- `RISKY` ou `INCOMPLETE` responses ;
- hallucinations ;
- incorrect Tool Trajectories ;
- latency anomalies ;
- random production samples.

Après human review et correction, ces cas sont ajoutés au Golden Dataset et deviennent de futurs Regression Tests.

```text
Business       → Expected Truth & Business Acceptance
AI Team        → Evaluation Methodology & Metrics
Engineering/QA → Software Quality & CI/CD
Operating Model → Production Quality & Continuous Improvement
```

Cordialement,

---

### Annexe — Sources et définitions

1. **Golden Dataset**

La définition, les tailles indicatives, la distribution et le Continuous Improvement Loop viennent de [Golden Dataset]().

2. **IDEAL / RISKY / INCOMPLETE / FAILURE**

La **Confusion Matrix** est définie dans [Evaluation Philosophy]() :

- `IDEAL` : Complete + Faithful ;
- `RISKY` : Complete + Unfaithful ;
- `INCOMPLETE` : Incomplete + Faithful ;
- `FAILURE` : Incomplete + Unfaithful.

Le seuil de 75 % de Completeness est celui utilisé par défaut par le framework. Le `RISKY rate < 5 %` est une proposition de quality gate fondée sur le risque particulier des réponses fluides mais erronées.

3. **Standard CI/CD vs AI Evaluation/MLOps**

La séparation entre deterministic tests, probabilistic evaluations et progressive threshold gating vient de [Testing vs Evaluation]().

4. **Common Evaluators**

Les metrics de Completeness, Faithfulness, Tool Coverage et Tool Efficiency sont décrites dans [Evaluators Guide]().

5. **Annotation Queue & Continuous Improvement**

Le passage Production Traces → Annotation Queue → Human Review → Golden Dataset est documenté dans [Annotation Workflow]() et [Online Evaluation]().