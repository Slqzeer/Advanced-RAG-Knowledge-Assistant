# Advanced RAG Knowledge Assistant

## Un moteur capable d'interroger une base documentaire technique avec citations et sources.

Par exemple, tu pourrais utiliser comme corpus :

 - documentation Python 
 - documentation FastAPI 
 - documentation Docker 
 - documentation Kubernetes 
 - documentation LangChain 
 - articles techniques 
 - PDF techniques

L'utilisateur pourrait poser :

`Comment fonctionne l'injection de dépendances dans FastAPI ?`

## Stack technique

Je partirais sur cette stack.

| Composant	| Technologie |
| ------------- | ----------------- |
| Langage Python | 3.12+ |
| API | FastAPI |
|Orchestration RAG | LangChain |
|Vector DB | Qdrant |
|Embeddings | OpenAI embeddings ou BGE | 
|LLM | OpenAI / modèle local interchangeable |
|Recherche lexicale	| BM25 |
|Reranking	| CrossEncoder / BGE Reranker / Cohere Rerank |
|Évaluation	| RAGAS + métriques maison |
|Validation	| Pydantic |
|Tests | Pytest |
|Observabilité | LangSmith ou OpenTelemetry |
|Conteneurisation | Docker / Docker Compose |
|CI	| GitHub Actions |

## Architecture finale

À terme, ton pipeline ressemblera à ceci :

```text
                        USER
                          |
                          v
                    User question
                          |
                          v
                 Query normalization
                          |
                          v
                    Query rewriting
                          |
              +-----------+-----------+
              |                       |
              v                       v
        Dense Retrieval          BM25 Retrieval
              |                       |
              +-----------+-----------+
                          |
                          v
                     RRF Fusion
                          |
                     Top 30 chunks
                          |
                          v
                      Reranker
                          |
                     Top 5 chunks
                          |
                          v
                    Context Builder
                          |
                          v
                       Prompt
                          |
                          v
                         LLM
                          |
                          v
                 Answer + citations
                          |
                          v
                     Evaluation
```

## Phase 0 — Préparer le projet

Avant de faire du RAG, crée une architecture propre.

```text
advanced-rag/
│
├── app/
│   ├── api/
│   ├── core/
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   ├── evaluation/
│   └── models/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── tests/
│
├── scripts/
│
├── notebooks/
│
├── docker/
│
├── .env.example
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Phase 1 — RAG minimal

Première version :

```text
    Documents
        ↓
    Chunking
        ↓
    Embeddings
        ↓
    Qdrant
```

Puis :

    Question
    ↓
    Embedding
    ↓
    Similarity Search
    ↓
    Top K chunks
    ↓
    LLM
    ↓
    Réponse

### À implémenter

    - charger des documents 
    - nettoyer le texte 
    - découper les documents
    - créer les embeddings 
    - les stocker dans Qdrant 
    - vectoriser la question 
    - faire une similarity search 
    - récupérer top_k=5 
    - envoyer ces chunks au LLM 
    - retourner la réponse

Ici, aucune optimisation.

## Phase 2 — Comprendre et améliorer le chunking

Maintenant, tu expérimentes.

### Version initiale :

    - chunk_size = 1000
    - overlap = 200

### Puis comparaison :

    - Fixed-size chunking
    - Recursive chunking
    - Sentence chunking
    - Semantic chunking
    - Parent-child chunking

### Tu mesures l'effet sur le retrieval.

Exemple :

    Question
        "Comment FastAPI gère les Depends ?"

    Chunking A
        Recall@5 = 0.65

    Chunking B
        Recall@5 = 0.82

À partir de cette étape, tu commences à traiter ton RAG comme un système de recherche, pas simplement comme un LLM.

## Phase 3 — Metadata filtering

Chaque chunk doit posséder des métadonnées.

```json
{
  "document_id": "fastapi_001",
  "title": "Dependencies",
  "source": "fastapi",
  "url": "...",
  "section": "Dependency Injection",
  "language": "en",
  "chunk_id": 37
}
```

## Phase 4 — Evaluation du retrieval

C'est probablement l'une des étapes les plus importantes du projet.

Tu crées un dataset :
```json
{
  "question": "What is dependency injection?",
  "relevant_documents": [
    "doc_123",
    "doc_145"
  ]
}
```

Puis tu calcules :

```text
Recall@K
Precision@K
MRR
Hit Rate
NDCG
```

Exemple :

Baseline Vector Search

|-----------|------|
| Recall@5 | 0.72 |
|Recall@10 | 0.81 |
|MRR | 0.68 | 
|HitRate@5 | 0.76 |

À partir de maintenant, chaque amélioration doit être justifiée par les résultats.

                Recall@5

|--------------|------|
|Naive RAG | 0.72|
|Better chunking | 0.78|
|Hybrid Search | 0.84|
|Reranking | 0.91|

C'est exactement le genre de graphique intéressant dans ton README.

Les guides actuels sur les architectures RAG de production insistent d'ailleurs fortement sur l'évaluation du retrieval plutôt que sur une simple observation subjective de la qualité des réponses.

## Phase 5 — Hybrid Search

Premier gros changement architectural.

La recherche vectorielle est bonne pour :
```text
    "how to protect an API"
```
qui peut retrouver :
```text
    "API authentication and authorization"
```
Même si les mots ne sont pas identiques.

Mais pour :
```text
    HTTPException 422
```
une recherche exacte est extrêmement utile.

Tu combines donc :
```text
    Dense Vector Search
            +
    BM25
```

Puis fusion des résultats :
```text
    Dense results ──┐
                    ├──> RRF ──> Top documents
    BM25 results ───┘
```

### RRF = Reciprocal Rank Fusion.

Architecture :
```text
query
  |
  +---------> embedding ---> vector search ----+
  |                                            |
  +---------> BM25 --------> lexical search ---+
                                               |
                                               v
                                           RRF Fusion
                                               |
                                               v
                                             Top K
```
Cette architecture hybride est particulièrement utile lorsque le corpus contient noms propres, références, termes techniques, identifiants ou morceaux de code.

## Phase 6 — Reranking

Maintenant ton retriever récupère par exemple :

```text
    Top 30 documents
```

Mais tu ne les donnes pas directement au LLM.

Tu ajoutes un reranker.

```text
    Retrieval
    ↓
    30 chunks
    ↓
    Cross Encoder
    ↓
    score(query, chunk)
    ↓
    Top 5
```

Par exemple :

    retrieval score

```text
    doc A → 0.81
    doc B → 0.79
    doc C → 0.77
```

Après reranking :

```text
    doc B → 0.96
    doc C → 0.88
    doc A → 0.53
```

Pipeline :

```text
    Hybrid Search
        ↓
    Top 30
        ↓
    Reranker
        ↓
    Top 5
        ↓
    LLM
```
C'est une amélioration importante car le premier retriever cherche surtout un bon recall, puis le reranker améliore la précision des candidats retenus. C'est également le principe recommandé dans l'architecture hybrid-search + reranking documentée par Qdrant.

## Phase 7 — Query rewriting
Maintenant tu travailles sur la question.

Utilisateur :
```text
    "et pour docker ?"
```

Cette requête seule ne veut presque rien dire.

Mais la conversation précédente contient :
```text
    Comment limiter la mémoire d'un container Kubernetes ?
```

Le système transforme donc :

```text
    "et pour docker ?"
```

en :

```text
    "How can memory limits be configured for Docker containers?"
```

Pipeline :

```text
    Conversation
        ↓
    Query Rewriter
        ↓
    Standalone Query
        ↓
    Retriever
```

Tu peux demander au LLM de produire une sortie structurée :

```json
    {
    "original_query": "et pour docker ?",
    "rewritten_query":
        "How to configure memory limits for Docker containers?"
    }
```

## Phase 8 — Multi-query retrieval
Une question peut être formulée de plusieurs façons.

Question :

```text
    How does FastAPI authentication work?
```

Le LLM génère :

```text
    FastAPI security authentication
    FastAPI OAuth2 authentication
    FastAPI authentication dependencies
```

Tu recherches avec les trois requêtes.

```text
                 Query
                   |
             Query Expansion
           /       |       \
          /        |        \
       query1    query2    query3
          \        |        /
           \       |       /
             Retrieval
                 |
              Fusion
```

Cela améliore parfois le recall lorsque le vocabulaire utilisé par l'utilisateur diffère fortement de celui du corpus. C'est toutefois plus coûteux, donc il vaut mieux l'ajouter après avoir mesuré les étapes précédentes.

## Phase 9 — Contextual compression

Tu peux récupérer un chunk de 1 000 tokens dont seulement 150 sont pertinents.

Au lieu d'envoyer tout :

```text
    chunk
    ↓
    LLM / extractor
    ↓
    relevant sentences
```

Puis seulement le texte réellement intéressant entre dans le contexte final.

Tu compares :

```text
    tokens utilisés
    latence
    coût
    qualité
```

## Phase 10 — Citations et grounded generation

Le LLM ne doit plus simplement répondre.

Prompt système :

```text
    Answer only using the provided context.
```

If the answer cannot be derived from the context,
say that you do not know.

Every factual statement must cite its source.

Résultat :

```text
    FastAPI uses dependency injection through the
    Depends mechanism [1].
```

Dependencies can themselves declare other
dependencies [2].

Sources:
[1] FastAPI - Dependencies
[2] FastAPI - Sub-dependencies

Ton API pourrait retourner :
```json
{
  "answer": "...",
  "sources": [
    {
      "title": "FastAPI Dependencies",
      "url": "...",
      "chunk_id": "..."
    }
  ]
}
```

## Phase 11 — Évaluation complète du RAG

Cette fois tu ne mesures plus seulement la recherche.

Tu évalues aussi la génération.

Par exemple :

```text
Retrieval metrics
├── Recall@K
├── Precision@K
├── MRR
└── NDCG

Generation metrics
├── Faithfulness
├── Answer relevance
├── Context precision
├── Context recall
└── Citation correctness
```

Avec RAGAS + tes propres métriques.

## Phase 12 — Guardrails

Tu ajoutes des règles.

Par exemple :

```text
    Pas assez de contexte
            ↓
    "I don't have enough information"
```

et :

```text
    retrieval_score < threshold
            ↓
    refuse_to_answer
```

Tu peux également détecter :

```text
    prompt injection
    questions hors domaine
    documents malveillants
    instructions présentes dans les documents
```

## Phase 13 — Cache

Ensuite optimisation.

 - Embedding cache
 - Retrieval cache
 - LLM response cache

Exemple :

```text
    query
    ↓
    hash
    ↓
    Redis
    ↓
    cache hit?
```

## Phase 14 — API professionnelle

FastAPI :

```text
POST /query
POST /documents
DELETE /documents/{id}

GET /documents
GET /health
GET /metrics
```

Exemple :

```text
POST /query
{
  "question": "How does FastAPI dependency injection work?"
}
```

réponse :

```json
{
  "answer": "...",
  "sources": [...],
  "retrieval": {
    "retrieved": 30,
    "reranked": 5
  },
  "latency_ms": 642
}
```

## Phase 15 — Observabilité

Pour chaque requête, enregistre :

```text
    - query
    - rewritten query

    - documents retrieved
    - retrieval scores

    - reranking scores

    - prompt

    - LLM response

    - tokens

    - latency

    - cost
```

Cela te permettra ensuite d'avoir un véritable dashboard.

## Phase 16 — Dockerisation

Ton projet :

```text
    Docker Compose

    ├── rag-api
    ├── qdrant
    ├── redis
    └── optional frontend
```

Lancement :

```text
    docker compose up
```

Ton projet devient reproductible pour quelqu'un qui arrive sur ton GitHub.

## Phase 17 — Tests et CI/CD

Tests :

```text
    unit tests
    ├── chunking
    ├── retrieval
    ├── reranking
    └── query rewriting

    integration tests
    ├── qdrant
    ├── embeddings
    └── complete pipeline

    RAG evaluation
    └── evaluation dataset
```

GitHub Actions :

```text
    push
    ↓
    lint
    ↓
    tests
    ↓
    RAG regression tests
    ↓
    Docker build
```

## Les versions de ton projet

Ton historique Git devrait quasiment raconter cette évolution.

```text
    v0.1
    Simple vector search

            ↓

    v0.2
    Basic RAG

            ↓

    v0.3
    Metadata + citations

            ↓

    v0.4
    Retrieval evaluation

            ↓

    v0.5
    Hybrid Search
    Dense + BM25

            ↓

    v0.6
    RRF Fusion

            ↓

    v0.7
    Cross-Encoder Reranking

            ↓

    v0.8
    Query Rewriting

            ↓

    v0.9
    Multi Query Retrieval

            ↓

    v1.0
    Production RAG
```
Ensuite :

```text
    v1.1
    Semantic chunking

    v1.2
    Context compression

    v1.3
    RAGAS evaluation

    v1.4
    Caching

    v1.5
    Observability

    v1.6
    Guardrails
```

## Un point important : ne commence pas avec LangChain partout

Je te conseille quelque chose de particulier pour apprendre correctement.

Au début, implémente toi-même :

```python
    embed(query)

    qdrant.search(...)

    build_context(...)

    generate_answer(...)
```

Au lieu de faire immédiatement :

```python
chain = magic_rag_chain(...)
```

Parce que ton objectif est de comprendre :

```text
question
→ embedding
→ similarity
→ retrieval
→ context
→ prompt
→ generation
```

Ensuite utilise davantage LangChain lorsqu'il devient réellement utile pour composer :

```text
retrievers
rerankers
query transformations
tracing
```

Ainsi, en entretien, tu seras capable d'expliquer ce qui se passe sous le framework.

## Le coeur pédagogique de ton projet

Le projet ne doit pas devenir :

```text
    "J'ai ajouté 25 techniques de RAG."
```

Il doit devenir :

```text
    "J'avais ce problème de retrieval. J'ai mesuré cette métrique. J'ai ajouté cette technique. Voilà son effet."
```

Par exemple :

| |Recall@5 | MRR | Latency |
|---------------|---------|---------|---------|
|Vector search | 0.71 | 0.67 | 80 ms|
|Better chunking | 0.76 | 0.71 | 83 ms|
|Hybrid search | 0.84 | 0.78 | 105 ms|
|+ Reranker | 0.91 | 0.87 | 180 ms|
|+ Multi-query | 0.93 | 0.89 | 460 ms|

Et là tu peux expliquer :

```text
Multi-query augmente légèrement le recall mais multiplie la latence ; je ne l'active donc que pour certaines requêtes.
```

Ça ressemble beaucoup plus à une réflexion d'ingénierie qu'à un simple tutoriel RAG.

## Ordre exact que je te recommande
```text
01. Python project architecture
02. Document ingestion
03. Cleaning
04. Basic chunking
05. Embeddings
06. Qdrant
07. Similarity search
08. Basic RAG
09. Citations
10. Retrieval evaluation dataset
11. Recall / Precision / MRR
12. Chunking experiments
13. Metadata filtering
14. BM25
15. Hybrid search
16. RRF
17. Reranking
18. Query rewriting
19. Multi-query retrieval
20. Context compression
21. RAGAS evaluation
22. Guardrails
23. Caching
24. Observability
25. FastAPI
26. Docker
27. Tests
28. CI/CD
29. Benchmark dashboard
30. Professional README
```

À l'étape 10, on aura déjà un projet présentable. Les étapes 11–20 vont en faire un projet RAG avancé. Les étapes 21–30 vont en faire un projet qui ressemble davantage à un système de production.