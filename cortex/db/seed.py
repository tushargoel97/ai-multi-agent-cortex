"""Seed the database with a general-knowledge corpus.

Usage:
    docker compose up -d
    python -m cortex.db.seed                # tables + data
    python -m cortex.db.seed --force        # drop & re-seed from scratch
    OPENAI_API_KEY=sk-... python -m cortex.db.seed --embeddings   # + embeddings
    OPENAI_API_KEY=sk-... python -m cortex.db.seed --force --embeddings  # full reset
"""

from __future__ import annotations

import logging
import uuid

from cortex.db.engine import engine, get_session
from cortex.db.models import Base, KnowledgeArticle, LLMModel, LLMProvider, ProviderKind

logger = logging.getLogger(__name__)


# ── Knowledge Base ───────────────────────────────────────────────────────────
# Small curated corpus for the local pgvector knowledge base.
# Edit this file to expand the assistant's domain knowledge.

ARTICLES: list[dict[str, str]] = [
    {
        "title": "What is a Large Language Model?",
        "category": "ai",
        "content": (
            "A large language model (LLM) is a neural network, typically based on "
            "the transformer architecture, trained on very large text corpora to "
            "predict the next token given previous tokens. Modern LLMs such as "
            "GPT-4, Claude, Gemini, and Llama have parameters numbering in the "
            "billions to trillions and exhibit emergent capabilities including "
            "in-context learning, instruction following, code generation, and "
            "multi-step reasoning. LLMs are typically pre-trained with a "
            "self-supervised next-token objective and then aligned with human "
            "preferences through supervised fine-tuning (SFT) and reinforcement "
            "learning from human feedback (RLHF) or direct preference optimisation "
            "(DPO)."
        ),
    },
    {
        "title": "Multi-Agent Systems",
        "category": "ai",
        "content": (
            "A multi-agent system is a software architecture in which several "
            "autonomous LLM-based agents, each with its own role, prompt, and "
            "tool surface, collaborate to solve a task. Common patterns include "
            "the supervisor pattern (a planner routes to specialists), the swarm "
            "pattern (peer agents hand off control), and the debate pattern "
            "(agents argue and a judge picks the winner). Multi-agent designs "
            "trade higher token cost and latency for improvements in tool "
            "specialisation, separation of concerns, and reliability under "
            "complex workflows."
        ),
    },
    {
        "title": "Retrieval-Augmented Generation (RAG)",
        "category": "ai",
        "content": (
            "Retrieval-Augmented Generation (RAG) is a technique that grounds "
            "an LLM in external knowledge by retrieving relevant documents at "
            "query time and inserting them into the prompt. A typical RAG "
            "pipeline embeds documents into a vector database (e.g. pgvector, "
            "Qdrant, Weaviate), embeds the user query at runtime, performs a "
            "nearest-neighbour search, and passes the top-k chunks plus the "
            "user query to the model. Advanced variants include hybrid search "
            "(dense + BM25), reranking with a cross-encoder, query expansion, "
            "and recursive retrieval for multi-hop questions. RAG improves "
            "factuality, enables fresh data without retraining, and is the "
            "foundation of most production knowledge assistants."
        ),
    },
    {
        "title": "Prompt Caching",
        "category": "ai",
        "content": (
            "Prompt caching is an optimisation where the key-value (KV) "
            "attention states for a prompt prefix are stored after the first "
            "request and reused on subsequent requests that share the same "
            "prefix. This reduces both time-to-first-token (TTFT) and cost "
            "for the cached tokens, often by 50–90%. To benefit from prompt "
            "caching, place stable content (system prompts, tool schemas, "
            "few-shot examples, retrieved context that does not change "
            "between turns) at the beginning of the prompt and place the "
            "variable content (the latest user message) at the end. Most "
            "frontier providers (OpenAI, Anthropic, Google) cache prefixes "
            "of around 1024 tokens or more, with cache lifetimes of 5–60 "
            "minutes of inactivity."
        ),
    },
    {
        "title": "LLM Evaluation",
        "category": "ai",
        "content": (
            "LLM evaluation is the practice of measuring an agent's quality "
            "across deterministic and probabilistic dimensions. A robust eval "
            "suite includes routing accuracy, tool-use correctness (the agent "
            "calls the right tool with the right arguments), response quality "
            "(faithfulness, answer relevance, toxicity), and end-to-end "
            "regression tests. Frameworks such as DeepEval, RAGAS, LangSmith, "
            "Promptfoo, and Braintrust provide LLM-as-Judge metrics, golden "
            "datasets, and CI integrations. Good eval suites are run on every "
            "pull request and provide pass/fail signals like a unit test."
        ),
    },
    {
        "title": "Observability for LLM Applications",
        "category": "ai",
        "content": (
            "LLM observability captures every model call as a span in a "
            "distributed trace, recording the prompt, model output, latency, "
            "token counts, cost, tool invocations, and metadata such as user "
            "and session IDs. Tools such as Langfuse, LangSmith, Arize "
            "Phoenix, and Helicone consume OpenTelemetry traces and provide "
            "cost dashboards, drift detection, and replay debugging. "
            "Observability is the prerequisite for evals, A/B tests, and "
            "production incident response in agentic systems."
        ),
    },
    {
        "title": "Guardrails for Agents",
        "category": "ai",
        "content": (
            "Guardrails are defensive layers that constrain an agent's "
            "behaviour. Common categories include: input filters (jailbreak "
            "and prompt-injection detection), output filters (toxicity, PII "
            "redaction, schema validation), tool-call guards (allowlist of "
            "callable tools, argument validation, hard limits on side "
            "effects), and human-in-the-loop interrupts (the agent pauses "
            "and asks for explicit approval before high-risk actions). "
            "Guardrails are typically implemented as middleware that wraps "
            "every model call and tool call."
        ),
    },
    {
        "title": "LangGraph",
        "category": "tooling",
        "content": (
            "LangGraph is an orchestration framework for building stateful, "
            "multi-step LLM applications as graphs of nodes and edges. Nodes "
            "are Python functions (or compiled agents) and edges describe "
            "the control flow between them. LangGraph supports conditional "
            "routing, cycles, persistent checkpoints (via PostgreSQL or "
            "in-memory stores), human-in-the-loop interrupts, and time "
            "travel, replaying a graph from any prior state. It is the "
            "standard runtime in the LangChain ecosystem for production "
            "agent workflows."
        ),
    },
    {
        "title": "Vector Databases and pgvector",
        "category": "tooling",
        "content": (
            "A vector database stores high-dimensional embeddings and "
            "supports efficient nearest-neighbour search via algorithms "
            "such as HNSW or IVF. pgvector is a PostgreSQL extension that "
            "adds a `vector` column type and indexing operators (cosine, "
            "L2, inner product) directly inside Postgres, enabling RAG "
            "applications to reuse their existing relational database "
            "rather than running a separate vector store. For applications "
            "with up to a few million vectors and modest QPS, pgvector is "
            "usually sufficient and operationally simpler than a dedicated "
            "vector database."
        ),
    },
    {
        "title": "Python, Quick Reference",
        "category": "programming",
        "content": (
            "Python is a high-level, dynamically-typed, garbage-collected "
            "programming language created by Guido van Rossum in 1991. It "
            "emphasises readability through significant indentation. Key "
            "features include first-class functions, comprehensions, "
            "generators, async/await, structural pattern matching, type "
            "hints (PEP 484), and a large standard library. The reference "
            "implementation is CPython; alternative interpreters include "
            "PyPy (JIT-compiled), MicroPython (embedded), and Pyodide "
            "(WebAssembly). Python is the dominant language for data "
            "science, machine learning, and scripting."
        ),
    },
]


# ── Seeding ──────────────────────────────────────────────────────────────────


def seed_database(force: bool = False) -> None:
    """Create all tables and insert the knowledge-base corpus."""
    if force:
        Base.metadata.drop_all(engine)
        logger.info("Existing tables dropped (--force).")

    Base.metadata.create_all(engine)
    logger.info("Database tables created.")

    with get_session() as session:
        if not force and session.query(KnowledgeArticle).first():
            logger.info("Database already seeded, skipping. Use --force to re-seed.")
            return

        for a in ARTICLES:
            session.add(KnowledgeArticle(id=uuid.uuid4(), **a))

    logger.info("Seed data inserted (%d articles).", len(ARTICLES))

    seed_llm_registry(force=force)


# ── LLM Provider / Model Registry ────────────────────────────────────────────


_DEFAULT_PROVIDERS: list[dict] = [
    {
        "name": "Self-Hosted (Local)",
        "kind": ProviderKind.LOCAL,
        "base_url": "http://ai:8100/v1",
        "models": [],  # filled in by user via admin → Local Models
    },
    {
        "name": "OpenAI",
        "kind": ProviderKind.OPENAI,
        "models": [
            ("gpt-4o-mini", "GPT-4o mini", True),
            ("gpt-4o", "GPT-4o", False),
        ],
    },
    {
        "name": "Anthropic",
        "kind": ProviderKind.ANTHROPIC,
        "models": [
            ("claude-3-5-sonnet-latest", "Claude 3.5 Sonnet", False),
            ("claude-3-5-haiku-latest", "Claude 3.5 Haiku", False),
        ],
    },
    {
        "name": "Google",
        "kind": ProviderKind.GOOGLE,
        "models": [
            ("gemini-1.5-flash", "Gemini 1.5 Flash", False),
            ("gemini-1.5-pro", "Gemini 1.5 Pro", False),
        ],
    },
]


def seed_llm_registry(force: bool = False) -> None:
    """Insert preset providers + models so the chat UI has a populated dropdown."""
    with get_session() as session:
        if not force and session.query(LLMProvider).first():
            logger.info("LLM registry already seeded, skipping.")
            return

        for prov_cfg in _DEFAULT_PROVIDERS:
            provider = LLMProvider(
                id=uuid.uuid4(),
                name=prov_cfg["name"],
                kind=prov_cfg["kind"].value,
                api_key="",
                base_url=prov_cfg.get("base_url"),
                enabled=True,
            )
            session.add(provider)
            session.flush()
            for model_id, display, is_default in prov_cfg["models"]:
                session.add(
                    LLMModel(
                        id=uuid.uuid4(),
                        provider_id=provider.id,
                        model_id=model_id,
                        display_name=display,
                        enabled=True,
                        is_default=is_default,
                    )
                )

    logger.info("LLM registry seeded (%d providers).", len(_DEFAULT_PROVIDERS))


def seed_embeddings() -> None:
    """Generate and store embeddings for knowledge articles.

    Requires OPENAI_API_KEY to be set in the environment.
    """
    from cortex.model_client import get_embedding_client

    embeddings = get_embedding_client()

    with get_session() as session:
        articles = (
            session.query(KnowledgeArticle)
            .filter(KnowledgeArticle.embedding.is_(None))
            .all()
        )
        if not articles:
            logger.info("All articles already have embeddings, skipping.")
            return

        texts = [f"{a.title}\n\n{a.content}" for a in articles]
        vectors = embeddings.embed_documents(texts)

        for article, vector in zip(articles, vectors):
            article.embedding = vector

    logger.info("Embeddings generated for %d articles.", len(articles))


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    force = "--force" in sys.argv
    seed_database(force=force)

    if "--embeddings" in sys.argv:
        seed_embeddings()
