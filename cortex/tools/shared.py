"""Shared tools available to multiple agents."""

import json

from pydantic import BaseModel, Field

from cortex.db.engine import get_session
from cortex.db.models import KnowledgeArticle
from cortex.model_client import get_embedding_client
from cortex.tools.registry import register_tool


class SearchKnowledgeBaseInput(BaseModel):
    """Input for searching the local knowledge base."""

    query: str = Field(
        description="Natural-language query for the local curated knowledge base"
    )


@register_tool(args_schema=SearchKnowledgeBaseInput)
def search_knowledge_base(query: str) -> str:
    """Semantic search over the local curated knowledge base (pgvector)."""
    with get_session() as session:
        query_embedding = get_embedding_client().embed_query(query)
        articles = KnowledgeArticle.search_by_embedding(session, query_embedding)
        return json.dumps(
            [
                {
                    "article_id": str(a.id),
                    "title": a.title,
                    "category": a.category,
                    "content": a.content,
                }
                for a in articles
            ],
            indent=2,
        )
