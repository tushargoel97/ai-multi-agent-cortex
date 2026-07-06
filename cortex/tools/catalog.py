"""Curated catalog of prebuilt third-party LangChain tools admins can enable.

Each entry lazily imports its integration, so a missing optional package just
marks the tool unavailable rather than breaking startup. Any config (e.g. API
keys) comes from the Tool row's ``config`` JSON.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    label: str
    description: str
    packages: tuple[str, ...]        # importable module names the integration needs
    config_fields: tuple[str, ...]   # config keys the admin may set (e.g. api_key)
    factory: Callable[[dict], BaseTool]


def _wikipedia(cfg: dict) -> BaseTool:
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    return WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=int(cfg.get("top_k", 3) or 3))
    )


def _arxiv(cfg: dict) -> BaseTool:
    from langchain_community.tools.arxiv.tool import ArxivQueryRun
    from langchain_community.utilities.arxiv import ArxivAPIWrapper

    return ArxivQueryRun(api_wrapper=ArxivAPIWrapper())


def _pubmed(cfg: dict) -> BaseTool:
    from langchain_community.tools.pubmed.tool import PubmedQueryRun

    return PubmedQueryRun()


def _stackexchange(cfg: dict) -> BaseTool:
    from langchain_community.tools.stackexchange.tool import StackExchangeTool
    from langchain_community.utilities.stackexchange import StackExchangeAPIWrapper

    return StackExchangeTool(api_wrapper=StackExchangeAPIWrapper())


def _ddg(cfg: dict) -> BaseTool:
    from langchain_community.tools import DuckDuckGoSearchRun

    return DuckDuckGoSearchRun()


def _tavily(cfg: dict) -> BaseTool:
    key = cfg.get("api_key") or os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("Tavily needs an API key (config.api_key or TAVILY_API_KEY)")
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

    return TavilySearchResults(
        api_wrapper=TavilySearchAPIWrapper(tavily_api_key=key),
        max_results=int(cfg.get("max_results", 5) or 5),
    )


CATALOG: dict[str, CatalogEntry] = {
    e.id: e
    for e in (
        CatalogEntry("wikipedia", "Wikipedia", "Search Wikipedia article summaries.", ("wikipedia",), (), _wikipedia),
        CatalogEntry("arxiv", "arXiv", "Search arXiv papers and abstracts.", ("arxiv",), (), _arxiv),
        CatalogEntry("pubmed", "PubMed", "Search biomedical literature on PubMed.", (), (), _pubmed),
        CatalogEntry("stackexchange", "Stack Exchange", "Search Stack Overflow / Stack Exchange.", ("stackapi",), (), _stackexchange),
        CatalogEntry("ddg_search", "DuckDuckGo Search", "Web search via DuckDuckGo.", ("duckduckgo_search",), (), _ddg),
        CatalogEntry("tavily_search", "Tavily Search", "AI-optimized web search (API key).", ("tavily",), ("api_key",), _tavily),
    )
}


def is_available(entry: CatalogEntry) -> bool:
    """Whether the integration's optional packages are importable."""
    if importlib.util.find_spec("langchain_community") is None:
        return False
    return all(importlib.util.find_spec(pkg) is not None for pkg in entry.packages)


def catalog_listing() -> list[dict[str, Any]]:
    """Serializable catalog for the admin UI."""
    return [
        {
            "id": e.id,
            "label": e.label,
            "description": e.description,
            "config_fields": list(e.config_fields),
            "available": is_available(e),
        }
        for e in CATALOG.values()
    ]


def build_catalog_tool(catalog_id: str, config: dict | None) -> BaseTool:
    entry = CATALOG.get(catalog_id)
    if entry is None:
        raise KeyError(f"Unknown catalog tool: {catalog_id}")
    return entry.factory(config or {})
