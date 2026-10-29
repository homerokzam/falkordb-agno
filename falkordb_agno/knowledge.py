"""FalkorGraphKnowledge: an agno ``Knowledge`` backed by a FalkorDB GraphRAG graph."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from agno.knowledge import Knowledge
from agno.knowledge.chunking.strategy import ChunkingStrategy
from agno.knowledge.reader.markdown_reader import MarkdownReader

from .chunking import HeadingChunking
from .config import ExtractionConfig, RetrievalConfig
from .graph_db import FalkorGraphDb
from .ontology.manager import OntologyManager
from .ontology.models import Ontology
from .store.connection import FalkorConnection


def summarize_ontology(o: Ontology) -> str:
    if o.is_open:
        return "The graph has no fixed ontology yet."
    ents = ", ".join(e.label for e in o.entity_types)
    rels = "; ".join(
        f"{r.label}({', '.join(f'{s}->{t}' for s, t in r.patterns) or 'any'})"
        for r in o.relation_types
    )
    return f"Graph entity types: {ents}. Relations: {rels}."


class FalkorGraphKnowledge(Knowledge):
    """Drop-in ``Agent(knowledge=...)`` that stores and retrieves through a knowledge graph.

    ``.ontology`` is the Ontology Lifecycle API (propose / approve / apply / history / ...).
    """

    def __init__(
        self,
        *,
        graph_name: str,
        model: Any,
        embedder: Any,
        host: str = "localhost",
        port: int = 6379,
        password: str | None = None,
        connection: FalkorConnection | None = None,
        ontology: Ontology | None = None,
        require_approval: bool = False,
        extraction: ExtractionConfig | None = None,
        retrieval: RetrievalConfig | None = None,
        chunking: ChunkingStrategy | None = None,
        **knowledge_kwargs: Any,
    ) -> None:
        # Used for Markdown files inserted without an explicit reader.
        self.chunking = chunking or HeadingChunking()
        conn = connection or FalkorConnection(host=host, port=port, password=password)
        self.graph_db = FalkorGraphDb(
            graph_name=graph_name,
            model=model,
            embedder=embedder,
            connection=conn,
            ontology=ontology,
            require_approval=require_approval,
            extraction=extraction,
            retrieval=retrieval,
        )
        knowledge_kwargs.setdefault("name", graph_name)
        super().__init__(vector_db=self.graph_db, **knowledge_kwargs)

    @property
    def ontology(self) -> OntologyManager:
        return self.graph_db.manager

    def _bind_default_reader(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Bind to ``Knowledge.insert``'s signature; a Markdown file without an explicit
        reader gets ``MarkdownReader(chunking_strategy=self.chunking)``."""
        bound = inspect.signature(Knowledge.insert).bind(self, *args, **kwargs)
        path = bound.arguments.get("path")
        if (
            bound.arguments.get("reader") is None
            and path
            and Path(path).suffix.lower() in {".md", ".markdown"}
            and not Path(path).is_dir()
        ):
            bound.arguments["reader"] = MarkdownReader(chunking_strategy=self.chunking)
        return bound

    def insert(self, *args: Any, **kwargs: Any) -> None:
        bound = self._bind_default_reader(args, kwargs)
        super().insert(*bound.args[1:], **bound.kwargs)

    async def ainsert(self, *args: Any, **kwargs: Any) -> None:
        bound = self._bind_default_reader(args, kwargs)
        await super().ainsert(*bound.args[1:], **bound.kwargs)

    def build_context(
        self, enable_agentic_filters: bool = False, user_id: str | None = None, **kwargs: Any
    ) -> str:
        base = super().build_context(
            enable_agentic_filters=enable_agentic_filters, user_id=user_id, **kwargs
        )
        return base.replace(
            "</knowledge_base>", f"{summarize_ontology(self.ontology.current)}\n</knowledge_base>"
        )

    async def abuild_context(
        self, enable_agentic_filters: bool = False, user_id: str | None = None, **kwargs: Any
    ) -> str:
        await self.ontology.ensure_initialized()
        base = await super().abuild_context(
            enable_agentic_filters=enable_agentic_filters, user_id=user_id, **kwargs
        )
        return base.replace(
            "</knowledge_base>", f"{summarize_ontology(self.ontology.current)}\n</knowledge_base>"
        )

    async def aclose(self) -> None:
        await self.graph_db.aclose()

    def close(self) -> None:
        self.graph_db.conn.close()
