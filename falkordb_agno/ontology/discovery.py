"""Ontology discovery from sample documents, and extension proposals."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agno.utils.log import log_warning
from pydantic import BaseModel, Field

from ..llm import StructuredLLM, StructuredOutputError
from ..store.indexes import RESERVED_LABELS
from .changes import ChangeSet, apply_ops, diff_ontologies, is_additive
from .errors import OntologyDiscoveryError
from .models import RESERVED_ATTRIBUTES, EntityType, Ontology, RelationType
from .prompts import SYSTEM, chunk_prompt, normalize_prompt, summary_prompt

if TYPE_CHECKING:
    from .manager import OntologyManager


class DocSummary(BaseModel):
    main_entities: list[str] = Field(default_factory=list)
    aboutness: str = ""


class ChunkProposal(BaseModel):
    entity_types: list[EntityType] = Field(default_factory=list)
    relation_types: list[RelationType] = Field(default_factory=list)


def _to_text(src: Any) -> str:
    if isinstance(src, Path):
        return src.read_text(encoding="utf-8")
    if hasattr(src, "content"):  # agno Document
        return str(src.content)
    return str(src)


def split_text(text: str, max_chars: int = 2000) -> list[str]:
    out: list[str] = []
    buf = ""
    for para in re.split(r"\n\s*\n", text):
        if buf and len(buf) + len(para) > max_chars:
            out.append(buf)
            buf = ""
        buf = f"{buf}\n\n{para}".strip()
    if buf:
        out.append(buf)
    return out or [text]


def sample(chunks: list[str], n: int) -> list[str]:
    if len(chunks) <= n:
        return chunks
    step = len(chunks) / n
    return [chunks[int(i * step)] for i in range(n)]


def _ident(s: str, upper: bool) -> str:
    parts = [p for p in re.split(r"[^0-9A-Za-z]+", s) if p]
    if not parts:
        return "Unknown"
    return (
        "_".join(p.upper() for p in parts)
        if upper
        else "".join(p[:1].upper() + p[1:] for p in parts)
    )


def sanitize(o: Ontology) -> Ontology:
    """Make an LLM proposal safe: identifier labels, no reserved names, no dangling patterns."""
    ents: dict[str, EntityType] = {}
    for e in o.entity_types:
        label = _ident(e.label, upper=False)
        if label in RESERVED_LABELS:
            continue
        e = e.model_copy(
            update={
                "label": label,
                "attributes": [a for a in e.attributes if a.name not in RESERVED_ATTRIBUTES],
            }
        )
        ents.setdefault(label, e)
    rels: dict[str, RelationType] = {}
    for r in o.relation_types:
        label = _ident(r.label, upper=True)
        pats = [(_ident(s, False), _ident(t, False)) for s, t in r.patterns]
        pats = [p for p in pats if p[0] in ents and p[1] in ents]
        rels.setdefault(
            label,
            r.model_copy(
                update={
                    "label": label,
                    "patterns": pats,
                    "attributes": [a for a in r.attributes if a.name not in RESERVED_ATTRIBUTES],
                }
            ),
        )
    return Ontology(entity_types=list(ents.values()), relation_types=list(rels.values()))


async def discover_ontology(
    sources: list[Any],
    llm: StructuredLLM,
    *,
    sample_chunks_per_doc: int = 3,
    existing: Ontology | None = None,
    max_retries: int = 3,
    concurrency: int = 4,
) -> Ontology:
    """Document summary -> per-chunk proposal -> corpus-wide normalisation."""
    sem = asyncio.Semaphore(concurrency)

    async def one_doc(src: Any) -> Ontology:
        chunks = split_text(_to_text(src))
        picked = sample(chunks, sample_chunks_per_doc)
        async with sem:
            summary = await llm.extract(
                summary_prompt(picked[0]), DocSummary, max_retries=max_retries, system=SYSTEM
            )
        merged = Ontology()
        for i, text in enumerate(picked):
            try:
                async with sem:
                    p = await llm.extract(
                        chunk_prompt(summary.aboutness, text, existing),
                        ChunkProposal,
                        max_retries=max_retries,
                        system=SYSTEM,
                    )
            except StructuredOutputError as e:
                # skipped on purpose: one bad chunk must not abort discovery
                log_warning(  # type: ignore[no-untyped-call]
                    str(OntologyDiscoveryError(f"chunk-{i}", e.attempts, e.last_error))
                )
                continue
            merged = merged.merge(sanitize(Ontology(**p.model_dump())))
        return merged

    drafts = await asyncio.gather(*(one_doc(s) for s in sources))
    corpus = Ontology()
    for d in drafts:
        corpus = corpus.merge(d)
    if corpus.is_open:
        return corpus
    try:
        normalized = sanitize(
            await llm.extract(
                normalize_prompt(corpus), Ontology, max_retries=max_retries, system=SYSTEM
            )
        )
    except StructuredOutputError:
        return corpus  # keep the un-normalised merge
    # An empty answer means the model failed at normalising, not that nothing was found.
    return corpus if normalized.is_open else normalized


async def suggest_extensions(
    manager: OntologyManager,
    sources: list[Any],
    *,
    author: str = "discovery",
    sample_chunks_per_doc: int = 3,
) -> ChangeSet:
    """Run discovery against the current ontology and return a *draft* with additions only."""
    current = await manager.get()
    found = await discover_ontology(
        sources,
        manager.db.llm,
        sample_chunks_per_doc=sample_chunks_per_doc,
        existing=current,
        max_retries=manager.db.extraction.max_retries,
    )
    ops = [o for o in diff_ontologies(current, current.merge(found)) if is_additive([o])]
    apply_ops(current, ops)  # must still be valid
    return await manager.propose(
        ops, author=author, message=f"discovery over {len(sources)} source(s)", source="discovery"
    )
