"""Configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExtractionConfig:
    concurrency: int = 4
    max_retries: int = 3
    max_entities_per_chunk: int = 30


@dataclass
class RetrievalConfig:
    chunk_k: int = 8
    fact_k: int = 8
    entity_k: int = 8
    expand_hops: int = 1
    include_graph_context: bool = True
