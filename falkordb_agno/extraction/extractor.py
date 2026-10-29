"""Chunk -> extracted graph, constrained by the ontology."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, WithJsonSchema

from ..llm import StructuredLLM
from ..ontology.models import Ontology
from ..store.indexes import entity_label, normalize_label
from .prompts import SYSTEM, extraction_prompt


class ExtractionError(RuntimeError):
    """Every chunk of a document failed extraction; nothing was written for it."""

    def __init__(self, document: str, errors: dict[str, str]) -> None:
        first = next(iter(errors.values()), "")
        super().__init__(
            f"extraction failed for all {len(errors)} chunk(s) of {document!r}: {first}"
        )
        self.document, self.errors = document, errors


class ExtractedAttribute(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)

    name: str
    value: str


def attributes_to_dict(raw: Any) -> Any:
    """Wire ``[{name, value}]`` -> ``{name: value}``; a repeated name keeps its last value."""
    if isinstance(raw, list):
        return {a.name: a.value for a in map(ExtractedAttribute.model_validate, raw)}
    return raw


# Structured-output schemas must not contain open dicts: Gemini rejects `additionalProperties`
# and OpenAI/xAI/Anthropic strict modes rewrite it to false (the model can only answer {}).
# The LLM sees a list of name/value pairs; Python code gets a dict.
NamedValues = Annotated[
    dict[str, Any],
    BeforeValidator(attributes_to_dict),
    WithJsonSchema({"type": "array", "items": ExtractedAttribute.model_json_schema()}),
]


class ExtractedEntity(BaseModel):
    name: str
    type: str
    description: str = ""
    attributes: NamedValues = Field(default_factory=dict)


class ExtractedRelation(BaseModel):
    source: str
    target: str
    type: str
    fact: str = ""
    description: str = ""


class ExtractedGraph(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


def normalize_labels(graph: ExtractedGraph) -> ExtractedGraph:
    """Entity and relation types as Cypher identifiers, so ids and writes agree across chunks.
    Reserved entity labels are renamed (``Document`` -> ``Document_Entity``)."""
    return ExtractedGraph(
        entities=[e.model_copy(update={"type": entity_label(e.type)}) for e in graph.entities],
        relations=[r.model_copy(update={"type": normalize_label(r.type)}) for r in graph.relations],
    )


class GraphExtractor:
    def __init__(self, llm: StructuredLLM, max_retries: int = 3, max_entities: int = 30) -> None:
        self.llm, self.max_retries, self.max_entities = llm, max_retries, max_entities

    async def extract(self, ontology: Ontology, text: str) -> ExtractedGraph:
        graph = await self.llm.extract(
            extraction_prompt(ontology, text, self.max_entities),
            ExtractedGraph,
            max_retries=self.max_retries,
            system=SYSTEM,
        )
        return normalize_labels(graph)
