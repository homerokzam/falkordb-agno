"""Ontology models: entity types, relation types, attributes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, WithJsonSchema, field_validator

from ..store.indexes import RESERVED_LABELS
from .errors import OntologyValidationError

AttrType = Literal["STRING", "INTEGER", "FLOAT", "BOOLEAN", "DATE", "LIST"]

RESERVED_ATTRIBUTES = frozenset(
    {"name", "description", "id", "rel_type", "fact", "source_chunk_ids", "embedding", "aliases"}
)


class Attribute(BaseModel):
    name: str
    type: AttrType = "STRING"
    description: str | None = None
    required: bool = False

    @field_validator("type", mode="before")
    @classmethod
    def _upper(cls, v: object) -> object:
        return v.upper() if isinstance(v, str) else v


class EntityType(BaseModel):
    label: str
    description: str | None = None
    attributes: list[Attribute] = Field(default_factory=list)

    def attribute(self, name: str) -> Attribute | None:
        return next((a for a in self.attributes if a.name == name), None)


def _pairs(raw: object) -> object:
    """Accept the wire form ``[{source, target}]`` as well as stored ``[[source, target]]``."""
    if isinstance(raw, list):
        return [(p.get("source"), p.get("target")) if isinstance(p, dict) else p for p in raw]
    return raw


# Tuples become `prefixItems`, outside the JSON Schema subset LLM providers accept for
# structured output, so the LLM sees {source, target} objects instead.
Patterns = Annotated[
    list[tuple[str, str]],
    BeforeValidator(_pairs),
    WithJsonSchema(
        {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"source": {"type": "string"}, "target": {"type": "string"}},
                "required": ["source", "target"],
            },
        }
    ),
]


class RelationType(BaseModel):
    label: str
    description: str | None = None
    # Directed (source label, target label) pairs. Empty means open mode: any endpoints.
    patterns: Patterns = Field(default_factory=list)
    attributes: list[Attribute] = Field(default_factory=list)


class Ontology(BaseModel):
    entity_types: list[EntityType] = Field(default_factory=list)
    relation_types: list[RelationType] = Field(default_factory=list)

    # ---- lookup -------------------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        return not self.entity_types and not self.relation_types

    def entity(self, label: str) -> EntityType | None:
        return next((e for e in self.entity_types if e.label == label), None)

    def relation(self, label: str) -> RelationType | None:
        return next((r for r in self.relation_types if r.label == label), None)

    # ---- validation ---------------------------------------------------------------------
    def validate_ontology(self) -> None:
        errors: list[str] = []
        labels = [e.label for e in self.entity_types]
        for dup in {x for x in labels if labels.count(x) > 1}:
            errors.append(f"duplicate entity type {dup!r}")
        rels = [r.label for r in self.relation_types]
        for dup in {x for x in rels if rels.count(x) > 1}:
            errors.append(f"duplicate relation type {dup!r}")
        for e in self.entity_types:
            if e.label in RESERVED_LABELS:
                errors.append(
                    f"entity label {e.label!r} is reserved for the graph's structure; "
                    f"choose another name, e.g. {e.label + '_Entity'!r}"
                )
            errors += _check_attrs(f"entity {e.label}", e.attributes)
        for r in self.relation_types:
            errors += _check_attrs(f"relation {r.label}", r.attributes)
            for src, tgt in r.patterns:
                for end in (src, tgt):
                    if end not in labels:
                        errors.append(f"relation {r.label}: pattern endpoint {end!r} is undeclared")
        if errors:
            raise OntologyValidationError("; ".join(errors))

    # ---- (de)serialisation --------------------------------------------------------------
    def digest(self) -> str:
        canon = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canon.encode()).hexdigest()[:16]

    @classmethod
    def from_file(cls, path: str | Path) -> Ontology:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_file(self, path: str | Path, indent: int = 2) -> None:
        Path(path).write_text(self.model_dump_json(indent=indent), encoding="utf-8")

    # ---- merge --------------------------------------------------------------------------
    def merge(self, other: Ontology) -> Ontology:
        """Union by label and attribute name; ``other`` wins on conflicts."""
        out = self.model_copy(deep=True)
        for e in other.entity_types:
            cur = out.entity(e.label)
            if cur is None:
                out.entity_types.append(e.model_copy(deep=True))
                continue
            cur.description = e.description or cur.description
            cur.attributes = _merge_attrs(cur.attributes, e.attributes)
        for r in other.relation_types:
            cur_r = out.relation(r.label)
            if cur_r is None:
                out.relation_types.append(r.model_copy(deep=True))
                continue
            cur_r.description = r.description or cur_r.description
            cur_r.attributes = _merge_attrs(cur_r.attributes, r.attributes)
            for p in r.patterns:
                if p not in cur_r.patterns:
                    cur_r.patterns.append(p)
        return out


def _check_attrs(owner: str, attrs: list[Attribute]) -> list[str]:
    errs: list[str] = []
    names = [a.name for a in attrs]
    for a in attrs:
        if a.name in RESERVED_ATTRIBUTES:
            errs.append(f"{owner}: attribute {a.name!r} is reserved")
    for dup in {n for n in names if names.count(n) > 1}:
        errs.append(f"{owner}: duplicate attribute {dup!r}")
    return errs


def _merge_attrs(base: list[Attribute], extra: list[Attribute]) -> list[Attribute]:
    by_name = {a.name: a for a in base}
    for a in extra:
        by_name[a.name] = a
    return list(by_name.values())
