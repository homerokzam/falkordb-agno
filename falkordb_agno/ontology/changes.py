"""Ontology change operations and changesets (draft -> approved -> applied)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .errors import ChangeSetStateError
from .models import Attribute, EntityType, Ontology, RelationType


class _Op(BaseModel):
    pass


class AddEntityType(_Op):
    op: Literal["add_entity_type"] = "add_entity_type"
    entity: EntityType


class AddRelationType(_Op):
    op: Literal["add_relation_type"] = "add_relation_type"
    relation: RelationType


class AddRelationPattern(_Op):
    op: Literal["add_relation_pattern"] = "add_relation_pattern"
    relation: str
    source: str
    target: str


class AddAttribute(_Op):
    op: Literal["add_attribute"] = "add_attribute"
    owner: str  # entity label, or relation label when owner_kind == "relation"
    owner_kind: Literal["entity", "relation"] = "entity"
    attribute: Attribute
    backfill: bool = True


class SetDescription(_Op):
    op: Literal["set_description"] = "set_description"
    owner: str
    owner_kind: Literal["entity", "relation"] = "entity"
    description: str


class RenameEntityType(_Op):
    op: Literal["rename_entity_type"] = "rename_entity_type"
    old: str
    new: str


class RenameRelationType(_Op):
    op: Literal["rename_relation_type"] = "rename_relation_type"
    old: str
    new: str


class RenameAttribute(_Op):
    op: Literal["rename_attribute"] = "rename_attribute"
    owner: str
    old: str
    new: str


class DropEntityType(_Op):
    op: Literal["drop_entity_type"] = "drop_entity_type"
    label: str


class DropRelationType(_Op):
    op: Literal["drop_relation_type"] = "drop_relation_type"
    label: str


class DropRelationPattern(_Op):
    op: Literal["drop_relation_pattern"] = "drop_relation_pattern"
    relation: str
    source: str
    target: str


class DropAttribute(_Op):
    op: Literal["drop_attribute"] = "drop_attribute"
    owner: str
    name: str


class BackfillEntity(_Op):
    op: Literal["backfill_entity"] = "backfill_entity"
    label: str


class BackfillRelationPattern(_Op):
    op: Literal["backfill_relation_pattern"] = "backfill_relation_pattern"
    relation: str
    source: str
    target: str


Op = Annotated[
    AddEntityType
    | AddRelationType
    | AddRelationPattern
    | AddAttribute
    | SetDescription
    | RenameEntityType
    | RenameRelationType
    | RenameAttribute
    | DropEntityType
    | DropRelationType
    | DropRelationPattern
    | DropAttribute
    | BackfillEntity
    | BackfillRelationPattern,
    Field(discriminator="op"),
]

ADDITIVE_OPS = (
    AddEntityType,
    AddRelationType,
    AddRelationPattern,
    AddAttribute,
    SetDescription,
    BackfillEntity,
    BackfillRelationPattern,
)


class ChangeSetStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    FAILED = "failed"


_TRANSITIONS: dict[ChangeSetStatus, set[ChangeSetStatus]] = {
    ChangeSetStatus.DRAFT: {ChangeSetStatus.APPROVED, ChangeSetStatus.REJECTED},
    ChangeSetStatus.APPROVED: {
        ChangeSetStatus.APPLIED,
        ChangeSetStatus.FAILED,
        ChangeSetStatus.REJECTED,
    },
    ChangeSetStatus.FAILED: {ChangeSetStatus.APPLIED, ChangeSetStatus.FAILED},
    ChangeSetStatus.REJECTED: set(),
    ChangeSetStatus.APPLIED: set(),
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ChangeSet(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: ChangeSetStatus = ChangeSetStatus.DRAFT
    base_version: int
    ops: list[Op]
    message: str = ""
    created_by: str = "system"
    source: str = "manual"  # manual | discovery | revert
    created_at: str = Field(default_factory=_now)
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    review_note: str | None = None
    applied_version: int | None = None
    error: str | None = None
    results: dict[str, object] = Field(default_factory=dict)

    def transition(self, to: ChangeSetStatus) -> None:
        if to not in _TRANSITIONS[self.status]:
            raise ChangeSetStateError(f"cannot go from {self.status.value} to {to.value}")
        self.status = to


def apply_ops(ontology: Ontology, ops: list[Op]) -> Ontology:
    """Pure function: the ontology that results from applying ``ops`` (schema only)."""
    from .errors import OntologyValidationError

    out = ontology.model_copy(deep=True)

    def need_entity(label: str) -> EntityType:
        e = out.entity(label)
        if e is None:
            raise OntologyValidationError(f"unknown entity type {label!r}")
        return e

    def need_relation(label: str) -> RelationType:
        r = out.relation(label)
        if r is None:
            raise OntologyValidationError(f"unknown relation type {label!r}")
        return r

    for op in ops:
        if isinstance(op, AddEntityType):
            if out.entity(op.entity.label):
                raise OntologyValidationError(f"entity type {op.entity.label!r} already exists")
            out.entity_types.append(op.entity.model_copy(deep=True))
        elif isinstance(op, AddRelationType):
            if out.relation(op.relation.label):
                raise OntologyValidationError(f"relation {op.relation.label!r} already exists")
            out.relation_types.append(op.relation.model_copy(deep=True))
        elif isinstance(op, AddRelationPattern):
            need_entity(op.source), need_entity(op.target)
            r = need_relation(op.relation)
            if (op.source, op.target) not in r.patterns:
                r.patterns.append((op.source, op.target))
        elif isinstance(op, AddAttribute):
            owner = need_entity(op.owner) if op.owner_kind == "entity" else need_relation(op.owner)
            if any(a.name == op.attribute.name for a in owner.attributes):
                raise OntologyValidationError(
                    f"{op.owner}: attribute {op.attribute.name!r} already declared"
                )
            owner.attributes.append(op.attribute.model_copy())
        elif isinstance(op, SetDescription):
            tgt = need_entity(op.owner) if op.owner_kind == "entity" else need_relation(op.owner)
            tgt.description = op.description
        elif isinstance(op, RenameEntityType):
            e = need_entity(op.old)
            if out.entity(op.new):
                raise OntologyValidationError(f"entity type {op.new!r} already exists")
            e.label = op.new
            for r in out.relation_types:
                r.patterns = [
                    (op.new if s == op.old else s, op.new if t == op.old else t)
                    for s, t in r.patterns
                ]
        elif isinstance(op, RenameRelationType):
            r = need_relation(op.old)
            if out.relation(op.new):
                raise OntologyValidationError(f"relation {op.new!r} already exists")
            r.label = op.new
        elif isinstance(op, RenameAttribute):
            owner = out.entity(op.owner) or need_relation(op.owner)
            attr = next((a for a in owner.attributes if a.name == op.old), None)
            if attr is None:
                raise OntologyValidationError(f"{op.owner}: unknown attribute {op.old!r}")
            attr.name = op.new
        elif isinstance(op, DropEntityType):
            need_entity(op.label)
            out.entity_types = [e for e in out.entity_types if e.label != op.label]
            for r in out.relation_types:
                r.patterns = [p for p in r.patterns if op.label not in p]
        elif isinstance(op, DropRelationType):
            need_relation(op.label)
            out.relation_types = [r for r in out.relation_types if r.label != op.label]
        elif isinstance(op, DropRelationPattern):
            r = need_relation(op.relation)
            r.patterns = [p for p in r.patterns if p != (op.source, op.target)]
        elif isinstance(op, DropAttribute):
            owner = out.entity(op.owner) or need_relation(op.owner)
            owner.attributes = [a for a in owner.attributes if a.name != op.name]
        elif isinstance(op, BackfillEntity):
            need_entity(op.label)
        elif isinstance(op, BackfillRelationPattern):
            need_relation(op.relation)
    out.validate_ontology()
    return out


def diff_ontologies(old: Ontology, new: Ontology) -> list[Op]:
    """Structural diff old -> new expressed as ops (adds/drops; no rename detection)."""
    ops: list[Op] = []
    for e in new.entity_types:
        cur = old.entity(e.label)
        if cur is None:
            ops.append(AddEntityType(entity=e))
            continue
        if e.description and e.description != cur.description:
            ops.append(SetDescription(owner=e.label, description=e.description))
        have = {a.name for a in cur.attributes}
        ops += [
            AddAttribute(owner=e.label, attribute=a) for a in e.attributes if a.name not in have
        ]
        want = {a.name for a in e.attributes}
        ops += [DropAttribute(owner=e.label, name=n) for n in have - want]
    for r in new.relation_types:
        cur_r = old.relation(r.label)
        if cur_r is None:
            ops.append(AddRelationType(relation=r))
            continue
        ops += [
            AddRelationPattern(relation=r.label, source=s, target=t)
            for s, t in r.patterns
            if (s, t) not in cur_r.patterns
        ]
        ops += [
            DropRelationPattern(relation=r.label, source=s, target=t)
            for s, t in cur_r.patterns
            if (s, t) not in r.patterns
        ]
    ops += [
        DropRelationType(label=r.label) for r in old.relation_types if not new.relation(r.label)
    ]
    ops += [DropEntityType(label=e.label) for e in old.entity_types if not new.entity(e.label)]
    return ops


def is_additive(ops: list[Op]) -> bool:
    return all(isinstance(o, ADDITIVE_OPS) for o in ops)
