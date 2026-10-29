"""OntologyManager: the public Ontology Lifecycle API."""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .changes import (
    AddAttribute,
    AddEntityType,
    AddRelationPattern,
    AddRelationType,
    BackfillEntity,
    BackfillRelationPattern,
    ChangeSet,
    ChangeSetStatus,
    DropAttribute,
    DropEntityType,
    DropRelationPattern,
    DropRelationType,
    Op,
    RenameAttribute,
    RenameEntityType,
    RenameRelationType,
    SetDescription,
    apply_ops,
    diff_ontologies,
)
from .errors import (
    ChangeSetStateError,
    OntologyConflictError,
    OntologyEvolutionError,
    OntologyValidationError,
    OntologyVersionConflictError,
)
from .migrations import migrate
from .models import Attribute, EntityType, Ontology, RelationType
from .store import OntologyStore, VersionInfo

if TYPE_CHECKING:
    from ..graph_db import FalkorGraphDb


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    estimates: dict[str, Any] = field(default_factory=dict)


@dataclass
class ApplyResult:
    changeset: ChangeSet
    version: int
    results: dict[str, Any]


class _Sync:
    """Blocking facade: ``manager.sync.apply(...)``."""

    def __init__(self, manager: OntologyManager) -> None:
        self._m = manager

    def __getattr__(self, name: str) -> Any:
        fn = getattr(self._m, name)
        if not inspect.iscoroutinefunction(fn):
            return fn
        return lambda *a, **kw: self._m.db.conn.run_sync(fn(*a, **kw))


class OntologyManager:
    def __init__(
        self,
        db: FalkorGraphDb,
        declared: Ontology | None = None,
        require_approval: bool = False,
    ) -> None:
        self.db = db
        self.declared = declared
        self.require_approval = require_approval
        self.store = OntologyStore(db.conn, db.graph_name)
        self._current = Ontology()
        self._ready = False
        self._init_locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}
        self.sync = _Sync(self)

    # ---- state --------------------------------------------------------------------------
    @property
    def current(self) -> Ontology:
        return self._current

    def reset(self) -> None:
        self._current, self._ready = Ontology(), False

    async def ensure_initialized(self) -> None:
        if self._ready:
            return
        lock = self._init_locks.setdefault(asyncio.get_running_loop(), asyncio.Lock())
        async with lock:
            if self._ready:
                return
            declared = self.declared
            if declared:
                declared.validate_ontology()
            version = await self.store.current_version()
            if version == 0:
                base = declared or Ontology()
                await self.store.save_state(base)
                await self.store.add_version(
                    base, author="system", message="initial", changeset_id=None
                )
                self._current, self._ready = base, True
                return
            persisted = await self.store.load()
            self._current, self._ready = persisted, True
            if declared:
                await self._reconcile_declared(persisted, declared)

    async def _reconcile_declared(self, persisted: Ontology, declared: Ontology) -> None:
        for e in declared.entity_types:
            cur = persisted.entity(e.label)
            for a in e.attributes:
                old = cur.attribute(a.name) if cur else None
                if old and old.type != a.type:
                    raise OntologyConflictError(
                        f"{e.label}.{a.name}: declared {a.type} but persisted {old.type}"
                    )
        ops = diff_ontologies(persisted, persisted.merge(declared))
        for op in ops:
            if isinstance(op, AddAttribute):
                op.backfill = False  # declaring a schema must not trigger LLM calls
        if not ops:
            return
        cs = await self.propose(
            ops, author="system", message="declared ontology", source="declared"
        )
        if not self.require_approval:
            await self.approve(cs.id, "system", "auto: additive declaration")
            await self.apply(cs.id)

    # ---- reads --------------------------------------------------------------------------
    async def get(self) -> Ontology:
        await self.ensure_initialized()
        return self._current.model_copy(deep=True)

    async def history(self) -> list[VersionInfo]:
        return await self.store.history()

    async def get_version(self, n: int) -> Ontology | None:
        return await self.store.get_version(n)

    async def diff(self, a: int, b: int) -> list[Op]:
        oa, ob = await self.store.get_version(a), await self.store.get_version(b)
        if oa is None or ob is None:
            raise OntologyValidationError("unknown version")
        return diff_ontologies(oa, ob)

    async def list_changesets(self, status: str | None = None) -> list[ChangeSet]:
        return await self.store.list_changesets(status)

    async def get_changeset(self, cs_id: str) -> ChangeSet:
        cs = await self.store.get_changeset(cs_id)
        if cs is None:
            raise ChangeSetStateError(f"unknown changeset {cs_id!r}")
        return cs

    # ---- changeset workflow -------------------------------------------------------------
    async def propose(
        self, ops: list[Op], author: str = "system", message: str = "", source: str = "manual"
    ) -> ChangeSet:
        await self.ensure_initialized()
        cs = ChangeSet(
            base_version=await self.store.current_version(),
            ops=ops,
            created_by=author,
            message=message,
            source=source,
        )
        apply_ops(self._current, cs.ops)  # fail fast on invalid proposals
        await self.store.save_changeset(cs)
        return cs

    async def validate(self, cs: ChangeSet | str) -> ValidationReport:
        if isinstance(cs, str):
            cs = await self.get_changeset(cs)
        await self.ensure_initialized()
        rep = ValidationReport(ok=True)
        try:
            apply_ops(self._current, cs.ops)
        except OntologyValidationError as e:
            rep.ok, rep.errors = False, [str(e)]
            return rep
        if cs.base_version != await self.store.current_version():
            rep.warnings.append("base_version is outdated; call rebase() before apply()")
        g = self.db.graph_name
        for op in cs.ops:
            if isinstance(op, DropEntityType):
                n = await self._count(g, f"MATCH (n:__Entity__:{op.label}) RETURN count(n)")
                rep.warnings.append(f"drop_entity_type {op.label}: deletes {n} entities")
                rep.estimates[f"drop:{op.label}"] = n
            elif isinstance(op, DropRelationType):
                n = await self._count(
                    g, "MATCH ()-[r:RELATES {rel_type:$l}]->() RETURN count(r)", {"l": op.label}
                )
                rep.warnings.append(f"drop_relation_type {op.label}: deletes {n} relations")
            elif isinstance(op, AddAttribute) and op.backfill and op.owner_kind == "entity":
                from .backfill import backfill_attribute

                r = await backfill_attribute(self.db, op, concurrency=1, dry_run=True)
                rep.estimates[f"llm_calls:{op.attribute.name}"] = r.llm_calls
            elif isinstance(op, BackfillEntity):
                n = await self._count(g, "MATCH (c:Chunk) RETURN count(c)")
                rep.estimates[f"llm_calls:backfill_entity:{op.label}"] = n
        return rep

    async def _count(self, g: str, cypher: str, params: dict[str, Any] | None = None) -> int:
        if not await self.db.conn.graph_exists(g):
            return 0
        rows = await self.db.conn.query(g, cypher, params, ro=True)
        return int(rows[0][0]) if rows else 0

    async def approve(self, cs_id: str, reviewer: str, note: str = "") -> ChangeSet:
        cs = await self.get_changeset(cs_id)
        cs.transition(ChangeSetStatus.APPROVED)
        cs.reviewed_by, cs.review_note = reviewer, note
        cs.reviewed_at = _iso()
        await self.store.save_changeset(cs)
        return cs

    async def reject(self, cs_id: str, reviewer: str, note: str = "") -> ChangeSet:
        cs = await self.get_changeset(cs_id)
        cs.transition(ChangeSetStatus.REJECTED)
        cs.reviewed_by, cs.review_note = reviewer, note
        cs.reviewed_at = _iso()
        await self.store.save_changeset(cs)
        return cs

    async def rebase(self, cs_id: str) -> ChangeSet:
        """Re-point a not-yet-applied changeset at the current version."""
        cs = await self.get_changeset(cs_id)
        if cs.status in (ChangeSetStatus.APPLIED, ChangeSetStatus.REJECTED):
            raise ChangeSetStateError(f"cannot rebase a {cs.status.value} changeset")
        apply_ops(await self.get(), cs.ops)
        cs.base_version = await self.store.current_version()
        await self.store.save_changeset(cs)
        return cs

    async def apply(self, cs_id: str, *, concurrency: int = 4) -> ApplyResult:
        await self.ensure_initialized()
        cs = await self.get_changeset(cs_id)
        if cs.status not in (ChangeSetStatus.APPROVED, ChangeSetStatus.FAILED):
            raise ChangeSetStateError(f"changeset is {cs.status.value}; it must be approved")
        current_v = await self.store.current_version()
        if cs.base_version != current_v:
            raise OntologyVersionConflictError(
                f"changeset based on v{cs.base_version}, ontology is at v{current_v}"
            )
        holder = await self.store.acquire_lock()
        try:
            async with self.db.evolution_lock():
                cur = self._current
                results: dict[str, Any] = {}
                try:
                    for i, op in enumerate(cs.ops):
                        nxt = apply_ops(cur, [op])
                        results[f"{i}:{op.op}"] = await migrate(
                            self.db, op, cur, nxt, concurrency=concurrency
                        )
                        cur = nxt
                    await self.store.save_state(cur)
                    n = await self.store.add_version(
                        cur,
                        author=cs.reviewed_by or cs.created_by,
                        message=cs.message,
                        changeset_id=cs.id,
                    )
                except Exception as e:
                    cs.status = ChangeSetStatus.FAILED
                    cs.error = str(e)
                    await self.store.save_changeset(cs)
                    if isinstance(e, OntologyEvolutionError):
                        raise
                    raise OntologyEvolutionError(str(e)) from e
                cs.status, cs.applied_version, cs.results, cs.error = (
                    ChangeSetStatus.APPLIED,
                    n,
                    results,
                    None,
                )
                await self.store.save_changeset(cs)
                self._current = cur
                return ApplyResult(cs, n, results)
        finally:
            await self.store.release_lock(holder)

    async def _shortcut(self, ops: list[Op], author: str, message: str) -> ChangeSet:
        cs = await self.propose(ops, author=author, message=message)
        if self.require_approval:
            return cs
        await self.approve(cs.id, author, "auto-approved (require_approval=False)")
        return (await self.apply(cs.id)).changeset

    # ---- one-op shortcuts ---------------------------------------------------------------
    async def add_entity_type(self, entity: EntityType, author: str = "system") -> ChangeSet:
        return await self._shortcut([AddEntityType(entity=entity)], author, f"add {entity.label}")

    async def add_relation_type(self, relation: RelationType, author: str = "system") -> ChangeSet:
        return await self._shortcut(
            [AddRelationType(relation=relation)], author, f"add {relation.label}"
        )

    async def add_relation_pattern(
        self, rel: str, src: str, tgt: str, author: str = "system"
    ) -> ChangeSet:
        return await self._shortcut(
            [AddRelationPattern(relation=rel, source=src, target=tgt)], author, f"pattern {rel}"
        )

    async def add_attribute(
        self, owner: str, attribute: Attribute, *, backfill: bool = True, author: str = "system"
    ) -> ChangeSet:
        return await self._shortcut(
            [AddAttribute(owner=owner, attribute=attribute, backfill=backfill)],
            author,
            f"add {owner}.{attribute.name}",
        )

    async def set_description(
        self, owner: str, description: str, author: str = "system"
    ) -> ChangeSet:
        return await self._shortcut(
            [SetDescription(owner=owner, description=description)], author, "describe"
        )

    async def rename_entity_type(self, old: str, new: str, author: str = "system") -> ChangeSet:
        return await self._shortcut(
            [RenameEntityType(old=old, new=new)], author, f"rename {old}->{new}"
        )

    async def rename_relation_type(self, old: str, new: str, author: str = "system") -> ChangeSet:
        return await self._shortcut(
            [RenameRelationType(old=old, new=new)], author, f"rename {old}->{new}"
        )

    async def rename_attribute(
        self, owner: str, old: str, new: str, author: str = "system"
    ) -> ChangeSet:
        return await self._shortcut(
            [RenameAttribute(owner=owner, old=old, new=new)], author, f"rename {owner}.{old}"
        )

    async def drop_entity_type(self, label: str, author: str = "system") -> ChangeSet:
        return await self._shortcut([DropEntityType(label=label)], author, f"drop {label}")

    async def drop_relation_type(self, label: str, author: str = "system") -> ChangeSet:
        return await self._shortcut([DropRelationType(label=label)], author, f"drop {label}")

    async def drop_relation_pattern(
        self, rel: str, src: str, tgt: str, author: str = "system"
    ) -> ChangeSet:
        return await self._shortcut(
            [DropRelationPattern(relation=rel, source=src, target=tgt)],
            author,
            f"drop pattern {rel}",
        )

    async def drop_attribute(self, owner: str, name: str, author: str = "system") -> ChangeSet:
        return await self._shortcut(
            [DropAttribute(owner=owner, name=name)], author, f"drop {owner}.{name}"
        )

    async def backfill_entity(self, label: str, author: str = "system") -> ChangeSet:
        return await self._shortcut([BackfillEntity(label=label)], author, f"backfill {label}")

    async def backfill_relation_pattern(
        self, rel: str, src: str, tgt: str, author: str = "system"
    ) -> ChangeSet:
        return await self._shortcut(
            [BackfillRelationPattern(relation=rel, source=src, target=tgt)],
            author,
            f"backfill {rel}",
        )

    # ---- revert -------------------------------------------------------------------------
    async def revert_to(self, version: int, author: str = "system") -> ChangeSet:
        """Draft changeset that undoes every applied changeset after ``version``."""
        await self.ensure_initialized()
        applied = [
            c
            for c in await self.store.list_changesets("applied")
            if (c.applied_version or 0) > version
        ]
        applied.sort(key=lambda c: c.applied_version or 0, reverse=True)
        if not applied:
            raise OntologyValidationError(f"nothing to revert after v{version}")
        inverse: list[Op] = []
        for c in applied:
            before = await self.store.get_version((c.applied_version or 1) - 1)
            if before is None:
                raise OntologyValidationError(
                    f"missing snapshot for v{(c.applied_version or 1) - 1}"
                )
            for op in reversed(c.ops):
                inv = _invert(op, before)
                if inv is not None:
                    inverse.append(inv)
        return await self.propose(
            inverse, author=author, message=f"revert to v{version}", source="revert"
        )

    # ---- discovery (see discovery.py) ---------------------------------------------------
    async def suggest_extensions(self, sources: list[Any], **kw: Any) -> ChangeSet:
        from .discovery import suggest_extensions

        return await suggest_extensions(self, sources, **kw)


def _invert(op: Op, before: Ontology) -> Op | None:
    if isinstance(op, AddEntityType):
        return DropEntityType(label=op.entity.label)
    if isinstance(op, AddRelationType):
        return DropRelationType(label=op.relation.label)
    if isinstance(op, AddRelationPattern):
        return DropRelationPattern(relation=op.relation, source=op.source, target=op.target)
    if isinstance(op, AddAttribute):
        return DropAttribute(owner=op.owner, name=op.attribute.name)
    if isinstance(op, RenameEntityType):
        return RenameEntityType(old=op.new, new=op.old)
    if isinstance(op, RenameRelationType):
        return RenameRelationType(old=op.new, new=op.old)
    if isinstance(op, RenameAttribute):
        return RenameAttribute(owner=op.owner, old=op.new, new=op.old)
    if isinstance(op, SetDescription):
        prev = before.entity(op.owner) or before.relation(op.owner)
        return SetDescription(owner=op.owner, description=(prev.description if prev else "") or "")
    if isinstance(op, (BackfillEntity, BackfillRelationPattern)):
        return None
    raise OntologyValidationError(f"{op.op} is destructive and cannot be reverted automatically")


def _iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
