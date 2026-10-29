"""OntologyTools: lets an agent inspect and *propose* ontology changes."""

from __future__ import annotations

import json
from typing import Any

from agno.tools import Toolkit
from pydantic import TypeAdapter

from .knowledge import FalkorGraphKnowledge
from .ontology.changes import Op


class OntologyTools(Toolkit):
    """Read-only by default. ``allow_apply=True`` adds approve/apply, each gated by Agno's
    human confirmation (``requires_confirmation_tools``)."""

    def __init__(
        self, knowledge: FalkorGraphKnowledge, *, allow_apply: bool = False, **kwargs: Any
    ) -> None:
        self.kb = knowledge
        tools: list[Any] = [
            self.get_ontology,
            self.list_changesets,
            self.propose_ontology_change,
            self.suggest_ontology_extensions,
        ]
        confirm: list[str] = []
        if allow_apply:
            tools += [self.approve_changeset, self.apply_changeset]
            confirm = ["approve_changeset", "apply_changeset"]
        super().__init__(
            name="ontology_tools",
            async_tools=[(t, t.__name__) for t in tools],
            requires_confirmation_tools=confirm,
            **kwargs,
        )

    async def get_ontology(self) -> str:
        """Return the current ontology (entity types, relation types) and its version."""
        m = self.kb.ontology
        o = await m.get()
        return json.dumps({"version": await m.store.current_version(), **o.model_dump(mode="json")})

    async def list_changesets(self, status: str = "") -> str:
        """List ontology changesets, optionally filtered by status (draft/approved/applied/...)."""
        items = await self.kb.ontology.list_changesets(status or None)
        return json.dumps(
            [
                {"id": c.id, "status": c.status.value, "message": c.message, "ops": len(c.ops)}
                for c in items
            ]
        )

    async def propose_ontology_change(self, ops_json: str, message: str = "") -> str:
        """Propose ontology changes as a DRAFT (nothing is applied).

        ops_json is a JSON list of ops, each with an "op" field such as add_entity_type,
        add_relation_pattern or add_attribute.
        """
        ops = TypeAdapter(list[Op]).validate_python(json.loads(ops_json))
        cs = await self.kb.ontology.propose(ops, author="agent", message=message)
        return f"draft changeset {cs.id} created; a human must approve it"

    async def suggest_ontology_extensions(self, text: str) -> str:
        """Analyse text and create a DRAFT changeset with the types/attributes it is missing."""
        cs = await self.kb.ontology.suggest_extensions([text])
        return f"draft changeset {cs.id} with {len(cs.ops)} op(s)"

    async def approve_changeset(self, changeset_id: str, note: str = "") -> str:
        """Approve a draft changeset (requires human confirmation)."""
        cs = await self.kb.ontology.approve(changeset_id, "agent", note)
        return f"{cs.id} is {cs.status.value}"

    async def apply_changeset(self, changeset_id: str) -> str:
        """Apply an approved changeset: migrates data and creates a new version."""
        r = await self.kb.ontology.apply(changeset_id)
        return f"applied as v{r.version}"
