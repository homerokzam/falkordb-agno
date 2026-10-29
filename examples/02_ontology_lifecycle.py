"""Ontology Lifecycle: propose -> validate -> approve -> apply -> history -> revert."""

import asyncio

from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.models.openai import OpenAIChat

from falkordb_agno import Attribute, EntityType, FalkorGraphKnowledge, Ontology
from falkordb_agno.ontology.changes import AddAttribute, RenameEntityType


async def main() -> None:
    kb = FalkorGraphKnowledge(
        graph_name="lifecycle",
        model=OpenAIChat(id="gpt-4o-mini"),
        embedder=OpenAIEmbedder(id="text-embedding-3-small", dimensions=1536),
        ontology=Ontology(entity_types=[EntityType(label="Person"), EntityType(label="Org")]),
        require_approval=True,  # every change waits for a human
    )
    m = kb.ontology
    await kb.ainsert(name="team", text_content="Ana Souza (31) works at Acme Corp.")

    # 1. propose: nothing changes yet
    cs = await m.propose(
        [
            RenameEntityType(old="Org", new="Organization"),
            AddAttribute(owner="Person", attribute=Attribute(name="age", type="INTEGER")),
        ],
        author="alice",
        message="rename Org, add Person.age",
    )
    # 2. validate: dry-run with impact estimates (no writes, no LLM calls)
    report = await m.validate(cs)
    print("ok:", report.ok, "| estimates:", report.estimates, "| warnings:", report.warnings)

    # 3. approve + apply: migrates the data graph, backfills ages with the LLM, creates v2
    await m.approve(cs.id, reviewer="bob", note="looks good")
    result = await m.apply(cs.id)
    print("applied as version", result.version)

    # 4. history and diff
    for h in await m.history():
        print(f"v{h.version} by {h.author}: {h.message}")
    print("diff v1->v2:", [op.op for op in await m.diff(1, 2)])

    # 5. revert: a *draft* that undoes later changesets (destructive ones can't be reverted)
    undo = await m.revert_to(1)
    print("revert draft ops:", [op.op for op in undo.ops])
    await kb.graph_db.async_drop()
    await kb.aclose()


asyncio.run(main())
