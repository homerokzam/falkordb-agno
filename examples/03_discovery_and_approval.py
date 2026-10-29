"""Discover an ontology from documents; review the proposal as a draft before applying."""

import asyncio

from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.models.openai import OpenAIChat

from falkordb_agno import FalkorGraphKnowledge

DOCS = [
    "Maria Silva is CTO of Nimbus. Nimbus acquired Orbit in 2024 for 30 million dollars.",
    "Orbit builds satellite software. Its founder, Joao Costa, now advises Nimbus.",
]


async def main() -> None:
    kb = FalkorGraphKnowledge(
        graph_name="discovery",
        model=OpenAIChat(id="gpt-4o-mini"),
        embedder=OpenAIEmbedder(id="text-embedding-3-small", dimensions=1536),
        require_approval=True,
    )
    draft = await kb.ontology.suggest_extensions(DOCS)  # nothing is applied
    print(f"draft {draft.id}: {len(draft.ops)} proposed ops")
    for op in draft.ops:
        print(" -", op.op)

    await kb.ontology.approve(draft.id, reviewer="me")
    await kb.ontology.apply(draft.id)
    for d in DOCS:
        await kb.ainsert(text_content=d)  # now extraction is constrained by the new ontology
    print((await kb.ontology.get()).model_dump_json(indent=2))
    await kb.graph_db.async_drop()
    await kb.aclose()


asyncio.run(main())
