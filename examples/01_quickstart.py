"""Agent + FalkorGraphKnowledge. Needs: FalkorDB on :6379 and OPENAI_API_KEY."""

import asyncio

from agno.agent import Agent
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.models.openai import OpenAIChat

from falkordb_agno import EntityType, FalkorGraphKnowledge, Ontology, RelationType

ontology = Ontology(
    entity_types=[
        EntityType(label="Person", description="A human being"),
        EntityType(label="Organization", description="A company or institution"),
    ],
    relation_types=[
        RelationType(
            label="WORKS_AT", description="Employment", patterns=[("Person", "Organization")]
        )
    ],
)


async def main() -> None:
    kb = FalkorGraphKnowledge(
        graph_name="quickstart",
        model=OpenAIChat(id="gpt-4o-mini"),
        embedder=OpenAIEmbedder(id="text-embedding-3-small", dimensions=1536),
        ontology=ontology,
    )
    await kb.ainsert(
        name="team",
        text_content="Ana Souza is a backend engineer at Acme Corp. Bruno Lima leads design at Acme Corp.",
    )
    agent = Agent(model=OpenAIChat(id="gpt-4o-mini"), knowledge=kb, search_knowledge=True)
    await agent.aprint_response("Who works at Acme Corp?", stream=True)
    await kb.aclose()


asyncio.run(main())
