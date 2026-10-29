"""Serve the knowledge through AgentOS. Run: fastapi dev examples/04_agentos.py"""

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.models.openai import OpenAIChat
from agno.os import AgentOS

from falkordb_agno import FalkorGraphKnowledge, OntologyTools

kb = FalkorGraphKnowledge(
    graph_name="agentos",
    model=OpenAIChat(id="gpt-4o-mini"),
    embedder=OpenAIEmbedder(id="text-embedding-3-small", dimensions=1536),
    contents_db=SqliteDb(db_file="tmp/contents.db"),  # AgentOS only lists knowledge with a contents_db
)
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    knowledge=kb,
    search_knowledge=True,
    # The agent can inspect and *propose* ontology changes; a human approves them.
    tools=[OntologyTools(kb)],
)
app = AgentOS(agents=[agent]).get_app()
