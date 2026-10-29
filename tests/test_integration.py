"""Needs a live FalkorDB: RUN_INTEGRATION=1 FALKORDB_PORT=6379 uv run pytest tests/"""

import os
import uuid

import pytest
from agno.knowledge.document import Document

from falkordb_agno import (
    Attribute,
    EntityType,
    FalkorGraphKnowledge,
    Ontology,
    OntologyConflictError,
    OntologyEvolutionError,
    OntologyVersionConflictError,
    RelationType,
)
from falkordb_agno.ontology.changes import DropEntityType
from tests.fakes import FakeEmbedder, FakeModel, kg_handler

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("RUN_INTEGRATION"), reason="set RUN_INTEGRATION=1"),
]
PORT = int(os.getenv("FALKORDB_PORT", "6379"))
ONTO = Ontology(
    entity_types=[EntityType(label="Person"), EntityType(label="Org")],
    relation_types=[RelationType(label="WORKS_AT", patterns=[("Person", "Org")])],
)
DOCS = [
    Document(id="c0", name="d1", content_id="d1", content="Ana works at Acme. Ana is 31."),
    Document(id="c1", name="d1", content_id="d1", content="Bob works at Acme. Bob lives in Rio."),
]


@pytest.fixture
async def kb():
    k = FalkorGraphKnowledge(
        graph_name=f"fa_test_{uuid.uuid4().hex[:8]}",
        model=FakeModel(kg_handler),
        embedder=FakeEmbedder(),
        port=PORT,
        ontology=ONTO,
    )
    await k.graph_db.async_create()
    yield k
    await k.graph_db.async_drop()
    await k.aclose()
    k.close()


async def cypher(kb, q, **p):
    return await kb.graph_db.conn.query(kb.graph_db.graph_name, q, p or None)


async def test_ingest_prunes_to_ontology_and_searches(kb):
    await kb.graph_db.async_insert("h", DOCS)
    ents = {r[0] for r in await cypher(kb, "MATCH (e:__Entity__) RETURN e.name")}
    assert ents == {"Ana", "Bob", "Acme"}  # Rio/City and LIVES_IN pruned
    assert (await cypher(kb, "MATCH ()-[r:RELATES]->() RETURN count(r)"))[0][0] == 2
    res = await kb.aretrieve("Who works at Acme?")
    assert {d.id for d in res} >= {"c0", "c1"}
    assert all("document_name" in d.meta_data for d in res if d.id != "graph_context")


async def test_filters_scope_results_and_graph_context(kb):
    await kb.graph_db.async_insert("h", DOCS, filters={"lang": "en"})
    assert await kb.graph_db.async_search("Acme", filters={"lang": "fr"}) == []
    assert await kb.graph_db.async_search("Acme", filters={"lang": "en"})


async def test_rename_migrates_data_and_versions(kb):
    await kb.graph_db.async_insert("h", DOCS)
    cs = await kb.ontology.rename_entity_type("Person", "Human")
    assert cs.applied_version == 2
    assert {r[0] for r in await cypher(kb, "MATCH (e:Human) RETURN e.id")} == {
        "ana__Human",
        "bob__Human",
    }
    assert (await kb.ontology.get()).relation("WORKS_AT").patterns == [("Human", "Org")]
    assert [h.version for h in await kb.ontology.history()] == [1, 2]
    assert (await kb.ontology.get_version(1)).entity("Person") is not None


async def test_add_attribute_backfill_is_idempotent(kb):
    await kb.graph_db.async_insert("h", DOCS)
    await kb.ontology.add_attribute("Person", Attribute(name="age", type="integer"))
    assert {r[0]: r[1] for r in await cypher(kb, "MATCH (p:Person) RETURN p.name, p.age")} == {
        "Ana": 31,
        "Bob": None,
    }
    calls = len(kb.graph_db.model.prompts)
    from falkordb_agno.ontology.backfill import backfill_attribute
    from falkordb_agno.ontology.changes import AddAttribute

    r = await backfill_attribute(
        kb.graph_db,
        AddAttribute(owner="Person", attribute=Attribute(name="age", type="integer")),
        concurrency=2,
    )
    assert r.skipped == 2 and r.scanned == 0 and len(kb.graph_db.model.prompts) == calls


async def test_failed_backfill_creates_no_version(kb):
    await kb.graph_db.async_insert("h", DOCS)

    def boom(prompt, schema):
        raise RuntimeError("llm down")

    kb.graph_db.model.handler = boom
    kb.graph_db.llm.model = kb.graph_db.model
    kb.graph_db.extraction.max_retries = 1
    with pytest.raises(OntologyEvolutionError):
        await kb.ontology.add_attribute("Person", Attribute(name="age", type="integer"))
    assert [h.version for h in await kb.ontology.history()] == [1]
    assert "age" not in {a.name for a in (await kb.ontology.get()).entity("Person").attributes}
    failed = await kb.ontology.list_changesets("failed")
    assert len(failed) == 1 and failed[0].error


async def test_approval_workflow_and_version_conflict():
    k = FalkorGraphKnowledge(
        graph_name=f"fa_test_{uuid.uuid4().hex[:8]}",
        model=FakeModel(kg_handler),
        embedder=FakeEmbedder(),
        port=PORT,
        ontology=ONTO,
        require_approval=True,
    )
    try:
        await k.graph_db.async_create()
        m = k.ontology
        a = await m.rename_entity_type("Person", "Human")  # draft only
        assert a.status.value == "draft" and (await m.get()).entity("Person")
        b = await m.propose([DropEntityType(label="Org")])
        await m.approve(a.id, "alice")
        await m.apply(a.id)
        await m.approve(b.id, "alice")
        with pytest.raises(OntologyVersionConflictError):
            await m.apply(b.id)  # based on v1, ontology is at v2
        r = await m.reject(b.id, "alice", "no")
        assert r.status.value == "rejected"
    finally:
        await k.graph_db.async_drop()
        await k.aclose()
        k.close()


async def test_open_ontology_normalises_spaced_labels():
    def handler(prompt, schema):
        return {
            "entities": [
                {"name": "Unimed", "type": "Hospital Service"},
                {"name": "Ana", "type": "Person"},
            ],
            "relations": [{"source": "Ana", "target": "Unimed", "type": "works at"}],
        }

    k = FalkorGraphKnowledge(
        graph_name=f"fa_test_{uuid.uuid4().hex[:8]}",
        model=FakeModel(handler),
        embedder=FakeEmbedder(),
        port=PORT,
    )
    try:
        await k.graph_db.async_create()
        await k.ainsert(name="contrato", text_content="Ana works at Unimed.")
        rows = await cypher(k, "MATCH (e:Hospital_Service) RETURN e.id")
        assert rows == [["unimed__Hospital_Service"]]
        rel = await cypher(k, "MATCH ()-[r:RELATES]->() RETURN r.rel_type")
        assert rel == [["works_at"]]
    finally:
        await k.graph_db.async_drop()
        await k.aclose()
        k.close()


async def test_pages_in_search_and_reserved_document_label(tmp_path):
    def handler(prompt, schema):
        return {"entities": [{"name": "NSCA 160-5", "type": "Document"}], "relations": []}

    pages = "\n\n".join(
        f"<!-- page-start: {n} -->\n# Section {n}\n\nText of page {n}." for n in range(1, 5)
    )
    pages = pages.replace("Text of page 4.", "The deadline is thirty days.")
    files = {"edital.md": pages, "plain.md": "# Plain\n\nNo page markers here.", "c.md": "# C\n\nc"}
    for fname, text in files.items():
        (tmp_path / fname).write_text(text)
    k = FalkorGraphKnowledge(
        graph_name=f"fa_test_{uuid.uuid4().hex[:8]}",
        model=FakeModel(handler),
        embedder=FakeEmbedder(),
        port=PORT,
    )
    try:
        await k.graph_db.async_create()
        for fname in files:
            await k.ainsert(path=str(tmp_path / fname))
        assert (await cypher(k, "MATCH (d:Document) RETURN count(d)"))[0][0] == 3
        assert (await cypher(k, "MATCH (e:Document_Entity) RETURN e.id")) == [
            ["nsca 160-5__Document_Entity"]
        ]

        hits = await k.graph_db.async_search("deadline thirty days", limit=10)
        top = next(d for d in hits if "deadline" in d.content)
        assert (top.meta_data["page_start"], top.meta_data["page_end"]) == (4, 4)
        plain = next(d for d in hits if "No page markers" in d.content)
        assert "page_start" not in plain.meta_data and "page_end" not in plain.meta_data
    finally:
        await k.graph_db.async_drop()
        await k.aclose()
        k.close()


async def test_delete_leaves_no_orphans(kb):
    await kb.graph_db.async_insert("h", DOCS)
    assert await kb.graph_db.async_delete_by_content_id("d1")
    assert (await cypher(kb, "MATCH (n) RETURN count(n)"))[0][0] == 0
    assert not await kb.graph_db.async_delete_by_content_id("d1")


async def test_revert_inverts_applied_changesets(kb):
    await kb.ontology.rename_entity_type("Person", "Human")
    await kb.ontology.add_entity_type(EntityType(label="City"))
    draft = await kb.ontology.revert_to(1)
    await kb.ontology.approve(draft.id, "me")
    await kb.ontology.apply(draft.id)
    o = await kb.ontology.get()
    assert {e.label for e in o.entity_types} == {"Person", "Org"}
    assert o.relation("WORKS_AT").patterns == [("Person", "Org")]


async def test_declared_ontology_conflicts_with_persisted_type():
    name = f"fa_test_{uuid.uuid4().hex[:8]}"
    first = Ontology(
        entity_types=[EntityType(label="P", attributes=[Attribute(name="age", type="INTEGER")])]
    )
    k1 = FalkorGraphKnowledge(
        graph_name=name,
        model=FakeModel(kg_handler),
        embedder=FakeEmbedder(),
        port=PORT,
        ontology=first,
    )
    await k1.graph_db.async_create()
    second = Ontology(
        entity_types=[EntityType(label="P", attributes=[Attribute(name="age", type="STRING")])]
    )
    k2 = FalkorGraphKnowledge(
        graph_name=name,
        model=FakeModel(kg_handler),
        embedder=FakeEmbedder(),
        port=PORT,
        ontology=second,
    )
    try:
        with pytest.raises(OntologyConflictError):
            await k2.ontology.ensure_initialized()
    finally:
        await k1.graph_db.async_drop()
        await k1.aclose()
        await k2.aclose()
        k1.close()
        k2.close()


async def test_discovery_creates_draft_then_applies():
    def handler(prompt, schema):
        if schema == "DocSummary":
            return {"main_entities": ["Ana"], "aboutness": "people at companies"}
        if schema == "ChunkProposal":
            return {
                "entity_types": [
                    {"label": "person", "attributes": [{"name": "age", "type": "integer"}]},
                    {"label": "Org"},
                ],
                "relation_types": [{"label": "works at", "patterns": [["person", "Org"]]}],
            }
        return {
            "entity_types": [],
            "relation_types": [],
        }  # normalisation returns junk -> still valid

    k = FalkorGraphKnowledge(
        graph_name=f"fa_test_{uuid.uuid4().hex[:8]}",
        model=FakeModel(handler),
        embedder=FakeEmbedder(),
        port=PORT,
        require_approval=True,
    )
    try:
        await k.graph_db.async_create()
        cs = await k.ontology.suggest_extensions(["Ana works at Acme. Ana is 31."])
        assert cs.status.value == "draft" and cs.source == "discovery" and cs.ops
        assert (await k.ontology.get()).is_open  # nothing applied yet
        await k.ontology.approve(cs.id, "me")
        await k.ontology.apply(cs.id)
        o = await k.ontology.get()
        assert o.entity("Person") and o.relation("WORKS_AT").patterns == [("Person", "Org")]
    finally:
        await k.graph_db.async_drop()
        await k.aclose()
        k.close()


async def test_agno_knowledge_pipeline_end_to_end(kb):
    """The real agno path: Reader + chunking + content hash, then agent-style retrieval."""
    await kb.ainsert(
        name="notes", text_content="Ana works at Acme. Ana is 31.", skip_if_exists=True
    )
    await kb.ainsert(
        name="notes", text_content="Ana works at Acme. Ana is 31.", skip_if_exists=True
    )  # dedup
    assert (await cypher(kb, "MATCH (d:Document) RETURN count(d)"))[0][0] == 1
    docs = await kb.asearch("Who works at Acme?")
    assert any("Ana works at Acme" in d.content for d in docs)
    # sync API must also work while an event loop is running (agno calls it from sync code paths)
    assert kb.retrieve("Acme")
    ctx = await kb.abuild_context()
    assert "Person" in ctx and "WORKS_AT" in ctx


def test_knowledge_is_served_by_agentos(tmp_path):
    """AgentOS only lists knowledge that is a real ``Knowledge`` with a contents_db."""
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("openai")
    from agno.agent import Agent
    from agno.db.sqlite import SqliteDb
    from agno.models.openai import OpenAIChat
    from agno.os import AgentOS
    from fastapi.testclient import TestClient

    k = FalkorGraphKnowledge(
        graph_name=f"fa_test_{uuid.uuid4().hex[:8]}",
        model=FakeModel(kg_handler),
        embedder=FakeEmbedder(),
        port=PORT,
        contents_db=SqliteDb(db_file=str(tmp_path / "c.db")),
    )
    try:
        agent = Agent(model=OpenAIChat(id="gpt-4o-mini", api_key="unused"), knowledge=k)
        with TestClient(AgentOS(agents=[agent]).get_app()) as c:
            assert c.get("/knowledge/config").status_code == 200
            assert c.post("/knowledge/search", json={"query": "Acme"}).status_code == 200
    finally:
        k.graph_db.drop()
        k.close()
