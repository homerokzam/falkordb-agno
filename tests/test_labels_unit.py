"""LLM-proposed type labels become Cypher identifiers. No DB, no LLM."""

import logging
from typing import Any

import pytest

from falkordb_agno import EntityType, Ontology, OntologyValidationError
from falkordb_agno.extraction.extractor import GraphExtractor
from falkordb_agno.extraction.pruning import prune
from falkordb_agno.extraction.resolution import ResolvedEntity, ResolvedRelation, resolve
from falkordb_agno.llm import StructuredLLM
from falkordb_agno.store import writer
from falkordb_agno.store.indexes import normalize_label, safe_label
from tests.fakes import FakeModel


@pytest.mark.parametrize(
    "raw, label",
    [
        ("Hospital Service", "Hospital_Service"),
        ("Equipamento Médico", "Equipamento_Medico"),
        ("3D Model", "_3D_Model"),
        ("  ", "Entity"),
        ("Person", "Person"),
        ("A-B/C", "A_B_C"),
        ("_Private", "_Private"),  # already valid: untouched
    ],
)
def test_normalize_label(raw, label):
    assert normalize_label(raw) == label
    assert safe_label(label) == label


@pytest.mark.parametrize("bad", ["Hospital Service", "Person\n", "a`b", ""])
def test_safe_label_still_rejects_invalid_identifiers(bad):
    with pytest.raises(ValueError):
        safe_label(bad)


def _extractor(types_by_chunk: dict[str, str]) -> GraphExtractor:
    """Each chunk text names the entity type the 'LLM' returns for 'Unimed'."""

    def handler(prompt: str, schema: str) -> dict[str, Any]:
        text = prompt.split("<<<UNTRUSTED>>>")[1].strip()
        return {
            "entities": [{"name": "Unimed", "type": types_by_chunk[text]}],
            "relations": [{"source": "Unimed", "target": "Unimed", "type": "part of"}],
        }

    return GraphExtractor(StructuredLLM(FakeModel(handler)), max_retries=1)


async def test_spaced_and_underscored_types_resolve_to_one_entity():
    ex = _extractor({"a": "Hospital Service", "b": "Hospital_Service"})
    open_onto = Ontology()
    per_chunk = [(c, prune(await ex.extract(open_onto, c), open_onto)) for c in ("a", "b")]
    ents, rels = resolve(per_chunk)
    assert list(ents) == ["unimed__Hospital_Service"]
    assert ents["unimed__Hospital_Service"].chunk_ids == {"a", "b"}
    assert {r.rel_type for r in rels.values()} == {"part_of"}


async def test_declared_type_matches_normalised_extraction():
    onto = Ontology(entity_types=[EntityType(label="Hospital_Service")])
    g = prune(await _extractor({"a": "Hospital Service"}).extract(onto, "a"), onto)
    assert [(e.name, e.type) for e in g.entities] == [("Unimed", "Hospital_Service")]


class _Conn:
    """Records queries; fails any query whose text contains ``fail_on``."""

    def __init__(self, fail_on: str) -> None:
        self.fail_on, self.queries = fail_on, []

    async def query(self, graph: str, cypher: str, params: Any = None, **_: Any) -> list[Any]:
        if self.fail_on in cypher or self.fail_on in str(params):
            raise RuntimeError(f"boom on {self.fail_on}")
        self.queries.append((cypher, params))
        return []


async def test_failed_entity_group_is_logged_and_others_written(caplog):
    ents = {
        "a__Bad Label": ResolvedEntity("a__Bad Label", "a", "Bad Label", chunk_ids={"c"}),
        "b__Person": ResolvedEntity("b__Person", "b", "Person", chunk_ids={"c"}),
    }
    conn = _Conn(fail_on="never")
    with caplog.at_level(logging.WARNING, logger="falkordb_agno"):
        await writer.write_entities(conn, "g", ents, {k: [0.0] for k in ents})  # type: ignore[arg-type]
    assert any("Bad Label" in r.getMessage() for r in caplog.records)
    assert any(":__Entity__:Person " in q for q, _ in conn.queries)


async def test_failed_relation_group_is_logged_and_others_written(caplog):
    rels = [
        ResolvedRelation("a", "b", "WORKS_AT", "a works at b", ""),
        ResolvedRelation("a", "b", "BROKEN", "a broken b", ""),
    ]
    conn = _Conn(fail_on="BROKEN")
    with caplog.at_level(logging.WARNING, logger="falkordb_agno"):
        await writer.write_relations(conn, "g", rels, [[0.0], [0.0]])  # type: ignore[arg-type]
    assert any("BROKEN" in r.getMessage() for r in caplog.records)
    ((_, params),) = conn.queries
    assert [r["rt"] for r in params["rows"]] == ["WORKS_AT"]


# ---- reserved labels ------------------------------------------------------------------------
async def test_reserved_entity_type_is_renamed_before_resolution():
    ex = _extractor({"a": "Document"})
    ents, _ = resolve([("a", prune(await ex.extract(Ontology(), "a"), Ontology()))])
    assert list(ents) == ["unimed__Document_Entity"]


@pytest.mark.parametrize("label", ["Chunk", "Document", "ChangeSet", "OntologyVersion", "Lock"])
def test_declared_reserved_label_is_rejected(label):
    with pytest.raises(OntologyValidationError) as e:
        Ontology(entity_types=[EntityType(label=label)]).validate_ontology()
    assert f"{label}_Entity" in str(e.value)


async def test_ensure_initialized_rejects_reserved_declared_label():
    from falkordb_agno import FalkorGraphDb
    from tests.fakes import FakeEmbedder

    db = FalkorGraphDb(
        graph_name="unit",
        model=FakeModel(lambda p, s: {}),
        embedder=FakeEmbedder(),
        ontology=Ontology(entity_types=[EntityType(label="Chunk")]),
    )
    with pytest.raises(OntologyValidationError):
        await db.manager.ensure_initialized()  # raises before touching the DB
