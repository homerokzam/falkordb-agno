"""Provider-portable structured-output schemas and visible extraction failures. No DB, no LLM."""

import json
import logging
from typing import Any

import pytest
from agno.knowledge.document import Document
from pydantic import BaseModel

from falkordb_agno import ExtractionError, FalkorGraphDb, Ontology, RelationType
from falkordb_agno.extraction.extractor import ExtractedEntity, ExtractedGraph
from falkordb_agno.extraction.pruning import coerce
from falkordb_agno.ontology.backfill import _AttrValues
from falkordb_agno.ontology.discovery import ChunkProposal, DocSummary
from falkordb_agno.store import writer
from tests.fakes import FakeEmbedder, FakeModel, kg_handler

# Every model passed to StructuredLLM.extract (i.e. sent as `response_format`).
RESPONSE_MODELS: list[type[BaseModel]] = [
    ExtractedGraph,
    _AttrValues,
    DocSummary,
    ChunkProposal,
    Ontology,
]


def _untyped(node: Any, path: str = "$") -> list[str]:
    """Paths of sub-schemas with no type (`Any`), which strict providers reject."""
    out: list[str] = []
    if isinstance(node, dict):
        for key in ("properties", "$defs"):
            for name, sub in node.get(key, {}).items():
                if not any(k in sub for k in ("type", "$ref", "anyOf", "enum", "const")):
                    out.append(f"{path}.{name}")
                out += _untyped(sub, f"{path}.{name}")
        for key in ("items", "anyOf"):
            subs = node.get(key)
            for sub in subs if isinstance(subs, list) else [subs] if subs else []:
                out += _untyped(sub, f"{path}[]")
    return out


@pytest.mark.parametrize("model", RESPONSE_MODELS, ids=lambda m: m.__name__)
def test_response_schema_is_portable_across_providers(model):
    schema = model.model_json_schema()
    text = json.dumps(schema)
    assert "additionalProperties" not in text  # Gemini 400s; OpenAI/xAI/Anthropic close it
    assert "prefixItems" not in text  # tuples
    assert _untyped(schema) == []


def test_attributes_list_becomes_dict_last_wins():
    e = ExtractedEntity.model_validate(
        {
            "name": "Ana",
            "type": "Person",
            "attributes": [
                {"name": "age", "value": 30},
                {"name": "city", "value": "Rio"},
                {"name": "age", "value": "31"},
            ],
        }
    )
    assert e.attributes == {"age": "31", "city": "Rio"}
    # Python callers and non-strict models may still pass a dict.
    assert ExtractedEntity(name="Ana", type="Person", attributes={"x": 1}).attributes == {"x": 1}


def test_patterns_accept_wire_objects_and_stored_pairs():
    wire = RelationType.model_validate(
        {"label": "WORKS_AT", "patterns": [{"source": "Person", "target": "Org"}]}
    )
    stored = RelationType.model_validate({"label": "WORKS_AT", "patterns": [["Person", "Org"]]})
    assert wire.patterns == stored.patterns == [("Person", "Org")]
    assert json.loads(wire.model_dump_json())["patterns"] == [["Person", "Org"]]


def test_backfill_values_from_pairs():
    out = _AttrValues.model_validate({"values": [{"name": "Ana", "value": "31"}]})
    assert out.values == {"Ana": "31"}


def test_list_attribute_from_json_string():
    assert coerce('["a", "b"]', "LIST") == ["a", "b"]
    assert coerce("[broken", "LIST") is None


# ---- extraction failures ------------------------------------------------------------------
@pytest.fixture
def db(monkeypatch):
    """A FalkorGraphDb whose DB writes are recorded instead of executed."""
    d = FalkorGraphDb(graph_name="unit", model=FakeModel(kg_handler), embedder=FakeEmbedder())
    d.extraction.max_retries = 1
    writes: dict[str, list[Any]] = {"document": [], "entities": [], "relations": []}

    async def noop() -> None:
        return None

    def record(kind: str):
        async def fn(*args: Any, **kwargs: Any) -> None:
            writes[kind].append((args, kwargs))

        return fn

    monkeypatch.setattr(d.manager, "ensure_initialized", noop)
    monkeypatch.setattr(writer, "write_document", record("document"))
    monkeypatch.setattr(writer, "write_entities", record("entities"))
    monkeypatch.setattr(writer, "write_relations", record("relations"))
    d.writes = writes  # type: ignore[attr-defined]
    return d


DOCS = [
    Document(id="c0", name="d1", content_id="d1", content="Ana works at Acme."),
    Document(id="c1", name="d1", content_id="d1", content="Bob works at Acme."),
]


async def test_all_chunks_failing_raises_and_writes_nothing(db):
    def boom(prompt, schema):
        raise RuntimeError("additionalProperties is not supported")

    db.model.handler = boom
    with pytest.raises(ExtractionError) as exc:
        await db.async_insert("h", DOCS)
    assert set(exc.value.errors) == {"c0", "c1"}
    assert "additionalProperties" in str(exc.value)
    assert db.writes == {"document": [], "entities": [], "relations": []}


async def test_one_failing_chunk_is_logged_and_document_is_written(db, caplog):
    def flaky(prompt, schema):
        if "Bob" in prompt:
            raise RuntimeError("rate limited")
        return kg_handler(prompt, schema)

    db.model.handler = flaky
    with caplog.at_level(logging.WARNING, logger="falkordb_agno"):
        await db.async_insert("h", DOCS)
    assert any("c1" in r.getMessage() and "rate limited" in r.getMessage() for r in caplog.records)
    assert len(db.writes["document"]) == 1
    ((args, _),) = db.writes["entities"]
    assert {e.name for e in args[2].values()} == {"Ana", "Acme"}
