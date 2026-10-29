import pytest

from falkordb_agno import (
    Attribute,
    ChangeSet,
    ChangeSetStateError,
    ChangeSetStatus,
    EntityType,
    Ontology,
    OntologyValidationError,
    RelationType,
)
from falkordb_agno.extraction.extractor import ExtractedEntity, ExtractedGraph, ExtractedRelation
from falkordb_agno.extraction.pruning import coerce, prune
from falkordb_agno.extraction.resolution import resolve
from falkordb_agno.ontology.changes import (
    AddAttribute,
    AddEntityType,
    DropEntityType,
    RenameEntityType,
    apply_ops,
    diff_ontologies,
    is_additive,
)
from falkordb_agno.ontology.discovery import sanitize, split_text
from falkordb_agno.ontology.manager import _invert


@pytest.fixture
def onto() -> Ontology:
    return Ontology(
        entity_types=[
            EntityType(label="Person", attributes=[Attribute(name="age", type="integer")]),
            EntityType(label="Org"),
        ],
        relation_types=[RelationType(label="WORKS_AT", patterns=[("Person", "Org")])],
    )


def test_validate_rejects_reserved_and_dangling(onto):
    onto.validate_ontology()
    bad = onto.model_copy(deep=True)
    bad.entity_types[0].attributes.append(Attribute(name="id"))
    bad.relation_types[0].patterns.append(("Person", "Ghost"))
    with pytest.raises(OntologyValidationError) as e:
        bad.validate_ontology()
    assert "reserved" in str(e.value) and "Ghost" in str(e.value)


def test_attribute_type_is_normalised(onto):
    assert onto.entity("Person").attribute("age").type == "INTEGER"


def test_merge_is_union_and_other_wins(onto):
    other = Ontology(
        entity_types=[
            EntityType(label="Person", description="a human", attributes=[Attribute(name="email")]),
            EntityType(label="City"),
        ]
    )
    m = onto.merge(other)
    assert {e.label for e in m.entity_types} == {"Person", "Org", "City"}
    assert {a.name for a in m.entity("Person").attributes} == {"age", "email"}
    assert m.entity("Person").description == "a human"


def test_diff_then_apply_roundtrip(onto):
    target = onto.merge(Ontology(entity_types=[EntityType(label="City")]))
    ops = diff_ontologies(onto, target)
    assert is_additive(ops)
    assert apply_ops(onto, ops).digest() == target.digest()


def test_rename_entity_updates_patterns(onto):
    out = apply_ops(onto, [RenameEntityType(old="Person", new="Human")])
    assert out.relation("WORKS_AT").patterns == [("Human", "Org")]
    assert onto.entity("Person") is not None  # input untouched


def test_drop_entity_removes_dependent_patterns(onto):
    out = apply_ops(onto, [DropEntityType(label="Org")])
    assert out.relation("WORKS_AT").patterns == []
    assert not is_additive([DropEntityType(label="Org")])


def test_invalid_ops_raise(onto):
    with pytest.raises(OntologyValidationError):
        apply_ops(onto, [AddEntityType(entity=EntityType(label="Org"))])
    with pytest.raises(OntologyValidationError):
        apply_ops(onto, [AddAttribute(owner="Person", attribute=Attribute(name="age"))])
    with pytest.raises(OntologyValidationError):
        apply_ops(onto, [DropEntityType(label="Nope")])


def test_changeset_state_machine():
    cs = ChangeSet(base_version=1, ops=[])
    with pytest.raises(ChangeSetStateError):
        cs.transition(ChangeSetStatus.APPLIED)  # draft cannot jump to applied
    cs.transition(ChangeSetStatus.APPROVED)
    cs.transition(ChangeSetStatus.FAILED)
    cs.transition(ChangeSetStatus.APPLIED)  # failed changesets can be retried
    with pytest.raises(ChangeSetStateError):
        cs.transition(ChangeSetStatus.REJECTED)


def test_changeset_parses_ops_from_json():
    cs = ChangeSet.model_validate(
        {"base_version": 1, "ops": [{"op": "drop_entity_type", "label": "X"}]}
    )
    assert isinstance(cs.ops[0], DropEntityType)
    with pytest.raises(ValueError):
        ChangeSet.model_validate({"base_version": 1, "ops": [{"op": "explode"}]})


def test_invert(onto):
    assert _invert(RenameEntityType(old="A", new="B"), onto).new == "A"
    with pytest.raises(OntologyValidationError):
        _invert(DropEntityType(label="Org"), onto)


def test_coerce():
    assert coerce("31", "INTEGER") == 31
    assert coerce("x", "INTEGER") is None
    assert coerce(True, "INTEGER") is None
    assert coerce("yes", "BOOLEAN") is True
    assert coerce("a", "LIST") == ["a"]


def test_prune_enforces_ontology(onto):
    g = ExtractedGraph(
        entities=[
            ExtractedEntity(name="Ana", type="Person", attributes={"age": "31", "x": 1}),
            ExtractedEntity(name="Acme", type="Org"),
            ExtractedEntity(name="Rio", type="City"),
        ],
        relations=[
            ExtractedRelation(source="Ana", target="Acme", type="WORKS_AT"),
            ExtractedRelation(source="Acme", target="Ana", type="WORKS_AT"),  # wrong direction
            ExtractedRelation(source="Ana", target="Rio", type="LIVES_IN"),
        ],
    )
    p = prune(g, onto)
    assert [e.name for e in p.entities] == ["Ana", "Acme"]
    assert p.entities[0].attributes == {"age": 31}
    assert [(r.source, r.target) for r in p.relations] == [("Ana", "Acme")]


def test_open_ontology_prunes_nothing():
    g = ExtractedGraph(entities=[ExtractedEntity(name="Rio", type="City")])
    assert prune(g, Ontology()) is g


def test_resolve_merges_across_chunks():
    g = ExtractedGraph(
        entities=[
            ExtractedEntity(name="Ana", type="Person"),
            ExtractedEntity(name="Acme", type="Org"),
        ],
        relations=[ExtractedRelation(source="Ana", target="Acme", type="WORKS_AT", fact="f")],
    )
    ents, rels = resolve([("c1", g), ("c2", g)])
    assert len(ents) == 2 and len(rels) == 1
    assert next(iter(rels.values())).chunk_ids == {"c1", "c2"}
    assert ents["ana__Person"].chunk_ids == {"c1", "c2"}


def test_sanitize_llm_proposal():
    raw = Ontology(
        entity_types=[
            EntityType(
                label="credit card", attributes=[Attribute(name="id"), Attribute(name="limit")]
            ),
            EntityType(label="Chunk"),
        ],
        relation_types=[RelationType(label="paid by", patterns=[("credit card", "Ghost")])],
    )
    out = sanitize(raw)
    assert [e.label for e in out.entity_types] == ["CreditCard"]
    assert [a.name for a in out.entity_types[0].attributes] == ["limit"]
    assert out.relation_types[0].label == "PAID_BY" and out.relation_types[0].patterns == []


def test_split_text_respects_max():
    parts = split_text("\n\n".join(["x" * 900] * 5), max_chars=2000)
    assert len(parts) >= 2 and all(len(p) <= 2000 for p in parts)
