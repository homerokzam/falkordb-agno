"""Index management for the data graph."""

from __future__ import annotations

import re
import unicodedata

from .connection import FalkorConnection

_LABEL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Structural labels of the data and ontology graphs; entities may never use them.
RESERVED_LABELS = frozenset(
    {"Document", "Chunk", "OntologyVersion", "ChangeSet", "Lock", "__Entity__"}
)


def safe_label(label: str) -> str:
    """Labels/rel types can't be Cypher parameters; only allow plain identifiers."""
    if not _LABEL.fullmatch(label):
        raise ValueError(f"invalid identifier for Cypher: {label!r}")
    return label


def normalize_label(raw: str) -> str:
    """Turn an LLM-proposed type into an identifier ``safe_label`` accepts, keeping its case:
    ``Hospital Service`` -> ``Hospital_Service``. Valid labels are returned unchanged."""
    if _LABEL.fullmatch(raw):
        return raw
    plain = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
    out = re.sub(r"[^A-Za-z0-9]+", "_", plain).strip("_")
    if not out:
        return "Entity"
    return f"_{out}" if out[0].isdigit() else out


def entity_label(raw: str) -> str:
    """``normalize_label`` plus renaming reserved labels: ``Document`` -> ``Document_Entity``."""
    label = normalize_label(raw)
    return f"{label}_Entity" if label in RESERVED_LABELS else label


async def ensure_indexes(conn: FalkorConnection, graph: str, dim: int) -> None:
    stmts = [
        "CREATE VECTOR INDEX FOR (c:Chunk) ON (c.embedding) "
        f"OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}",
        "CREATE VECTOR INDEX FOR (e:__Entity__) ON (e.embedding) "
        f"OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}",
        "CREATE VECTOR INDEX FOR ()-[r:RELATES]-() ON (r.embedding) "
        f"OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}",
        "CREATE INDEX FOR (c:Chunk) ON (c.id)",
        "CREATE INDEX FOR (c:Chunk) ON (c.content_id)",
        "CREATE INDEX FOR (d:Document) ON (d.content_id)",
        "CREATE INDEX FOR (e:__Entity__) ON (e.id)",
        "CALL db.idx.fulltext.createNodeIndex('__Entity__', 'name', 'description')",
    ]
    for s in stmts:
        try:
            await conn.query(graph, s)
        except Exception as e:  # noqa: BLE001 - "already indexed" is expected on re-run
            if "already" not in str(e).lower():
                raise
