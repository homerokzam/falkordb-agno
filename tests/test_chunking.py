from pathlib import Path
from typing import Any

import pytest
from agno.knowledge.document import Document
from agno.knowledge.reader.markdown_reader import MarkdownReader
from agno.knowledge.reader.text_reader import TextReader

from falkordb_agno import FalkorGraphDb, FalkorGraphKnowledge, HeadingChunking
from tests.fakes import FakeEmbedder, FakeModel, kg_handler

MD = "# A\nalpha\n\n## B\nbeta\n\n### C\ngamma\n"


def test_one_chunk_per_heading():
    chunks = HeadingChunking().chunk(Document(name="doc", content=MD))
    assert [c.content.splitlines()[0] for c in chunks] == ["# A", "## B", "### C"]
    assert [c.meta_data["chunk"] for c in chunks] == [1, 2, 3]
    assert all(c.name == "doc" for c in chunks)


def test_preamble_before_first_heading_is_its_own_chunk():
    chunks = HeadingChunking().chunk(Document(name="d", content="intro\n\n# A\nalpha"))
    assert [c.content for c in chunks] == ["intro", "# A\nalpha"]


def test_hash_without_space_is_not_a_heading():
    chunks = HeadingChunking().chunk(Document(name="d", content="# A\n#tag inline\n#### D\nx"))
    assert len(chunks) == 2 and "#tag inline" in chunks[0].content


def test_large_section_splits_on_paragraphs_only():
    paras = [f"p{i:02d} " + "x" * 996 for i in range(12)]  # 12 paragraphs, 1000 chars each
    content = "# Big\n\n" + "\n\n".join(paras)
    chunks = HeadingChunking(chunk_size=5000).chunk(Document(name="d", content=content))
    assert len(chunks) >= 3
    pieces = [p for c in chunks for p in c.content.split("\n\n")]
    assert pieces == ["# Big", *paras]  # every paragraph intact, order kept
    assert all(len(c.content) <= 5000 for c in chunks)


def test_oversized_paragraph_is_kept_whole():
    huge = "y" * 7000
    chunks = HeadingChunking(chunk_size=5000).chunk(Document(name="d", content=f"# H\n\n{huge}"))
    assert any(c.content == huge for c in chunks)


def test_meta_data_is_kept():
    doc = Document(name="d", content=MD, meta_data={"lang": "pt"})
    assert all(c.meta_data["lang"] == "pt" for c in HeadingChunking().chunk(doc))


@pytest.mark.parametrize("content", ["", "   \n\n  "])
def test_empty_content_gives_no_chunks(content):
    assert HeadingChunking().chunk(Document(name="d", content=content)) == []


def test_ids_are_deterministic():
    doc = Document(name="d", content=MD)
    first = [c.id for c in HeadingChunking().chunk(doc)]
    assert first == [c.id for c in HeadingChunking().chunk(doc)]
    assert len(set(first)) == 3


def test_ids_are_deterministic_through_markdown_reader(tmp_path: Path):
    """MarkdownReader assigns a random uuid to every read; chunk ids must not depend on it."""
    f = tmp_path / "x.md"
    f.write_text(MD)
    reader = MarkdownReader(chunking_strategy=HeadingChunking())
    assert [d.id for d in reader.read(f)] == [d.id for d in reader.read(f)]


@pytest.fixture
def captured(monkeypatch) -> list[Document]:
    """FalkorGraphKnowledge with the graph store stubbed out: no FalkorDB, no LLM."""
    docs: list[Document] = []

    async def capture(self: Any, content_hash: str, documents: list[Document], *a: Any, **k: Any):
        docs.extend(documents)

    monkeypatch.setattr(FalkorGraphDb, "exists", lambda self: True)
    monkeypatch.setattr(FalkorGraphDb, "content_hash_exists", lambda self, *a, **k: False)
    monkeypatch.setattr(FalkorGraphDb, "async_delete_by_content_id", capture_none)
    for name in ("async_insert", "async_upsert"):
        monkeypatch.setattr(FalkorGraphDb, name, capture)
    for name in ("insert", "upsert"):
        monkeypatch.setattr(FalkorGraphDb, name, lambda self, h, d, *a, **k: docs.extend(d))
    return docs


async def capture_none(*a: Any, **k: Any) -> bool:
    return False


def make_kb(**kw: Any) -> FalkorGraphKnowledge:
    return FalkorGraphKnowledge(
        graph_name="unit", model=FakeModel(kg_handler), embedder=FakeEmbedder(), **kw
    )


@pytest.mark.parametrize("suffix", [".md", ".markdown"])
def test_insert_markdown_uses_heading_chunking_by_default(tmp_path, captured, suffix):
    f = tmp_path / f"notes{suffix}"
    f.write_text(MD)
    make_kb().insert(path=str(f))
    assert [d.meta_data["chunk"] for d in captured] == [1, 2, 3]


async def test_ainsert_markdown_uses_heading_chunking_by_default(tmp_path, captured):
    f = tmp_path / "notes.md"
    f.write_text(MD)
    await make_kb().ainsert(path=str(f))
    assert len(captured) == 3


def test_custom_chunking_parameter(tmp_path, captured):
    f = tmp_path / "notes.md"
    f.write_text(MD)
    make_kb(chunking=HeadingChunking(chunk_size=10)).insert(path=str(f))
    assert len(captured) >= 3


def test_explicit_reader_wins(tmp_path, captured):
    f = tmp_path / "notes.md"
    f.write_text(MD)
    make_kb().insert(path=str(f), reader=TextReader(chunk=False))
    assert len(captured) == 1 and "chunk" not in captured[0].meta_data


def test_positional_arguments_still_work(tmp_path, captured):
    f = tmp_path / "notes.md"
    f.write_text(MD)
    make_kb().insert("notes", None, str(f))  # name, description, path
    assert len(captured) == 3


# ---- source pages ---------------------------------------------------------------------------
PAGED = """<!-- page-start: 1 -->
# Edital

Intro on page one.

## Prazo

Prazo starts on page one.

<!-- page-start: 2 -->

and continues on page two.

<!-- page-start: 3 -->
## Anexo

Anexo on page three.
"""


def _pages(docs: list[Document]) -> list[tuple[int, int] | None]:
    return [
        (d.meta_data["page_start"], d.meta_data["page_end"])
        if "page_start" in d.meta_data
        else None
        for d in docs
    ]


def test_pages_follow_markers_across_headings():
    docs = HeadingChunking().chunk(Document(name="edital", content=PAGED))
    by_head = {d.content.split("\n")[0]: d for d in docs}
    # `## Anexo` starts right after marker 3, which ends the previous chunk.
    assert _pages([by_head["## Anexo"]]) == [(3, 3)]
    # Starts on page 1, crosses marker 2; the trailing marker 3 isn't content of this chunk.
    assert _pages([by_head["## Prazo"]]) == [(1, 2)]
    assert "<!-- page-start: 2 -->" in by_head["## Prazo"].content  # markers stay in the text


def test_paragraph_subchunks_get_their_own_start_page():
    docs = HeadingChunking(chunk_size=40).chunk(Document(name="edital", content=PAGED))
    pages = {d.content: _pages([d])[0] for d in docs}
    assert pages["## Prazo\n\nPrazo starts on page one."] == (1, 1)
    assert pages["and continues on page two."] == (2, 2)
    assert pages["## Anexo\n\nAnexo on page three."] == (3, 3)


def test_no_markers_no_page_keys():
    docs = HeadingChunking().chunk(Document(name="x", content=MD))
    assert docs and all(
        "page_start" not in d.meta_data and "page_end" not in d.meta_data for d in docs
    )


def test_chunk_ids_and_texts_unchanged_by_page_tracking():
    # Captured from the implementation before page tracking existed.
    expected = [
        ("edital_5ac86e13220b_1", "<!-- page-start: 1 -->"),
        ("edital_5ac86e13220b_2", "# Edital\n\nIntro on page one."),
        ("edital_5ac86e13220b_3", "## Prazo\n\nPrazo starts on page one."),
        ("edital_5ac86e13220b_4", "<!-- page-start: 2 -->"),
        ("edital_5ac86e13220b_5", "and continues on page two."),
        ("edital_5ac86e13220b_6", "<!-- page-start: 3 -->"),
        ("edital_5ac86e13220b_7", "## Anexo\n\nAnexo on page three."),
    ]
    docs = HeadingChunking(chunk_size=40).chunk(Document(name="edital", content=PAGED))
    assert [(d.id, d.content) for d in docs] == expected
    assert [d.id for d in HeadingChunking().chunk(Document(name="edital", content=PAGED))] == [
        f"edital_5ac86e13220b_{i}" for i in range(1, 5)
    ]
