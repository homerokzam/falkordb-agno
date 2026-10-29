# falkordb-agno

GraphRAG over [FalkorDB](https://www.falkordb.com) as an [Agno](https://docs.agno.com) **Knowledge**,
with a full **Ontology Lifecycle**. Inspired by FalkorDB's GraphRAG-SDK; does not depend on it.
Runtime dependencies: `agno`, `falkordb`, `pydantic`.

```python
kb = FalkorGraphKnowledge(graph_name="team", model=OpenAIChat(id="gpt-4o-mini"),
                          embedder=OpenAIEmbedder(), ontology=my_ontology)
await kb.ainsert(name="notes", text_content="Ana works at Acme.")
agent = Agent(model=..., knowledge=kb, search_knowledge=True)
```

`FalkorGraphKnowledge` subclasses `agno.knowledge.Knowledge`, so it works with `Agent`, `Team` and
the AgentOS knowledge endpoints (give it a `contents_db`). Any agno `Model` and `Embedder` work.

## Chunking

You don't need to build a `Reader` or a chunking strategy. `kb.insert(path="notes.md")` (or
`.markdown`) produces **one chunk per heading section** via `HeadingChunking`. A section larger than
`chunk_size` (default 5000 chars) is split on blank-line paragraphs, never mid-paragraph; text before
the first heading is its own chunk. Chunk ids are deterministic (document name + content hash).
Without this, agno would read `.md` with its plain `TextReader`.

```python
kb = FalkorGraphKnowledge(..., chunking=HeadingChunking(chunk_size=3000))  # or any agno ChunkingStrategy
kb.insert(path="notes.md")                                      # uses kb.chunking
kb.insert(path="notes.md", reader=MarkdownReader(...))          # an explicit reader always wins
```

**Source pages.** Markdown converted from PDF may carry one `<!-- page-start: N -->` line per
page. `HeadingChunking` keeps the markers in the text and stores `page_start`/`page_end` (ints) in
each chunk's `meta_data`: the page the chunk's content starts on (the last marker before it, even
when that marker ended the previous chunk) and the last page it reaches. Markers at the very start
or end of a chunk aren't content, so a chunk ending with the marker that precedes the next heading
doesn't claim that page. Without markers the keys are absent.

Other formats keep agno's defaults. The default only applies to a single Markdown file passed as
`path`; directories, `text_content` and `insert_many` use agno's readers unchanged.

## Setup (uv only)

```bash
uv venv --python 3.14
uv pip install -r requirements-dev.txt        # library + tests
uv pip install -r requirements-examples.txt   # optional: AgentOS + OpenAI for the examples
docker compose up -d falkordb                 # :6379, browser UI on :3000
uv run pytest -q                              # unit tests (no DB, no LLM)
RUN_INTEGRATION=1 uv run pytest -q            # + integration tests (needs FalkorDB)
uv run ruff check falkordb_agno/ && uv run mypy falkordb_agno/
```

Use it from another project with `uv pip install -r <path>/requirements.txt` and put this folder on
`PYTHONPATH`. Python 3.14 (3.15 isn't supported yet: PyYAML, an agno dependency, has no wheel).

## Graph model

Data graph `<name>`: `Document <-PART_OF- Chunk -NEXT_CHUNK-> Chunk`; entities carry their type
label plus `__Entity__` and link to chunks via `MENTIONED_IN`; **every** relation is one
`RELATES` edge with `rel_type` and a `fact` string (so one vector index covers all relations, and
renaming a relation is a `SET`). Ontology graph `<name>__ontology`: current types, versions,
changesets and a lock.

Reserved labels: `Document`, `Chunk`, `OntologyVersion`, `ChangeSet`, `Lock`, `__Entity__`. An
extracted entity typed with one of them is renamed `<Label>_Entity` (`Document` →
`Document_Entity`), so `MATCH (d:Document)` only ever sees ingested files. A declared ontology using
one raises `OntologyValidationError`.

## Search results

Each chunk `Document` returned by `search`/`retrieve` keeps `content` as stored (page markers
included) and carries `meta_data`: `document_name`, `content_id`, `chunk_id`, `chunk_index`,
`page_start`/`page_end` (only when the chunk has them), `score`, `facts`. Citing pages is left to
whoever builds the agent prompt. Without filters/`user_id`, a first `graph_context` document lists
the related facts.

## Ontology Lifecycle

| Stage | What it does |
|---|---|
| **Define / persist / load** | `Ontology(entity_types, relation_types)` with typed attributes and directed relation patterns. Persisted on first use as v1. Re-declaring an ontology is additive; changing an attribute type raises `OntologyConflictError`. |
| **Constrain** | Extraction prompts list only declared types; pruning drops undeclared labels, relation types and endpoint pairs, and coerces attributes to their declared types. An empty ontology is *open* (nothing pruned). |
| **Propose** | `ontology.propose(ops)` creates a `draft` changeset. Ops: `add_*`, `rename_*`, `drop_*`, `set_description`, `backfill_*`. |
| **Validate** | `ontology.validate(cs)`: dry-run with impact estimates (entities deleted, LLM calls). No writes. |
| **Approve / reject** | `approve(id, reviewer)` / `reject(...)`. States: `draft -> approved -> applied`, `-> rejected`, `approved -> failed` (retryable). |
| **Apply** | Takes the lock, migrates the data graph op by op (idempotent Cypher; LLM backfills), writes the ontology, creates version N+1. A failure creates **no** version and leaves the changeset `failed` with the error. |
| **Version** | `history()`, `get_version(n)`, `diff(a, b)`. A changeset based on an old version raises `OntologyVersionConflictError`; `rebase(id)` re-points it. |
| **Revert** | `revert_to(n)` drafts the inverse of later changesets. Destructive ops (drops) cannot be reverted and raise. |
| **Discover** | `suggest_extensions(sources)` samples chunks, proposes types, normalises them, and returns a **draft** with additive ops only. |

One-op shortcuts (`rename_entity_type`, `add_attribute`, ...) apply immediately unless
`require_approval=True`, in which case they return a draft. Blocking versions: `ontology.sync.<method>`.

Backfills are idempotent: each processed chunk is marked in `Chunk.extracted_ops`, so a re-run
skips finished work.

## Agent tools

`OntologyTools(kb)` lets an agent read the ontology and **propose** changes (drafts only).
`allow_apply=True` adds approve/apply, each requiring Agno's human confirmation.

## Limitations (read before relying on it)

- Ingestion and ontology apply are serialised by an in-process lock; across processes only apply
  is protected (by the persisted lock). Don't ingest from one process while another applies.
- Data migrations are not transactional. A crash mid-apply leaves data ahead of the schema; re-running
  the changeset is safe because every step is idempotent.
- `filters` support equality on scalar metadata only. `user_id` scopes chunks, but entities and
  relations are shared across users, so facts are omitted from results when `filters`/`user_id` are set.
- Any agno model with structured output works (OpenAI, Anthropic, Gemini, xAI/Grok, ...). Every
  schema sent as `response_format` stays in the subset all of them accept: no open dicts
  (`dict[str, Any]` → `additionalProperties`, which Gemini rejects and strict modes silently
  close), no tuples (`prefixItems`), no `Any`. Attributes travel as `[{name, value}]` pairs.
- Entity and relation types returned by the LLM are normalised to Cypher identifiers right after
  extraction, keeping their case: `Hospital Service` → `Hospital_Service`, `Equipamento Médico` →
  `Equipamento_Medico`, `3D Model` → `_3D_Model`. Declared ontology labels are compared in the same
  normalised form, so a declared `Hospital_Service` accepts an extracted `Hospital Service`. If a
  whole label group still fails to write, it is logged on the `falkordb_agno` logger and the rest
  of the document is written.
- A chunk whose extraction fails is logged as a warning on the `falkordb_agno` logger and
  skipped. If every chunk of a document fails, `FalkorGraphDb.async_insert` raises
  `ExtractionError` and writes nothing; through `Knowledge.ainsert` agno logs it and marks the
  content `FAILED`.
- The examples need an `OPENAI_API_KEY`; the test-suite uses deterministic fakes.
