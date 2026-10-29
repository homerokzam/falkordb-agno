"""Default chunking: one chunk per Markdown heading section."""

from __future__ import annotations

import bisect
import hashlib
import re
from dataclasses import replace
from typing import Any

from agno.knowledge.chunking.strategy import ChunkingStrategy
from agno.knowledge.document import Document

_HEADING = re.compile(r"(?m)^(?=#{1,6} )")
# One per page in Markdown converted from PDF; kept in the chunk text.
PAGE_MARKER = re.compile(r"<!--\s*page-start:\s*(\d+)\s*-->")
_LEADING_MARKERS = re.compile(rf"(?:\s*{PAGE_MARKER.pattern})*\s*")
_TRAILING_MARKERS = re.compile(rf"(?:{PAGE_MARKER.pattern}\s*)*$")


class HeadingChunking(ChunkingStrategy):
    """Split before every ``#``..``######`` heading; oversized sections split on paragraphs.

    A paragraph longer than ``chunk_size`` is kept whole rather than cut mid-sentence.
    Uses no extra dependency (agno's ``MarkdownChunking`` needs ``unstructured``).
    """

    def __init__(self, chunk_size: int = 5000) -> None:
        self.chunk_size = chunk_size

    def chunk(self, document: Document) -> list[Document]:
        content = document.content or ""
        bounds = [0, *(m.start() for m in _HEADING.finditer(content) if m.start()), len(content)]
        pieces: list[tuple[str, int, int]] = []  # (text, start, end) offsets in `content`
        for a, b in zip(bounds, bounds[1:], strict=False):
            pieces.extend(self._fit(content[a:b], a))
        markers = [(m.start(), int(m.group(1))) for m in PAGE_MARKER.finditer(content)]
        # agno's MarkdownReader gives every read a random uuid as document id, so ids are
        # derived from name + content instead: the same file always yields the same chunk ids.
        digest = hashlib.md5(content.encode("utf-8")).hexdigest()[:12]  # nosec B324
        stable = replace(document, id=f"{document.name or 'doc'}_{digest}")
        return [
            Document(
                id=self._generate_chunk_id(stable, i, text),
                name=document.name,
                meta_data={**document.meta_data, "chunk": i, **_pages(markers, text, start)},
                content=text,
            )
            for i, (text, start, _end) in enumerate(pieces, start=1)
        ]

    def _fit(self, section: str, offset: int) -> list[tuple[str, int, int]]:
        text = section.strip()
        if not text:
            return []
        start = offset + len(section) - len(section.lstrip())
        if len(text) <= self.chunk_size:
            return [(text, start, start + len(text))]
        out: list[tuple[str, int, int]] = []
        buf, buf_start, buf_end = "", 0, 0
        pos = start
        for part in text.split("\n\n"):
            para, para_start = part.strip(), pos + len(part) - len(part.lstrip())
            pos += len(part) + 2
            if not para:
                continue
            if buf and len(buf) + 2 + len(para) > self.chunk_size:
                out.append((buf, buf_start, buf_end))
                buf = ""
            if not buf:
                buf_start = para_start
            buf = f"{buf}\n\n{para}" if buf else para
            buf_end = para_start + len(para)
        if buf:
            out.append((buf, buf_start, buf_end))
        return out


def _pages(markers: list[tuple[int, int]], text: str, start: int) -> dict[str, Any]:
    """``page_start``/``page_end`` of a chunk at ``start`` in the document; ``{}`` if unknown.

    Markers at the very start or end of the chunk aren't content: a chunk ending with the
    marker that precedes the next heading doesn't reach that page.
    """
    if not markers:
        return {}
    positions = [p for p, _ in markers]
    lead = _LEADING_MARKERS.match(text)
    trail = _TRAILING_MARKERS.search(text)
    c_start = start + (lead.end() if lead else 0)
    c_end = start + (trail.start() if trail else len(text))
    if c_end <= c_start:  # only markers: the page they open
        c_start = c_end = start + len(text)
    before = bisect.bisect_left(positions, c_start)
    inside = [n for p, n in markers[before : bisect.bisect_left(positions, c_end)]]
    first = markers[before - 1][1] if before else (inside[0] if inside else None)
    if first is None:
        return {}
    return {"page_start": first, "page_end": inside[-1] if inside else first}
