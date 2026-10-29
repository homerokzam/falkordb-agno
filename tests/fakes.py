"""Deterministic fakes: no network, no LLM."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

DIM = 32


class FakeEmbedder:
    dimensions = DIM

    async def async_get_embedding(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        return v if any(v) else [1.0] + [0.0] * (DIM - 1)

    def get_embedding(self, text: str) -> list[float]:
        import asyncio

        return asyncio.run(self.async_get_embedding(text))


class FakeModel:
    """``handler(prompt, schema_name) -> dict`` produces the JSON the 'LLM' returns."""

    def __init__(self, handler: Callable[[str, str], dict[str, Any]]) -> None:
        self.handler = handler
        self.prompts: list[str] = []

    async def aresponse(self, messages: list[Any], response_format: Any = None, **_: Any) -> Any:
        prompt = messages[-1].content
        self.prompts.append(prompt)
        out = self.handler(prompt, getattr(response_format, "__name__", ""))
        return SimpleNamespace(content=json.dumps(out), parsed=None)


def kg_handler(prompt: str, schema: str) -> dict[str, Any]:
    """Extracts 'X works at Y' sentences; attribute backfill reads 'age N'."""
    text = prompt.split("<<<UNTRUSTED>>>")[1] if "<<<UNTRUSTED>>>" in prompt else prompt
    if schema == "_AttrValues":
        names = re.findall(r"Entities: \[(.*?)\]", prompt)
        out: list[dict[str, Any]] = []
        for n in re.findall(r"'([^']+)'", names[0] if names else ""):
            m = re.search(rf"{re.escape(n)} is (\d+)", text)
            if m:
                out.append({"name": n, "value": m.group(1)})
        return {"values": out}
    ents: dict[str, str] = {}
    rels = []
    for a, b in re.findall(r"([A-Z][a-z]+) works at ([A-Z][a-z]+)", text):
        ents[a], ents[b] = "Person", "Org"
        rels.append({"source": a, "target": b, "type": "WORKS_AT", "fact": f"{a} works at {b}"})
    for a in re.findall(r"([A-Z][a-z]+) lives in Rio", text):
        ents[a], ents["Rio"] = "Person", "City"
        rels.append({"source": a, "target": "Rio", "type": "LIVES_IN", "fact": f"{a} lives in Rio"})
    return {
        "entities": [
            {"name": n, "type": t, "description": f"{n} the {t}"} for n, t in ents.items()
        ],
        "relations": rels,
    }
