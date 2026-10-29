"""Structured LLM calls on top of an agno ``Model``."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from agno.models.message import Message
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class StructuredOutputError(RuntimeError):
    def __init__(self, attempts: int, last_error: str) -> None:
        super().__init__(f"no valid structured output after {attempts} attempts: {last_error}")
        self.attempts, self.last_error = attempts, last_error


def _parse[M: BaseModel](schema: type[M], resp: Any) -> M:
    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, schema):
        return parsed
    content = getattr(resp, "content", None)
    if isinstance(content, schema):
        return content
    if isinstance(content, dict):
        return schema.model_validate(content)
    text = _FENCE.sub("", (content or "").strip())
    return schema.model_validate_json(text)


class StructuredLLM:
    """``extract(prompt, schema)`` -> validated pydantic object.

    On invalid output the validation errors are sent back to the model and the call is
    retried, up to ``max_retries`` times.
    """

    def __init__(self, model: Any, system: str | None = None) -> None:
        self.model = model
        self.system = system
        self.calls = 0

    async def extract(
        self, prompt: str, schema: type[T], *, max_retries: int = 3, system: str | None = None
    ) -> T:
        messages: list[Message] = []
        if system or self.system:
            messages.append(Message(role="system", content=system or self.system))
        messages.append(
            Message(
                role="user",
                content=f"{prompt}\n\nReturn ONLY JSON matching this schema:\n"
                f"{json.dumps(schema.model_json_schema())}",
            )
        )
        last = ""
        for attempt in range(1, max_retries + 1):
            self.calls += 1
            resp = await self.model.aresponse(messages=list(messages), response_format=schema)
            try:
                return _parse(schema, resp)
            except (ValidationError, ValueError) as e:
                last = str(e)[:800]
                messages.append(
                    Message(role="assistant", content=str(getattr(resp, "content", "")))
                )
                messages.append(
                    Message(
                        role="user",
                        content=f"That output was invalid: {last}\nFix it and return only JSON.",
                    )
                )
        raise StructuredOutputError(max_retries, last)
