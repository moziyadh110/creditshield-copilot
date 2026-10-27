"""The agent's three tools, exposed to the LLM through function calling."""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import mlflow
from mlflow.entities import SpanType

from app.agent.template import template_for_agent
from app.db.models import Chunk
from app.repositories.reason_codes import ReasonCatalog

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_reason_codes",
            "description": "Approved wording and banned wordings for the given reason features.",
            "parameters": {
                "type": "object",
                "properties": {"features": {"type": "array", "items": {"type": "string"}}},
                "required": ["features"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_regulation",
            "description": "Search Regulation B (12 CFR 1002.9 and Appendix C sample forms). "
            "Returns the 3 most relevant sections with their citations.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_notice_template",
            "description": "The fixed notice letter, showing where your paragraph goes.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class Toolbox:
    def __init__(self, catalog: ReasonCatalog, search: Callable[[str], list[Chunk]]):
        self.catalog = catalog
        self.search = search
        self.retrieved_ids: list[str] = []

    def get_reason_codes(self, features: list[str]) -> dict[str, Any]:
        out = {}
        for f in features:
            rc = self.catalog.features.get(f)
            out[f] = (
                {"code": rc.code, "approved_wording": rc.reason, "banned_wordings": rc.banned}
                if rc
                else {"error": "unknown feature"}
            )
        return out

    async def search_regulation(self, query: str) -> list[dict[str, str]]:
        with mlflow.start_span(
            name="retrieval.search_regulation", span_type=SpanType.RETRIEVER
        ) as span:
            span.set_inputs({"query": query})
            chunks = await asyncio.to_thread(self.search, query)
            span.set_outputs([{"section": c.section_id, "source": c.source_url} for c in chunks])
        self.retrieved_ids += [
            c.section_id for c in chunks if c.section_id not in self.retrieved_ids
        ]
        return [{"section": c.section_id, "text": c.text} for c in chunks]

    async def call(self, name: str, arguments: str) -> str:
        """Run one tool call; errors go back to the model as data instead of crashing the run."""
        with mlflow.start_span(name=f"tool.{name}", span_type=SpanType.TOOL) as span:
            span.set_inputs({"arguments": arguments})
            try:
                args = json.loads(arguments or "{}")
                if name == "get_reason_codes":
                    result: Any = self.get_reason_codes(list(args["features"]))
                elif name == "search_regulation":
                    result = await self.search_regulation(str(args["query"]))
                elif name == "get_notice_template":
                    result = template_for_agent()
                else:
                    result = {"error": f"unknown tool {name}"}
            except (KeyError, TypeError, ValueError) as e:
                result = {"error": f"bad arguments: {e}"}
            span.set_outputs(result)
        return result if isinstance(result, str) else json.dumps(result)
