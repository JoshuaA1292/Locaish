"""Meta's models as a Google ADK model, so the agents run unchanged on Muse.

The scout and the coverage planner are written against the Agent Development
Kit: tools declared as Python functions or an MCP toolset, a loop that turns
function calls into tool runs and feeds the results back. None of that is
Gemini-specific, but the wire format is: ADK speaks google.genai `Content`
and `Part`, and the Meta Model API speaks OpenAI Chat Completions. This
module is the translation, both ways, as a `BaseLlm` the agents take in
place of a model id. It is the same conversion ADK's own `LiteLlm` does,
done directly over `MetaClient` so the agent has one HTTP path, one retry
policy and one error type, and no second provider library in the bundle.

What crosses the boundary and how:

- the system instruction becomes the first `system` message; each `Content`
  becomes `user`, `assistant` (text plus `tool_calls`) or one `tool` message
  per function response, keyed by the call id the API gave us. ADK keeps
  the ids on its `FunctionCall`/`FunctionResponse` parts, so a result maps
  back to its call without any bookkeeping here; when an id is missing the
  response is paired with the earliest unanswered call of the same name.
- inline images become `image_url` data URLs; inline text becomes text.
- function declarations (hand-written functions, MCP tools) become OpenAI
  tool schemas. ADK already emits JSON schema for most of them; the genai
  `Schema` form is lowered (upper-case types, `nullable`, `any_of`) when it
  appears.
- an output schema becomes `response_format: json_schema`, with the
  `json_object` fallback the client uses elsewhere, because some ids reject
  the stricter form.

The answer comes back as text parts and `function_call` parts with the API's
ids preserved, plus token usage when the API reports it. Streaming is
answered with the complete turn as a single final response: the studio
reads final events only, and partial text would buy nothing but complexity.
`tool_choice` is never sent except as `auto`, which is all the API accepts.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import BaseModel, Field

from .client import ChatResult, MetaClient, MetaError

PROVIDER = "meta"


# ---------------------------------------------------------------------------
# genai -> OpenAI: messages
# ---------------------------------------------------------------------------


def _system_text(config: types.GenerateContentConfig | None) -> str | None:
    si = getattr(config, "system_instruction", None)
    if si is None:
        return None
    if isinstance(si, str):
        return si or None
    if isinstance(si, types.Content):
        texts = [p.text for p in (si.parts or []) if p.text]
        return "\n".join(texts) or None
    if isinstance(si, list):
        texts = []
        for item in si:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, types.Part) and item.text:
                texts.append(item.text)
        return "\n".join(texts) or None
    return None


def _content_parts(parts: list[types.Part]) -> str | list[dict[str, Any]]:
    """User-side parts as OpenAI content: a bare string when it is only text."""
    out: list[dict[str, Any]] = []
    for p in parts:
        if p.text:
            out.append({"type": "text", "text": p.text})
        elif p.inline_data is not None and p.inline_data.data:
            mime = (p.inline_data.mime_type or "application/octet-stream").split(";")[0].strip().lower()
            if mime.startswith("text/"):
                out.append({"type": "text", "text": p.inline_data.data.decode("utf-8", errors="replace")})
            elif mime.startswith("image/"):
                b64 = base64.b64encode(p.inline_data.data).decode("ascii")
                out.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
            else:
                raise ValueError(f"MetaLlm cannot send an inline part of type {mime}")
        elif p.file_data is not None and p.file_data.file_uri:
            mime = (p.file_data.mime_type or "").lower()
            if mime.startswith("image/") and p.file_data.file_uri.startswith(("http://", "https://")):
                out.append({"type": "image_url", "image_url": {"url": p.file_data.file_uri}})
            else:
                raise ValueError(f"MetaLlm cannot send a file reference ({p.file_data.file_uri})")
        # thought parts, executable code and the like do not cross.
    if len(out) == 1 and out[0]["type"] == "text":
        return out[0]["text"]
    return out


def _dumps(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, default=str)
    except (TypeError, ValueError):
        return str(obj)


def to_messages(llm_request: LlmRequest) -> list[dict[str, Any]]:
    """The request as OpenAI chat messages, system first."""
    messages: list[dict[str, Any]] = []
    system = _system_text(llm_request.config)
    if system:
        messages.append({"role": "system", "content": system})

    # Calls the model has made and not yet seen answered, for responses
    # that arrive without an id.
    open_calls: list[tuple[str, str]] = []   # (id, name)

    for content in llm_request.contents or []:
        parts = list(content.parts or [])
        if not parts:
            continue
        responses = [p for p in parts if p.function_response is not None]
        calls = [p for p in parts if p.function_call is not None]
        rest = [p for p in parts if p.function_response is None and p.function_call is None
                and not getattr(p, "thought", False)]

        for p in responses:
            fr = p.function_response
            call_id = fr.id or ""
            if not call_id:
                for i, (cid, cname) in enumerate(open_calls):
                    if cname == fr.name:
                        call_id = cid
                        open_calls.pop(i)
                        break
            else:
                open_calls[:] = [(cid, n) for cid, n in open_calls if cid != call_id]
            messages.append({"role": "tool", "tool_call_id": call_id, "name": fr.name or "",
                             "content": _dumps(fr.response if fr.response is not None else {})})

        if content.role == "model" or calls:
            tool_calls = []
            for i, p in enumerate(calls):
                fc = p.function_call
                cid = fc.id or f"call_{len(messages)}_{i}"
                open_calls.append((cid, fc.name or ""))
                tool_calls.append({
                    "id": cid, "type": "function",
                    "function": {"name": fc.name or "", "arguments": json.dumps(dict(fc.args or {}), default=str)},
                })
            text = "".join(p.text for p in rest if p.text) or None
            if text or tool_calls:
                msg: dict[str, Any] = {"role": "assistant", "content": text}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                messages.append(msg)
        elif rest:
            body = _content_parts(rest)
            if body:
                messages.append({"role": "user", "content": body})

    # Every tool call needs an answer before the next assistant turn; a call
    # ADK dropped (a budget stop, an interrupted run) gets an honest stub.
    return _close_open_calls(messages)


def _close_open_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        out.append(m)
        i += 1
        if m.get("role") == "assistant" and m.get("tool_calls"):
            wanted = [tc["id"] for tc in m["tool_calls"]]
            answered: set[str] = set()
            while i < len(messages) and messages[i].get("role") == "tool":
                answered.add(messages[i].get("tool_call_id", ""))
                out.append(messages[i])
                i += 1
            for cid in wanted:
                if cid not in answered:
                    out.append({"role": "tool", "tool_call_id": cid,
                                "content": json.dumps({"status": "error", "error": "no result was returned for this call"})})
    return out


# ---------------------------------------------------------------------------
# genai -> OpenAI: tool schemas and output schemas
# ---------------------------------------------------------------------------

_RENAME = {
    "any_of": "anyOf", "min_items": "minItems", "max_items": "maxItems",
    "min_length": "minLength", "max_length": "maxLength", "min_properties": "minProperties",
    "max_properties": "maxProperties", "additional_properties": "additionalProperties",
    "defs": "$defs", "ref": "$ref",
}
_DROP = {"property_ordering", "example"}


def json_schema(schema: Any) -> dict[str, Any]:
    """A genai `Schema` (or a dict in its vocabulary) as plain JSON schema."""
    if isinstance(schema, types.Schema):
        d: dict[str, Any] = schema.model_dump(exclude_none=True, mode="json")
    elif isinstance(schema, dict):
        d = dict(schema)
    else:
        return schema
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k in _DROP:
            continue
        out[_RENAME.get(k, k)] = v
    t = out.get("type")
    if isinstance(t, str):
        out["type"] = t.lower()
    elif isinstance(t, list):
        out["type"] = [str(x).lower() for x in t]
    if out.pop("nullable", False):
        t = out.get("type")
        if isinstance(t, str) and t != "null":
            out["type"] = [t, "null"]
    fmt = out.get("format")
    if fmt is not None and (out.get("type") in ("number", "integer") or fmt == "enum"
                            or (isinstance(out.get("type"), list) and set(out["type"]) & {"number", "integer"})):
        out.pop("format", None)
    if isinstance(out.get("enum"), list):
        out["enum"] = [e for e in out["enum"] if e is not None]
    if isinstance(out.get("properties"), dict):
        out["properties"] = {k: json_schema(v) for k, v in out["properties"].items()}
    if isinstance(out.get("items"), (dict, types.Schema)):
        out["items"] = json_schema(out["items"])
    if isinstance(out.get("anyOf"), list):
        out["anyOf"] = [json_schema(s) for s in out["anyOf"]]
    if isinstance(out.get("$defs"), dict):
        out["$defs"] = {k: json_schema(v) for k, v in out["$defs"].items()}
    return out


def to_tools(llm_request: LlmRequest) -> list[dict[str, Any]]:
    """Every function declaration in the request as an OpenAI tool."""
    tools: list[dict[str, Any]] = []
    config = llm_request.config
    for tool in (getattr(config, "tools", None) or []):
        if not isinstance(tool, types.Tool):
            continue
        for decl in tool.function_declarations or []:
            if not decl.name:
                continue
            if decl.parameters_json_schema:
                params = json_schema(dict(decl.parameters_json_schema))
            elif decl.parameters is not None:
                params = json_schema(decl.parameters)
            else:
                params = {"type": "object", "properties": {}}
            if params.get("type") is None:
                params["type"] = "object"
            params.setdefault("properties", {})
            tools.append({
                "type": "function",
                "function": {"name": decl.name, "description": decl.description or "", "parameters": params},
            })
    return tools


def to_response_format(config: types.GenerateContentConfig | None) -> dict[str, Any] | None:
    """`response_format` for the request, or None when free text is wanted."""
    if config is None:
        return None
    schema = getattr(config, "response_schema", None)
    if schema is not None:
        name = "response"
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            d = schema.model_json_schema()
            name = schema.__name__
        elif isinstance(schema, types.Schema):
            d = json_schema(schema)
        elif isinstance(schema, BaseModel):
            d = schema.__class__.model_json_schema()
            name = schema.__class__.__name__
        elif isinstance(schema, dict):
            if str(schema.get("type", "")).lower() in ("json_schema", "json_object", "text"):
                return dict(schema)
            d = json_schema(schema)
            name = str(schema.get("title") or name)
        else:
            d = None
        if d is not None:
            return {"type": "json_schema", "json_schema": {"name": name, "schema": d}}
    if getattr(config, "response_mime_type", None) == "application/json":
        return {"type": "json_object"}
    return None


# ---------------------------------------------------------------------------
# OpenAI -> genai: the answer
# ---------------------------------------------------------------------------

_FINISH = {
    "stop": types.FinishReason.STOP,
    "tool_calls": types.FinishReason.STOP,
    "function_call": types.FinishReason.STOP,
    "length": types.FinishReason.MAX_TOKENS,
    "content_filter": types.FinishReason.SAFETY,
}


def to_llm_response(result: ChatResult, *, model: str) -> LlmResponse:
    parts: list[types.Part] = []
    if result.text:
        parts.append(types.Part.from_text(text=result.text))
    for tc in result.tool_calls:
        part = types.Part.from_function_call(name=tc.name, args=dict(tc.arguments or {}))
        if tc.id:
            part.function_call.id = tc.id
        parts.append(part)
    usage = None
    if result.usage:
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=result.usage.get("prompt_tokens"),
            candidates_token_count=result.usage.get("completion_tokens"),
            total_token_count=result.usage.get("total_tokens"),
        )
    return LlmResponse(
        content=types.Content(role="model", parts=parts),
        partial=False,
        turn_complete=True,
        finish_reason=_FINISH.get(result.finish_reason.lower()) if result.finish_reason else None,
        usage_metadata=usage,
        model_version=result.model or model,
        custom_metadata={"provider": PROVIDER, "model": result.model or model},
    )


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class MetaLlm(BaseLlm):
    """A Meta Model API model (Muse Spark, Muse Wonder) for ADK agents.

    `model` is the API id. `client` is a `MetaClient` (or the test fake);
    when none is given one is built from the environment on first use, so
    constructing the model never needs a key and never opens a connection.
    """

    client: Any = Field(default=None, exclude=True)
    temperature: float = 0.0
    max_tokens: int | None = None

    def __init__(self, model: str, client: Any = None, **kwargs: Any) -> None:
        super().__init__(model=model, client=client, **kwargs)

    @property
    def capabilities(self):
        from google.adk.models._capabilities import LlmCapabilities

        # Tools and a response schema in one call is untested on the API;
        # ADK then routes a schema-plus-tools agent through its
        # set_model_response tool, which works on any model.
        return LlmCapabilities(output_schema_and_tools=False)

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"muse-.*", r"meta/.*"]

    def _client(self) -> Any:
        if self.client is None:
            self.client = MetaClient()
        return self.client

    def build_request(self, llm_request: LlmRequest) -> dict[str, Any]:
        """Everything `client.chat` needs, as keyword arguments. Exposed for tests."""
        model = (llm_request.model or self.model)
        if model.startswith("meta/"):
            model = model[len("meta/"):]
        config = llm_request.config
        kwargs: dict[str, Any] = {
            "messages": to_messages(llm_request),
            "model": model,
            "tools": to_tools(llm_request) or None,
            "response_format": to_response_format(config),
            "temperature": self.temperature if getattr(config, "temperature", None) is None else float(config.temperature),
        }
        max_tokens = getattr(config, "max_output_tokens", None) or self.max_tokens
        if max_tokens:
            kwargs["max_tokens"] = int(max_tokens)
        return kwargs

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self._maybe_append_user_content(llm_request)
        kwargs = self.build_request(llm_request)
        client = self._client()
        try:
            result = await asyncio.to_thread(client.chat, **kwargs)
        except MetaError as exc:
            fmt = kwargs.get("response_format") or {}
            if exc.status == 400 and fmt.get("type") == "json_schema":
                # Some ids reject json_schema; json_object is universally honoured.
                kwargs["response_format"] = {"type": "json_object"}
                result = await asyncio.to_thread(client.chat, **kwargs)
            else:
                raise
        yield to_llm_response(result, model=kwargs["model"])


def register() -> None:
    """Let `LlmAgent(model="muse-spark-1.3")` resolve to `MetaLlm`."""
    from google.adk.models.registry import LLMRegistry

    LLMRegistry.register(MetaLlm)


register()

__all__ = ["MetaLlm", "register", "to_messages", "to_tools", "to_response_format", "to_llm_response",
           "json_schema", "PROVIDER"]
