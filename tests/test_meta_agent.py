"""The ADK agents on Meta's models: the wire conversion both ways, one real
ADK round trip over a mocked HTTP transport, provider selection, and the
frame reviewer. No network; the model is a script, as in test_coverage."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from google.adk.models.llm_request import LlmRequest
from google.adk.tools import FunctionTool, ToolContext
from google.genai import types

from locaish.agent import core
from locaish.agent import coverage as agentcov
from locaish.film import coverage as cov
from locaish.meta import adk
from locaish.meta import client as mc

PNG = mc._blank_png()

GEMINI_VARS = ("GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE",
               "GOOGLE_CLOUD_PROJECT")
META_VARS = ("META_API_KEY", "MODEL_API_KEY")


@pytest.fixture
def no_providers(monkeypatch):
    for var in GEMINI_VARS + META_VARS + ("LOCAISH_AGENT_PROVIDER",):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def measure_room(width_m: float, note: str = "", tool_context: ToolContext = None) -> dict:
    """Measure the room.

    Args:
        width_m: The width to check, in metres.
        note: Optional note.
    """
    return {"status": "success", "width_m": width_m, "area_m2": round(width_m * 3.0, 2)}


# ---------------------------------------------------------------------------
# (a) conversion
# ---------------------------------------------------------------------------


def _request_with_history() -> LlmRequest:
    req = LlmRequest(model="muse-wonder-1.0")
    req.config.system_instruction = "You are the scout."
    req.append_tools([FunctionTool(measure_room)])
    call = types.Part.from_function_call(name="measure_room", args={"width_m": 4.2})
    call.function_call.id = "call_a1"
    resp = types.Part.from_function_response(name="measure_room", response={"status": "success", "area_m2": 12.6})
    resp.function_response.id = "call_a1"
    req.contents = [
        types.Content(role="user", parts=[types.Part(text="how wide is it?"),
                                          types.Part.from_bytes(data=PNG, mime_type="image/png")]),
        types.Content(role="model", parts=[types.Part(text="Measuring."), call]),
        types.Content(role="user", parts=[resp]),
    ]
    return req


def test_request_becomes_openai_messages_with_tool_ids_preserved():
    messages = adk.to_messages(_request_with_history())
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert messages[0]["content"] == "You are the scout."
    user = messages[1]["content"]
    assert user[0] == {"type": "text", "text": "how wide is it?"}
    assert user[1]["type"] == "image_url"
    assert user[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assistant = messages[2]
    assert assistant["content"] == "Measuring."
    tc = assistant["tool_calls"][0]
    assert tc["id"] == "call_a1" and tc["type"] == "function"
    assert tc["function"]["name"] == "measure_room"
    assert json.loads(tc["function"]["arguments"]) == {"width_m": 4.2}
    tool = messages[3]
    assert tool["tool_call_id"] == "call_a1" and tool["name"] == "measure_room"
    assert json.loads(tool["content"]) == {"status": "success", "area_m2": 12.6}


def test_text_only_user_turn_is_a_plain_string():
    req = LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="hi")])])
    assert adk.to_messages(req) == [{"role": "user", "content": "hi"}]


def test_function_response_without_an_id_pairs_with_the_open_call_by_name():
    req = LlmRequest(contents=[
        types.Content(role="model", parts=[types.Part.from_function_call(name="measure_room", args={})]),
        types.Content(role="user", parts=[types.Part.from_function_response(name="measure_room", response={"ok": 1})]),
    ])
    messages = adk.to_messages(req)
    assert messages[0]["tool_calls"][0]["id"] == messages[1]["tool_call_id"]


def test_unanswered_tool_call_gets_a_stub_result_so_the_transcript_stays_valid():
    req = LlmRequest(contents=[
        types.Content(role="model", parts=[types.Part.from_function_call(name="measure_room", args={})]),
        types.Content(role="user", parts=[types.Part(text="never mind")]),
    ])
    messages = adk.to_messages(req)
    assert [m["role"] for m in messages] == ["assistant", "tool", "user"]
    assert "error" in json.loads(messages[1]["content"])


def test_function_declarations_become_openai_tool_schemas():
    tools = adk.to_tools(_request_with_history())
    assert len(tools) == 1
    fn = tools[0]["function"]
    assert tools[0]["type"] == "function" and fn["name"] == "measure_room"
    assert "width_m" in fn["description"]
    params = fn["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["width_m"]["type"] == "number"
    assert params["properties"]["note"]["type"] == "string"
    assert params["required"] == ["width_m"]
    # tool_context never reaches the model
    assert "tool_context" not in params["properties"]


def test_genai_schema_is_lowered_to_json_schema():
    decl = types.FunctionDeclaration(
        name="run_query", description="SQL",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "sql": types.Schema(type="STRING"),
                "limit": types.Schema(type="INTEGER", nullable=True, format="int32"),
                "mode": types.Schema(type="STRING", enum=["fast", "full"], format="enum"),
                "cols": types.Schema(type="ARRAY", items=types.Schema(type="STRING"), min_items=1),
            },
            required=["sql"], property_ordering=["sql", "limit"],
        ),
    )
    req = LlmRequest()
    req.config.tools = [types.Tool(function_declarations=[decl])]
    params = adk.to_tools(req)[0]["function"]["parameters"]
    assert params["type"] == "object" and params["required"] == ["sql"]
    assert params["properties"]["sql"] == {"type": "string"}
    assert params["properties"]["limit"] == {"type": ["integer", "null"]}
    assert params["properties"]["mode"] == {"type": "string", "enum": ["fast", "full"]}
    assert params["properties"]["cols"] == {"type": "array", "items": {"type": "string"}, "minItems": 1}
    assert "property_ordering" not in params and "propertyOrdering" not in params


def test_output_schema_becomes_json_schema_response_format():
    from pydantic import BaseModel

    class Answer(BaseModel):
        title: str
        n: int | None = None

    req = LlmRequest()
    req.set_output_schema(Answer)
    fmt = adk.to_response_format(req.config)
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["name"] == "Answer"
    assert fmt["json_schema"]["schema"]["required"] == ["title"]
    # Plain JSON without a schema asks for json_object; free text asks for nothing.
    req2 = LlmRequest()
    req2.config.response_mime_type = "application/json"
    assert adk.to_response_format(req2.config) == {"type": "json_object"}
    assert adk.to_response_format(LlmRequest().config) is None


def test_chat_result_becomes_an_llm_response_with_ids_and_usage():
    res = mc.ChatResult(
        text="Here.", model="muse-wonder-1.0", finish_reason="tool_calls",
        tool_calls=[mc.ToolCall(id="call_z", name="measure_room", arguments={"width_m": 2})],
        usage={"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
    )
    out = adk.to_llm_response(res, model="muse-wonder-1.0")
    parts = out.content.parts
    assert parts[0].text == "Here."
    fc = out.get_function_calls()[0]
    assert fc.id == "call_z" and fc.name == "measure_room" and fc.args == {"width_m": 2}
    assert out.usage_metadata.prompt_token_count == 12 and out.usage_metadata.candidates_token_count == 5
    assert out.finish_reason == types.FinishReason.STOP
    assert out.custom_metadata["provider"] == "meta"
    assert out.partial is False


def test_streaming_yields_one_final_response():
    fake = mc.FakeMetaClient(chat_answers=["done"])
    llm = adk.MetaLlm("muse-spark-1.3", client=fake)
    req = LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="go")])])

    async def collect():
        return [r async for r in llm.generate_content_async(req, stream=True)]

    out = asyncio.run(collect())
    assert len(out) == 1 and out[0].partial is False and out[0].content.parts[0].text == "done"
    assert fake.log[0]["model"] == "muse-spark-1.3"


def test_registry_resolves_muse_ids_to_meta_llm():
    from google.adk.models.registry import LLMRegistry

    llm = LLMRegistry.new_llm("muse-spark-1.3")
    assert isinstance(llm, adk.MetaLlm) and llm.model == "muse-spark-1.3"
    assert adk.MetaLlm.supported_models()[0] == r"muse-.*"


# ---------------------------------------------------------------------------
# (b) a real ADK round trip over a mocked transport
# ---------------------------------------------------------------------------


def test_adk_agent_runs_a_tool_through_the_meta_api(no_providers):
    from google.adk.agents import LlmAgent
    from google.adk.runners import InMemoryRunner

    seen: list[dict] = []
    executed: list[float] = []

    def tool(width_m: float, tool_context: ToolContext = None) -> dict:
        """Measure the room.

        Args:
            width_m: Width in metres.
        """
        executed.append(width_m)
        return {"status": "success", "area_m2": width_m * 3.0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            return httpx.Response(200, json={
                "model": "muse-wonder-1.0",
                "choices": [{"finish_reason": "tool_calls", "message": {
                    "content": None,
                    "tool_calls": [{"id": "call_7", "type": "function",
                                    "function": {"name": "tool", "arguments": json.dumps({"width_m": 4.0})}}],
                }}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 9, "total_tokens": 49},
            })
        return httpx.Response(200, json={
            "model": "muse-wonder-1.0",
            "choices": [{"finish_reason": "stop", "message": {"content": "The room is 12.0 m2 (from the tool)."}}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 12, "total_tokens": 92},
        })

    client = mc.MetaClient("test-key", transport=httpx.MockTransport(handler))
    agent = LlmAgent(
        name="scout", model=adk.MetaLlm("muse-wonder-1.0", client=client),
        instruction="Measure, then answer with the number.", tools=[tool],
    )

    async def run():
        runner = InMemoryRunner(agent=agent, app_name="t")
        await runner.session_service.create_session(app_name="t", user_id="u", session_id="s")
        msg = types.Content(role="user", parts=[types.Part(text="How big is it? It is 4 m wide.")])
        texts, calls, responses, usage = [], [], [], []
        async for ev in runner.run_async(user_id="u", session_id="s", new_message=msg):
            calls.extend(ev.get_function_calls())
            responses.extend(ev.get_function_responses())
            if ev.usage_metadata is not None:
                usage.append(ev.usage_metadata.prompt_token_count)
            if ev.is_final_response() and ev.content and ev.content.parts:
                texts.extend(p.text or "" for p in ev.content.parts)
        await runner.close()
        return texts, calls, responses, usage

    texts, calls, responses, usage = asyncio.run(run())

    assert executed == [4.0]
    assert "".join(texts) == "The room is 12.0 m2 (from the tool)."
    assert calls[0].name == "tool" and calls[0].id == "call_7"
    assert responses[0].id == "call_7" and responses[0].response["area_m2"] == 12.0
    assert usage == [40, 80]

    # What went over the wire: the system prompt, the tool, tool_choice auto,
    # and on the second turn the call and its result paired by id.
    first, second = seen
    assert first["model"] == "muse-wonder-1.0" and first["tool_choice"] == "auto"
    assert first["messages"][0]["role"] == "system" and "Measure" in first["messages"][0]["content"]
    assert [t["function"]["name"] for t in first["tools"]] == ["tool"]
    assert first["tools"][0]["function"]["parameters"]["properties"]["width_m"]["type"] == "number"
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert second["messages"][2]["tool_calls"][0]["id"] == "call_7"
    assert second["messages"][3]["tool_call_id"] == "call_7"
    assert json.loads(second["messages"][3]["content"])["area_m2"] == 12.0
    assert client.calls == 2


def test_json_schema_rejection_falls_back_to_json_object():
    seen: list[dict] = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(400, text="response_format.json_schema is not supported")
        return httpx.Response(200, json={"choices": [{"message": {"content": "{\"title\": \"x\"}"}}]})

    from pydantic import BaseModel

    class Out(BaseModel):
        title: str

    client = mc.MetaClient("k", transport=httpx.MockTransport(handler))
    llm = adk.MetaLlm("muse-spark-1.3", client=client)
    req = LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="go")])])
    req.set_output_schema(Out)

    async def one():
        return [r async for r in llm.generate_content_async(req)]

    out = asyncio.run(one())
    assert json.loads(out[0].content.parts[0].text) == {"title": "x"}
    assert [b["response_format"]["type"] for b in seen] == ["json_schema", "json_object"]


# ---------------------------------------------------------------------------
# (c) provider selection
# ---------------------------------------------------------------------------


def test_provider_auto_prefers_meta_when_a_key_is_present(no_providers):
    assert core.provider() == "gemini"
    assert core.provider_configured() is False
    assert agentcov.agent_configured() is False
    no_providers.setenv("META_API_KEY", "k")
    assert core.provider() == "meta"
    assert core.provider_configured() is True
    assert agentcov.agent_configured() is True
    assert core.provider_label() == "Meta Muse"


def test_provider_flag_overrides_auto(no_providers):
    no_providers.setenv("META_API_KEY", "k")
    no_providers.setenv("GOOGLE_API_KEY", "g")
    no_providers.setenv("LOCAISH_AGENT_PROVIDER", "gemini")
    assert core.provider() == "gemini" and core.provider_configured() is True
    no_providers.setenv("LOCAISH_AGENT_PROVIDER", "meta")
    assert core.provider() == "meta" and core.provider_configured() is True
    # Asking for Meta without a key is a known, named failure.
    no_providers.delenv("META_API_KEY")
    assert core.provider() == "meta" and core.provider_configured() is False
    no_providers.setenv("LOCAISH_AGENT_PROVIDER", "openai")
    with pytest.raises(core.AgentUnavailable):
        core.provider()


def test_gemini_only_box_keeps_working(no_providers):
    no_providers.setenv("GOOGLE_API_KEY", "g")
    assert core.provider() == "gemini" and core.provider_configured() is True
    assert agentcov.agent_configured() is True
    assert core.agent_model("agent") == core.DEFAULT_MODEL
    assert isinstance(core.agent_model("breakdown"), str)


def test_agent_model_returns_meta_llm_per_job(no_providers):
    no_providers.setenv("META_API_KEY", "k")
    no_providers.setenv("LOCAISH_META_AGENT_MODEL", "muse-wonder-9")
    scout = core.agent_model("agent")
    assert isinstance(scout, adk.MetaLlm) and scout.model == "muse-wonder-9"
    assert core.agent_model("planner").model == "muse-wonder-9"
    assert core.agent_model("breakdown").model == mc.DEFAULT_MODELS["breakdown"]
    assert core.agent_model("notes").model == mc.DEFAULT_MODELS["breakdown"]
    assert core.agent_model("reviewer").model == mc.DEFAULT_MODELS["vision"]
    # The coverage workflow's agents pick up the same objects.
    agents = agentcov.build_workflow()
    assert isinstance(agents["placer"].model, adk.MetaLlm)
    assert isinstance(agents["breakdown"].model, adk.MetaLlm)
    assert agents["breakdown"].model.model == mc.DEFAULT_MODELS["breakdown"]


def test_runner_refuses_with_both_configurations_named(no_providers):
    svc = core.AgentService()
    with pytest.raises(core.AgentUnavailable) as exc:
        svc._ensure_runner()
    msg = str(exc.value)
    assert "META_API_KEY" in msg and "GOOGLE_API_KEY" in msg and "LOCAISH_AGENT_PROVIDER" in msg


# ---------------------------------------------------------------------------
# (d) the frame reviewer
# ---------------------------------------------------------------------------


def _frame(tmp_path: Path) -> Path:
    p = tmp_path / "shot_001.png"
    p.write_bytes(PNG)
    return p


ROW = {"focal_mm": 50, "distance_m": 2.1, "cam_z": 1.5, "shot_size": "cu", "window_behind_subject": 0}


def test_meta_reviewer_returns_a_review_from_the_frame(tmp_path):
    fake = mc.FakeMetaClient(chat_answers=[{
        "score": 6.5, "verdict": "adjust", "notes": "Too tight; the doorway crowds the frame.",
        "suggestion": {"height": "none", "lens_mm": 35, "size": "", "no_window_behind": False},
    }])
    review = agentcov.meta_reviewer(client=fake)
    shot = cov.Shot(number=1, description="Close-up MAYA", size="cu", subject="MAYA").normalised()
    rv = review(_frame(tmp_path), shot, ROW)
    assert isinstance(rv, cov.Review)
    assert rv.score == 6.5 and rv.verdict == "adjust"
    assert rv.notes.startswith("Too tight")
    assert rv.suggestion == {"lens_mm": 35}          # the empties and "none" are dropped
    assert rv.model == f"meta/{mc.DEFAULT_MODELS['vision']}"
    call = fake.log[0]
    assert call["kind"] == "chat_json" and call["images"] == 1
    assert "50 mm" in call["text"] and "Close-up MAYA" in call["text"]
    assert call["schema"]["type"] == "object"
    assert call["schema"]["properties"]["verdict"]["enum"] == ["keep", "adjust", "reject"]


def test_meta_reviewer_survives_a_sloppy_answer(tmp_path):
    fake = mc.FakeMetaClient(chat_answers=[{"score": "8", "verdict": "KEEP", "notes": "fine"}])
    rv = agentcov.meta_reviewer(model="muse-spark-9", client=fake)(
        _frame(tmp_path), cov.Shot(number=2, description="x", size="ms", subject="A").normalised(), ROW)
    assert rv.score == 8.0 and rv.verdict == "keep" and rv.model == "meta/muse-spark-9"


def test_reviewer_factory_follows_the_provider(no_providers):
    no_providers.setenv("META_API_KEY", "k")
    assert agentcov.reviewer().__qualname__.startswith("meta_reviewer")
