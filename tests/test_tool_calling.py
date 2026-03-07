import asyncio
import base64

from agent_smithers.app import AppContext
from agent_smithers.config import AppConfig, LLMConfig, MatrixConfig


class FakeLLM:
    def __init__(self):
        self.calls = 0

    async def create_response(self, **payload):
        self.calls += 1
        if self.calls == 1:
            return {
                "id": "resp_1",
                "output": [
                    {
                        "id": "approve_1",
                        "type": "mcp_approval_request",
                        "server_label": "deepwiki",
                    }
                ],
            }
        assert "previous_response_id" not in payload
        assert payload["input_items"] == [
            {
                "id": "approve_1",
                "type": "mcp_approval_request",
                "server_label": "deepwiki",
            },
            {
                "type": "mcp_approval_response",
                "approval_request_id": "approve_1",
                "approve": True,
            }
        ]
        assert payload["instructions"] == "you are p."
        return {
            "id": "resp_2",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Result is 4"}],
                }
            ],
        }


class FakeMatrix:
    def __init__(self):
        self.sent_images = []
        self.sent_videos = []

    async def send_image(self, room_id, path, filename=None, log=None):
        self.sent_images.append((room_id, path, filename))

    async def send_video(self, room_id, path, filename=None, log=None):
        self.sent_videos.append((room_id, path, filename))


class CaptureLLM:
    def __init__(self):
        self.last_payload = None

    async def create_response(self, **payload):
        self.last_payload = payload
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "ok"}],
                }
            ]
        }


class RecordingStatus:
    def __init__(self, message, events):
        self.message = message
        self.events = events

    def __enter__(self):
        self.events.append(("enter", self.message))
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        self.events.append(("exit", self.message))
        return False

    def update(self, status=None, **kwargs):
        del kwargs
        if status:
            self.message = status
            self.events.append(("update", status))


def test_mcp_auto_approval_loop_completes():
    cfg = AppConfig(
        llm=LLMConfig(
            models={"openai": ["gpt-5-mini"]},
            api_keys={"openai": "X"},
            default_model="gpt-5-mini",
            personality="p",
            prompt=["you are ", "."],
            mcp_servers={
                "deepwiki": {
                    "server_url": "https://mcp.deepwiki.com/mcp",
                    "auto_approve": True,
                }
            },
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admins=[]),
    )
    ctx = AppContext(cfg)
    try:
        ctx.llm = FakeLLM()
        messages = [{"role": "system", "content": "you are p."}]
        out = asyncio.run(ctx.respond_with_tools(messages, room_id="!r"))
        assert out == "Result is 4"
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def _ctx():
    cfg = AppConfig(
        llm=LLMConfig(
            models={"openai": ["gpt-5-mini"]},
            api_keys={"openai": "X"},
            default_model="gpt-5-mini",
            personality="p",
            prompt=["you are ", "."],
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admins=[]),
    )
    ctx = AppContext(cfg)
    ctx.matrix = FakeMatrix()
    return ctx


def test_send_response_artifacts_handles_inline_image_payload():
    ctx = _ctx()
    try:
        response = {
            "output": [
                {
                    "type": "image_generation_call",
                    "result": [{"b64_json": "aGVsbG8="}],
                }
            ]
        }
        sent = asyncio.run(ctx._send_response_artifacts(response, "!r", provider="openai"))
        assert sent is True
        assert len(ctx.matrix.sent_images) == 1
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_send_response_artifacts_handles_file_backed_image_payload():
    ctx = _ctx()
    try:
        async def fake_download_image_bytes(*, provider, file_id, container_id):
            assert provider == "openai"
            assert file_id == "file_123"
            assert container_id == "container_123"
            return b"pngbytes"

        ctx._download_image_bytes = fake_download_image_bytes
        response = {
            "output": [
                {
                    "type": "image_generation_call",
                    "container_id": "container_123",
                    "result": [{"file_id": "file_123"}],
                }
            ]
        }
        sent = asyncio.run(ctx._send_response_artifacts(response, "!r", provider="openai"))
        assert sent is True
        assert len(ctx.matrix.sent_images) == 1
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_generate_reply_uses_spinner_status_while_waiting_for_model_response():
    ctx = _ctx()
    events = []
    try:
        ctx.llm = CaptureLLM()
        ctx.status = lambda message, spinner="dots": RecordingStatus(message, events)
        out = asyncio.run(
            ctx.generate_reply(
                [{"role": "user", "content": "hello"}],
                room_id="!r",
            )
        )
        assert out == "ok"
        assert ("enter", "Generating reply with gpt-5-mini") in events
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_generate_video_call_updates_spinner_status():
    class FakeVideoLLM:
        async def generate_video(self, **kwargs):
            assert kwargs["backend"] == "sora"
            kwargs["on_status"]("Generating video with Sora [queued]")
            return {"id": "vid_123"}

        async def download_video_content(self, video_id, *, provider):
            assert video_id == "vid_123"
            assert provider == "openai"
            return b"video-bytes"

    ctx = _ctx()
    events = []
    try:
        ctx.llm = FakeVideoLLM()
        ctx.status = lambda message, spinner="dots": RecordingStatus(message, events)
        result = asyncio.run(
            ctx._handle_generate_image_calls(
                {
                    "output": [
                        {
                            "type": "function_call",
                            "name": "sora_generate_video",
                            "call_id": "call_1",
                            "arguments": '{"prompt":"animate this"}',
                        }
                    ]
                },
                model="gpt-5-mini",
                room_id="!r",
                thread_user="@user:test",
            )
        )
        assert result == [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Video generated and sent.",
            }
        ]
        assert ("enter", "Generating video with Sora") in events
        assert ("update", "Generating video with Sora [queued]") in events
        assert ("update", "Downloading video from Sora") in events
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_generate_reply_continues_openai_followup_after_local_tool_output():
    class FakeOpenAIFollowupLLM:
        def __init__(self):
            self.calls = 0

        async def create_response(self, **payload):
            self.calls += 1
            if self.calls == 1:
                assert payload["model"] == "gpt-5-mini"
                return {
                    "id": "resp_1",
                    "output": [
                        {
                            "id": "rs_1",
                            "type": "reasoning",
                            "summary": [],
                        },
                        {
                            "type": "function_call",
                            "name": "grok_generate_image",
                            "call_id": "call_img",
                            "arguments": '{"prompt":"draw a cat"}',
                        }
                    ],
                }
            if self.calls == 2:
                assert "previous_response_id" not in payload
                assert payload["input_items"] == [
                    {"role": "user", "content": "make me a cat"},
                    {"id": "rs_1", "type": "reasoning", "summary": []},
                    {
                        "type": "function_call",
                        "name": "grok_generate_image",
                        "call_id": "call_img",
                        "arguments": '{"prompt":"draw a cat"}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_img",
                        "output": "Image generated and sent.",
                    },
                ]
                return {
                    "id": "resp_2",
                    "output": [
                        {
                            "id": "approve_1",
                            "type": "mcp_approval_request",
                            "server_label": "deepwiki",
                        }
                    ],
                }
            assert "previous_response_id" not in payload
            assert payload["input_items"] == [
                {"role": "user", "content": "make me a cat"},
                {"id": "rs_1", "type": "reasoning", "summary": []},
                {
                    "type": "function_call",
                    "name": "grok_generate_image",
                    "call_id": "call_img",
                    "arguments": '{"prompt":"draw a cat"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_img",
                    "output": "Image generated and sent.",
                },
                {
                    "id": "approve_1",
                    "type": "mcp_approval_request",
                    "server_label": "deepwiki",
                },
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": "approve_1",
                    "approve": True,
                },
            ]
            return {
                "id": "resp_3",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Here is the cat image."}],
                    }
                ],
            }

        async def generate_image(self, **payload):
            assert payload["provider_override"] == "xai"
            return {
                "data": [
                    {"b64_json": base64.b64encode(b"pngbytes").decode()}
                ]
            }

    ctx = _ctx()
    try:
        ctx.llm = FakeOpenAIFollowupLLM()
        ctx._mcp_auto_approve = {"deepwiki"}
        out = asyncio.run(
            ctx.generate_reply(
                [{"role": "user", "content": "make me a cat"}],
                model="gpt-5-mini",
                room_id="!r",
                use_tools=True,
            )
        )
        assert out == "Here is the cat image."
        assert len(ctx.matrix.sent_images) == 1
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_extract_text_keeps_inline_citation_markers():
    response = {
        "output_text": "Answer with sources【abc†source】",
    }
    assert AppContext._extract_text(response) == "Answer with sources【abc†source】"


def test_extract_text_keeps_annotation_text_from_output_items():
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Answer text",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "text": "",
                            }
                        ],
                    }
                ],
            }
        ]
    }
    assert AppContext._extract_text(response) == "Answer text"


def test_strip_inline_citations_is_now_a_noop():
    text = "Answer text"
    annotations = [{"type": "url_citation", "text": ""}]
    assert AppContext._strip_inline_citations(text, annotations) == "Answer text"


def test_xai_hosted_tools_include_x_search_and_map_mcp_fields():
    cfg = AppConfig(
        llm=LLMConfig(
            models={"xai": ["grok-4"]},
            api_keys={"xai": "X"},
            default_model="grok-4",
            personality="p",
            prompt=["you are ", "."],
            tools={
                "web_search": True,
                "x_search": True,
                "code_interpreter": True,
            },
            mcp_servers={
                "deepwiki": {
                    "server_url": "https://mcp.deepwiki.com/mcp",
                    "allowed_tools": ["ask_question"],
                    "headers": {"X-Test": "1"},
                    "require_approval": "never",
                }
            },
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admins=[]),
    )
    ctx = AppContext(cfg)
    try:
        assert {"type": "web_search"} in ctx.hosted_tools
        assert {"type": "x_search"} in ctx.hosted_tools
        assert {"type": "code_interpreter"} in ctx.hosted_tools
        function_names = {tool["name"] for tool in ctx.hosted_tools if tool["type"] == "function"}
        assert function_names == {"grok_generate_image", "grok_edit_image", "grok_generate_video"}
        mcp_tool = next(tool for tool in ctx.hosted_tools if tool["type"] == "mcp")
        assert mcp_tool["allowed_tool_names"] == ["ask_question"]
        assert mcp_tool["extra_headers"] == {"X-Test": "1"}
        assert "require_approval" not in mcp_tool
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_dual_provider_tool_sets_follow_selected_model():
    cfg = AppConfig(
        llm=LLMConfig(
            models={"openai": ["gpt-5-mini"], "xai": ["grok-4"]},
            api_keys={"openai": "O", "xai": "X"},
            default_model="gpt-5-mini",
            personality="p",
            prompt=["you are ", "."],
            tools={
                "web_search": True,
                "x_search": True,
                "code_interpreter": True,
                "image_generation": True,
            },
            web_search_country="US",
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admins=[]),
    )
    ctx = AppContext(cfg)
    try:
        openai_tools = ctx._tools_for_model("gpt-5-mini")
        xai_tools = ctx._tools_for_model("grok-4")
        assert {"type": "image_generation"} in openai_tools
        assert {tool["name"] for tool in openai_tools if tool["type"] == "function"} == {
            "grok_generate_image",
            "grok_edit_image",
            "grok_generate_video",
            "sora_generate_video",
        }
        assert {"type": "x_search"} not in openai_tools
        assert {"type": "x_search"} in xai_tools
        assert {"type": "image_generation"} not in xai_tools
        openai_web_search = next(tool for tool in openai_tools if tool["type"] == "web_search")
        xai_web_search = next(tool for tool in xai_tools if tool["type"] == "web_search")
        assert openai_web_search["user_location"] == {"type": "approximate", "country": "US"}
        assert xai_web_search == {"type": "web_search"}
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_xai_non_grok4_models_do_not_get_hosted_tools_or_mcp():
    cfg = AppConfig(
        llm=LLMConfig(
            models={"xai": ["grok-code-fast-1", "grok-4"]},
            api_keys={"xai": "X"},
            default_model="grok-code-fast-1",
            personality="p",
            prompt=["you are ", "."],
            tools={
                "web_search": True,
                "x_search": True,
                "code_interpreter": True,
                "image_generation": True,
            },
            mcp_servers={
                "deepwiki": {
                    "server_url": "https://mcp.deepwiki.com/mcp",
                }
            },
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admins=[]),
    )
    ctx = AppContext(cfg)
    try:
        code_tools = ctx._tools_for_model("grok-code-fast-1")
        grok4_tools = ctx._tools_for_model("grok-4")
        assert {tool["name"] for tool in code_tools if tool["type"] == "function"} == {
            "grok_generate_image",
            "grok_edit_image",
            "grok_generate_video",
        }
        assert not any(tool["type"] == "web_search" for tool in code_tools)
        assert not any(tool["type"] == "x_search" for tool in code_tools)
        assert not any(tool["type"] == "code_interpreter" for tool in code_tools)
        assert not any(tool["type"] == "mcp" for tool in code_tools)
        assert any(tool["type"] == "web_search" for tool in grok4_tools)
        assert any(tool["type"] == "x_search" for tool in grok4_tools)
        assert any(tool["type"] == "code_interpreter" for tool in grok4_tools)
        assert any(tool["type"] == "mcp" for tool in grok4_tools)
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_xai_search_country_policy_is_added_to_messages():
    cfg = AppConfig(
        llm=LLMConfig(
            models={"xai": ["grok-4"]},
            api_keys={"xai": "X"},
            default_model="grok-4",
            personality="p",
            prompt=["you are ", "."],
            tools={
                "web_search": True,
                "x_search": True,
            },
            web_search_country="US",
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admins=[]),
    )
    ctx = AppContext(cfg)
    try:
        capture = CaptureLLM()
        ctx.llm = capture
        out = asyncio.run(
            ctx.generate_reply(
                [{"role": "system", "content": "be concise"}, {"role": "user", "content": "hello"}],
                model="grok-4",
                use_tools=True,
            )
        )
        assert out == "ok"
        assert capture.last_payload is not None
        message0 = capture.last_payload["messages"][0]
        assert message0["role"] == "system"
        assert "prioritize results and sources from US" in message0["content"]
        assert "x_search" in message0["content"]
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_xai_non_grok4_models_do_not_apply_search_policy_without_search_tools():
    cfg = AppConfig(
        llm=LLMConfig(
            models={"xai": ["grok-code-fast-1"]},
            api_keys={"xai": "X"},
            default_model="grok-code-fast-1",
            personality="p",
            prompt=["you are ", "."],
            tools={
                "web_search": True,
                "x_search": True,
                "image_generation": True,
            },
            web_search_country="US",
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admins=[]),
    )
    ctx = AppContext(cfg)
    try:
        capture = CaptureLLM()
        ctx.llm = capture
        out = asyncio.run(
            ctx.generate_reply(
                [{"role": "system", "content": "be concise"}, {"role": "user", "content": "hello"}],
                model="grok-code-fast-1",
                use_tools=True,
            )
        )
        assert out == "ok"
        assert capture.last_payload is not None
        assert capture.last_payload["tools"] == [tool for tool in ctx._tools_for_model("grok-code-fast-1")]
        message0 = capture.last_payload["messages"][0]
        assert message0["content"] == "be concise"
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


# --- _apply_search_country_policy ---

def test_apply_search_country_policy_no_country_returns_unmodified():
    cfg = AppConfig(
        llm=LLMConfig(
            models={"xai": ["grok-4"]},
            api_keys={"xai": "X"},
            default_model="grok-4",
            personality="p",
            prompt=["you are ", "."],
            web_search_country="",
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admins=[]),
    )
    ctx = AppContext(cfg)
    try:
        messages = [{"role": "user", "content": "hi"}]
        result = ctx._apply_search_country_policy(messages, provider="xai", tools=[{"type": "web_search"}])
        assert len(result) == 1
        assert result[0]["content"] == "hi"
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_apply_search_country_policy_non_xai_returns_unmodified():
    cfg = AppConfig(
        llm=LLMConfig(
            models={"openai": ["gpt-5-mini"]},
            api_keys={"openai": "X"},
            default_model="gpt-5-mini",
            personality="p",
            prompt=["you are ", "."],
            web_search_country="US",
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admins=[]),
    )
    ctx = AppContext(cfg)
    try:
        messages = [{"role": "user", "content": "hi"}]
        result = ctx._apply_search_country_policy(messages, provider="openai", tools=[{"type": "web_search"}])
        assert len(result) == 1
        assert result[0]["content"] == "hi"
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_apply_search_country_policy_no_search_tools_returns_unmodified():
    cfg = AppConfig(
        llm=LLMConfig(
            models={"xai": ["grok-4"]},
            api_keys={"xai": "X"},
            default_model="grok-4",
            personality="p",
            prompt=["you are ", "."],
            web_search_country="US",
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admins=[]),
    )
    ctx = AppContext(cfg)
    try:
        messages = [{"role": "user", "content": "hi"}]
        result = ctx._apply_search_country_policy(messages, provider="xai", tools=[{"type": "code_interpreter"}])
        assert len(result) == 1
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_apply_search_country_policy_merges_into_existing_system():
    cfg = AppConfig(
        llm=LLMConfig(
            models={"xai": ["grok-4"]},
            api_keys={"xai": "X"},
            default_model="grok-4",
            personality="p",
            prompt=["you are ", "."],
            web_search_country="US",
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admins=[]),
    )
    ctx = AppContext(cfg)
    try:
        messages = [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "hi"},
        ]
        result = ctx._apply_search_country_policy(messages, provider="xai", tools=[{"type": "x_search"}])
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "be concise" in result[0]["content"]
        assert "prioritize results" in result[0]["content"]
        assert result[1]["content"] == "hi"
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_apply_search_country_policy_prepends_system_when_none():
    cfg = AppConfig(
        llm=LLMConfig(
            models={"xai": ["grok-4"]},
            api_keys={"xai": "X"},
            default_model="grok-4",
            personality="p",
            prompt=["you are ", "."],
            web_search_country="US",
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admins=[]),
    )
    ctx = AppContext(cfg)
    try:
        messages = [{"role": "user", "content": "hi"}]
        result = ctx._apply_search_country_policy(messages, provider="xai", tools=[{"type": "web_search"}])
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "prioritize results" in result[0]["content"]
        assert result[1]["content"] == "hi"
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


# --- _walk_image_results ---

def test_walk_image_results_string_yields_inline():
    results = list(AppContext._walk_image_results("base64data"))
    assert results == [{"inline": "base64data"}]


def test_walk_image_results_list_recurses():
    results = list(AppContext._walk_image_results(["a", "b"]))
    assert results == [{"inline": "a"}, {"inline": "b"}]


def test_walk_image_results_dict_with_b64_json():
    results = list(AppContext._walk_image_results({"b64_json": "abc123"}))
    assert results == [{"inline": "abc123"}]


def test_walk_image_results_dict_with_file_id():
    results = list(AppContext._walk_image_results({"file_id": "f1", "container_id": "c1"}))
    assert results == [{"file_id": "f1", "container_id": "c1"}]


def test_walk_image_results_dict_with_nested_file():
    results = list(AppContext._walk_image_results({
        "file": {"id": "f2", "container_id": "c2"},
    }))
    assert {"file_id": "f2", "container_id": "c2"} in results


def test_walk_image_results_non_dict_non_list_non_str():
    results = list(AppContext._walk_image_results(42))
    assert results == []


def test_walk_image_results_empty_dict():
    results = list(AppContext._walk_image_results({}))
    assert results == []


# --- _iter_image_sources ---

def test_iter_image_sources_output_image_with_file_id():
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_image", "file_id": "f1", "container_id": "c1"},
                ],
            }
        ]
    }
    sources = list(AppContext._iter_image_sources(response))
    assert {"file_id": "f1", "container_id": "c1"} in sources


def test_iter_image_sources_output_image_with_data_uri():
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_image", "image_url": "data:image/png;base64,abc"},
                ],
            }
        ]
    }
    sources = list(AppContext._iter_image_sources(response))
    assert {"inline": "data:image/png;base64,abc"} in sources


def test_iter_image_sources_annotation_file_ids():
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "see image",
                        "annotations": [
                            {"type": "file_citation", "file_id": "f3", "container_id": "c3"},
                        ],
                    },
                ],
            }
        ]
    }
    sources = list(AppContext._iter_image_sources(response))
    assert {"file_id": "f3", "container_id": "c3"} in sources


def test_iter_image_sources_direct_file_id_on_image_gen_call():
    response = {
        "output": [
            {
                "type": "image_generation_call",
                "file_id": "direct_f",
                "container_id": "direct_c",
                "result": None,
            }
        ]
    }
    sources = list(AppContext._iter_image_sources(response))
    assert {"file_id": "direct_f", "container_id": "direct_c"} in sources


def test_iter_image_sources_empty_output():
    assert list(AppContext._iter_image_sources({})) == []
    assert list(AppContext._iter_image_sources({"output": []})) == []
    assert list(AppContext._iter_image_sources({"output": None})) == []


# --- _decode_base64_image ---

def test_decode_base64_image_raw():
    encoded = base64.b64encode(b"hello").decode()
    assert AppContext._decode_base64_image(encoded) == b"hello"


def test_decode_base64_image_data_uri():
    encoded = base64.b64encode(b"pixels").decode()
    data_uri = f"data:image/png;base64,{encoded}"
    assert AppContext._decode_base64_image(data_uri) == b"pixels"


# --- clean_response_text ---

def test_clean_response_text_strips_think_tags():
    ctx = _ctx()
    try:
        result = ctx.clean_response_text(
            "<think>internal thought</think>Hello world",
            sender_display="User",
            sender_id="@u",
        )
        assert result == "Hello world"
        assert "<think>" not in result
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_clean_response_text_strips_begin_of_thought_tags():
    ctx = _ctx()
    try:
        result = ctx.clean_response_text(
            "<|begin_of_thought|>deep thought<|end_of_thought|>Actual answer",
            sender_display="User",
            sender_id="@u",
        )
        assert result == "Actual answer"
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_clean_response_text_strips_solution_tags():
    ctx = _ctx()
    try:
        result = ctx.clean_response_text(
            "prefix<|begin_of_solution|>The answer is 42<|end_of_solution|>suffix",
            sender_display="User",
            sender_id="@u",
        )
        assert result == "The answer is 42"
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_clean_response_text_handles_empty_and_none():
    ctx = _ctx()
    try:
        assert ctx.clean_response_text("", sender_display="U", sender_id="@u") == ""
        assert ctx.clean_response_text("  ", sender_display="U", sender_id="@u") == ""
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_clean_response_text_no_tags_returns_stripped():
    ctx = _ctx()
    try:
        result = ctx.clean_response_text("  just text  ", sender_display="U", sender_id="@u")
        assert result == "just text"
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


# --- _strip_inline_citations ---

def test_strip_inline_citations_keeps_bracket_citations():
    text = "Answer 【abc†source】 and more 〖def†source〗"
    result = AppContext._strip_inline_citations(text)
    assert result == text


def test_strip_inline_citations_keeps_annotation_text():
    text = "Answer [1] text [2] more"
    annotations = [
        {"type": "url_citation", "text": " [1]"},
        {"type": "url_citation", "text": " [2]"},
    ]
    result = AppContext._strip_inline_citations(text, annotations)
    assert result == text


def test_strip_inline_citations_keeps_whitespace():
    text = "word  ,  extra   spaces"
    result = AppContext._strip_inline_citations(text)
    assert result == text


def test_strip_inline_citations_ignores_non_citation_annotations():
    text = "hello"
    annotations = [{"type": "some_other_type", "text": "hello"}]
    result = AppContext._strip_inline_citations(text, annotations)
    assert result == "hello"


def test_strip_inline_citations_handles_non_dict_annotations():
    text = "hello"
    annotations = ["not a dict", None, 42]
    result = AppContext._strip_inline_citations(text, annotations)
    assert result == "hello"


# --- _extract_text ---

def test_extract_text_from_output_items():
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Part 1"},
                    {"type": "output_text", "text": "Part 2"},
                ],
            }
        ]
    }
    assert AppContext._extract_text(response) == "Part 1\nPart 2"


def test_extract_text_falls_back_to_output_text():
    response = {"output_text": "Fallback text 【x†source】"}
    result = AppContext._extract_text(response)
    assert result == "Fallback text 【x†source】"


def test_extract_text_empty_response():
    assert AppContext._extract_text({}) == ""
    assert AppContext._extract_text({"output": []}) == ""


# --- _build_hosted_tool edge cases ---

def test_build_hosted_tool_returns_none_for_false():
    ctx = _ctx()
    try:
        result = ctx._build_hosted_tool("openai", "web_search", False)
        assert result is None
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_build_hosted_tool_returns_none_for_none():
    ctx = _ctx()
    try:
        result = ctx._build_hosted_tool("openai", "web_search", None)
        assert result is None
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_build_hosted_tool_invalid_spec_type_returns_none():
    ctx = _ctx()
    try:
        result = ctx._build_hosted_tool("openai", "web_search", 42)
        assert result is None
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_build_hosted_tool_dict_spec_merges():
    ctx = _ctx()
    try:
        result = ctx._build_hosted_tool("openai", "web_search", {"search_context_size": "high"})
        assert result["type"] == "web_search"
        assert result["search_context_size"] == "high"
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_build_hosted_tool_code_interpreter_adds_container_for_openai():
    ctx = _ctx()
    try:
        result = ctx._build_hosted_tool("openai", "code_interpreter", True)
        assert result["type"] == "code_interpreter"
        assert result["container"] == {"type": "auto"}
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_build_hosted_tool_code_interpreter_no_container_for_xai():
    """xai provider should not get auto container on code_interpreter."""
    cfg = AppConfig(
        llm=LLMConfig(
            models={"xai": ["grok-4"]},
            api_keys={"xai": "X"},
            default_model="grok-4",
            personality="p",
            prompt=["you are ", "."],
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admins=[]),
    )
    ctx = AppContext(cfg)
    try:
        result = ctx._build_hosted_tool("xai", "code_interpreter", True)
        assert result["type"] == "code_interpreter"
        assert "container" not in result
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_xai_video_generation_tool_can_be_disabled():
    cfg = AppConfig(
        llm=LLMConfig(
            models={"xai": ["grok-4"]},
            api_keys={"xai": "X"},
            default_model="grok-4",
            personality="p",
            prompt=["you are ", "."],
            tools={
                "image_generation": True,
                "video_generation": False,
            },
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admins=[]),
    )
    ctx = AppContext(cfg)
    try:
        function_names = {tool["name"] for tool in ctx._tools_for_model("grok-4") if tool["type"] == "function"}
        assert function_names == {"grok_generate_image", "grok_edit_image"}
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


# --- _build_mcp_tool edge cases ---

def test_build_mcp_tool_invalid_spec_returns_none():
    ctx = _ctx()
    try:
        result = ctx._build_mcp_tool("openai", "test", "not a dict")
        assert result is None
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_build_mcp_tool_missing_server_url_and_connector_returns_none():
    ctx = _ctx()
    try:
        result = ctx._build_mcp_tool("openai", "test", {"server_description": "desc"})
        assert result is None
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_build_mcp_tool_authorization_env(monkeypatch):
    """When authorization_env is set and env var exists, Bearer token is added."""
    monkeypatch.setenv("MY_MCP_TOKEN", "secret123")
    ctx = _ctx()
    try:
        result = ctx._build_mcp_tool("openai", "test", {
            "server_url": "https://mcp.example.com",
            "authorization_env": "MY_MCP_TOKEN",
        })
        assert result["authorization"] == "Bearer secret123"
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_build_mcp_tool_authorization_env_missing_env_var(monkeypatch):
    """When authorization_env is set but env var is empty, no authorization added."""
    monkeypatch.delenv("MY_MCP_TOKEN", raising=False)
    ctx = _ctx()
    try:
        result = ctx._build_mcp_tool("openai", "test", {
            "server_url": "https://mcp.example.com",
            "authorization_env": "MY_MCP_TOKEN",
        })
        assert "authorization" not in result
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_build_mcp_tool_explicit_authorization_takes_precedence(monkeypatch):
    """Explicit authorization in spec should not be overridden by authorization_env."""
    monkeypatch.setenv("MY_MCP_TOKEN", "secret123")
    ctx = _ctx()
    try:
        result = ctx._build_mcp_tool("openai", "test", {
            "server_url": "https://mcp.example.com",
            "authorization": "Bearer explicit",
            "authorization_env": "MY_MCP_TOKEN",
        })
        assert result["authorization"] == "Bearer explicit"
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_build_mcp_tool_with_connector_id():
    ctx = _ctx()
    try:
        result = ctx._build_mcp_tool("openai", "test", {
            "connector_id": "conn_123",
        })
        assert result is not None
        assert result["connector_id"] == "conn_123"
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


# --- _iter_image_sources container_id propagation ---

def test_iter_image_sources_propagates_container_id_to_walk_results():
    """When image_generation_call has container_id but result dict doesn't, it should be added."""
    response = {
        "output": [
            {
                "type": "image_generation_call",
                "container_id": "c_from_call",
                "result": {"file_id": "f1"},
            }
        ]
    }
    sources = list(AppContext._iter_image_sources(response))
    file_source = [s for s in sources if s.get("file_id") == "f1"][0]
    assert file_source["container_id"] == "c_from_call"


# --- _send_response_artifacts ---

def test_send_response_artifacts_returns_false_when_no_room_id():
    ctx = _ctx()
    try:
        sent = asyncio.run(ctx._send_response_artifacts({"output": []}, None, provider="openai"))
        assert sent is False
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_handle_generate_image_calls_supports_edit_image():
    ctx = _ctx()
    try:
        class FakeMediaLLM:
            async def edit_image(self, **payload):
                assert payload["prompt"] == "make it cinematic"
                assert payload["image_urls"] == ["https://example.com/source.png"]
                return {"data": [{"b64_json": base64.b64encode(b"edited").decode()}]}

        ctx.llm = FakeMediaLLM()
        response = {
            "output": [
                {
                    "type": "function_call",
                    "name": "grok_edit_image",
                    "call_id": "call_1",
                    "arguments": '{"prompt":"make it cinematic","image_url":"https://example.com/source.png"}',
                }
            ]
        }
        output_items = asyncio.run(ctx._handle_generate_image_calls(response, model="grok-4", room_id="!r"))
        assert output_items == [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Image edited and sent.",
            }
        ]
        assert len(ctx.matrix.sent_images) == 1
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_handle_generate_image_calls_uses_latest_generated_image_when_edit_input_missing():
    ctx = _ctx()
    try:
        ctx._remember_generated_media(
            "!r",
            "@u",
            kind="image",
            reference="data:image/png;base64,c291cmNl",
            mime_type="image/png",
        )

        class FakeMediaLLM:
            async def edit_image(self, **payload):
                assert payload["image_urls"] == ["data:image/png;base64,c291cmNl"]
                return {"data": [{"b64_json": base64.b64encode(b"edited").decode()}]}

        ctx.llm = FakeMediaLLM()
        response = {
            "output": [
                {
                    "type": "function_call",
                    "name": "grok_edit_image",
                    "call_id": "call_implicit",
                    "arguments": '{"prompt":"make it warmer"}',
                }
            ]
        }
        output_items = asyncio.run(
            ctx._handle_generate_image_calls(response, model="grok-4", room_id="!r", thread_user="@u")
        )
        assert output_items == [
            {
                "type": "function_call_output",
                "call_id": "call_implicit",
                "output": "Image edited and sent.",
            }
        ]
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_handle_generate_image_calls_supports_grok_generate_video():
    ctx = _ctx()
    try:
        class FakeMediaLLM:
            async def generate_video(self, **payload):
                assert payload["prompt"] == "pan slowly across the room"
                assert payload["image_url"] == "https://example.com/source.png"
                return {"url": "https://cdn.example.com/output.mp4"}

            async def download_url(self, url, provider=None):
                assert url == "https://cdn.example.com/output.mp4"
                assert provider == "xai"
                return b"mp4-bytes"

        ctx.llm = FakeMediaLLM()
        response = {
            "output": [
                {
                    "type": "function_call",
                    "name": "grok_generate_video",
                    "call_id": "call_2",
                    "arguments": '{"prompt":"pan slowly across the room","image_url":"https://example.com/source.png","duration":5}',
                }
            ]
        }
        output_items = asyncio.run(ctx._handle_generate_image_calls(response, model="grok-4", room_id="!r"))
        assert output_items == [
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": "Video generated and sent.",
            }
        ]
        assert len(ctx.matrix.sent_videos) == 1
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_handle_generate_image_calls_supports_sora_generate_video():
    ctx = _ctx()
    try:
        class FakeMediaLLM:
            async def generate_video(self, **payload):
                assert payload["prompt"] == "animate this logo"
                assert payload["image_url"] == "data:image/png;base64,c291cmNl"
                assert payload["seconds"] == 8
                assert payload["size"] == "1280x720"
                assert payload["backend"] == "sora"
                return {"id": "vid_openai"}

            async def download_video_content(self, video_id, *, provider):
                assert video_id == "vid_openai"
                assert provider == "openai"
                return b"openai-video"

        ctx.llm = FakeMediaLLM()
        ctx._remember_generated_media(
            "!r",
            "@u",
            kind="image",
            reference="data:image/png;base64,c291cmNl",
            mime_type="image/png",
        )
        response = {
            "output": [
                {
                    "type": "function_call",
                    "name": "sora_generate_video",
                    "call_id": "call_openai_video",
                    "arguments": '{"prompt":"animate this logo","seconds":8,"size":"1280x720"}',
                }
            ]
        }
        output_items = asyncio.run(
            ctx._handle_generate_image_calls(response, model="gpt-5-mini", room_id="!r", thread_user="@u")
        )
        assert output_items == [
            {
                "type": "function_call_output",
                "call_id": "call_openai_video",
                "output": "Video generated and sent.",
            }
        ]
        assert len(ctx.matrix.sent_videos) == 1
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_handle_generate_image_calls_supports_sora_generate_video_from_xai_model():
    ctx = _ctx()
    try:
        class FakeMediaLLM:
            async def generate_video(self, **payload):
                assert payload["backend"] == "sora"
                assert payload["model"] == "grok-4"
                return {"id": "vid_cross_provider"}

            async def download_video_content(self, video_id, *, provider):
                assert video_id == "vid_cross_provider"
                assert provider == "openai"
                return b"openai-video"

        ctx.llm = FakeMediaLLM()
        response = {
            "output": [
                {
                    "type": "function_call",
                    "name": "sora_generate_video",
                    "call_id": "call_sora_backend",
                    "arguments": '{"prompt":"turn this into a product video","image_url":"data:image/png;base64,c291cmNl","seconds":4,"size":"1280x720"}',
                }
            ]
        }
        output_items = asyncio.run(
            ctx._handle_generate_image_calls(response, model="grok-4", room_id="!r", thread_user="@u")
        )
        assert output_items == [
            {
                "type": "function_call_output",
                "call_id": "call_sora_backend",
                "output": "Video generated and sent.",
            }
        ]
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_generate_reply_adds_media_context_note_for_thread():
    cfg = AppConfig(
        llm=LLMConfig(
            models={"xai": ["grok-4"]},
            api_keys={"xai": "X"},
            default_model="grok-4",
            personality="p",
            prompt=["you are ", "."],
            tools={"image_generation": True},
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admins=[]),
    )
    ctx = AppContext(cfg)
    try:
        ctx._remember_generated_media(
            "!r",
            "@u",
            kind="image",
            reference="data:image/png;base64,c291cmNl",
            mime_type="image/png",
        )
        capture = CaptureLLM()
        ctx.llm = capture
        out = asyncio.run(
            ctx.generate_reply(
                [{"role": "system", "content": "be concise"}, {"role": "user", "content": "edit that image"}],
                model="grok-4",
                room_id="!r",
                thread_user="@u",
                use_tools=True,
            )
        )
        assert out == "ok"
        assert "Recent generated media is available in this thread." in capture.last_payload["messages"][0]["content"]
        assert "runtime will supply it" in capture.last_payload["messages"][0]["content"]
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


# --- _approval_items ---

def test_approval_items_extracts_approval_requests():
    ctx = _ctx()
    try:
        response = {
            "output": [
                {"type": "mcp_approval_request", "id": "a1", "server_label": "test"},
                {"type": "message", "content": []},
                {"type": "mcp_approval_request_item", "id": "a2", "server_label": "test"},
            ]
        }
        items = ctx._approval_items(response)
        assert len(items) == 2
        assert items[0]["id"] == "a1"
        assert items[1]["id"] == "a2"
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_approval_items_empty_output():
    ctx = _ctx()
    try:
        assert ctx._approval_items({}) == []
        assert ctx._approval_items({"output": None}) == []
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


# --- _should_auto_approve ---

def test_should_auto_approve_returns_false_for_unknown_label():
    ctx = _ctx()
    try:
        assert ctx._should_auto_approve({"server_label": "unknown"}) is False
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


# --- _configured_providers ---

def test_configured_providers_excludes_unconfigured():
    """Only providers with API keys (openai/xai) or base_urls (lmstudio) are returned."""
    cfg = AppConfig(
        llm=LLMConfig(
            models={"openai": ["gpt-5-mini"], "xai": [], "lmstudio": []},
            api_keys={"openai": "X", "xai": ""},
            base_urls={"lmstudio": ""},
            default_model="gpt-5-mini",
            personality="p",
            prompt=["you are ", "."],
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admins=[]),
    )
    ctx = AppContext(cfg)
    try:
        providers = ctx._configured_providers()
        assert "openai" in providers
        assert "xai" not in providers
        assert "lmstudio" not in providers
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)


def test_lmstudio_only_exposes_mcp_tools():
    cfg = AppConfig(
        llm=LLMConfig(
            models={"lmstudio": ["local-model"]},
            api_keys={},
            base_urls={"lmstudio": "http://127.0.0.1:1234/v1"},
            default_model="local-model",
            personality="p",
            prompt=["you are ", "."],
            tools={
                "web_search": True,
                "x_search": True,
                "code_interpreter": True,
                "image_generation": True,
            },
            mcp_servers={
                "deepwiki": {
                    "server_url": "http://127.0.0.1:8765/mcp",
                }
            },
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admins=[]),
    )
    ctx = AppContext(cfg)
    try:
        assert ctx.hosted_tools == [
            {
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "http://127.0.0.1:8765/mcp",
            }
        ]
    finally:
        ctx.executor.shutdown(wait=False, cancel_futures=True)
