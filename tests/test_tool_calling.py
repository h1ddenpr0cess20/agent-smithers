import asyncio

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
        assert payload["previous_response_id"] == "resp_1"
        assert payload["input_items"] == [
            {
                "type": "mcp_approval_response",
                "approval_request_id": "approve_1",
                "approve": True,
            }
        ]
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

    async def send_image(self, room_id, path, filename=None, log=None):
        self.sent_images.append((room_id, path, filename))


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


def test_extract_text_strips_inline_citation_markers():
    response = {
        "output_text": "Answer with sources【abc†source】",
    }
    assert AppContext._extract_text(response) == "Answer with sources"


def test_extract_text_strips_annotation_text_from_output_items():
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


def test_strip_inline_citations_removes_annotation_text():
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
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admins=[]),
    )
    ctx = AppContext(cfg)
    try:
        openai_tools = ctx._tools_for_model("gpt-5-mini")
        xai_tools = ctx._tools_for_model("grok-4")
        assert {"type": "image_generation"} in openai_tools
        assert {"type": "x_search"} not in openai_tools
        assert {"type": "x_search"} in xai_tools
        assert {"type": "image_generation"} not in xai_tools
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
