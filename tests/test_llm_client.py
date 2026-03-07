import asyncio
import base64
from io import BytesIO

from PIL import Image
from agent_smithers.config import AppConfig, LLMConfig, MatrixConfig
from agent_smithers.llm_client import LLMClient


def _cfg(default_model="gpt-5-mini"):
    llm = LLMConfig(
        models={"openai": ["gpt-5-mini"], "xai": ["grok-4"]},
        api_keys={"openai": "O", "xai": "X"},
        default_model=default_model,
        personality="p",
        prompt=["you are ", "."],
    )
    matrix = MatrixConfig(server="s", username="u", password="p", channels=["!r"], admin="a")
    return AppConfig(llm=llm, matrix=matrix)


class FakeResponse:
    def __init__(self, payload=None, content=b""):
        self._payload = payload or {}
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class RecordingAsyncClient:
    responses = []
    requests = []

    def __init__(self, *args, **kwargs):
        del args, kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False

    async def post(self, url, headers=None, json=None, files=None, data=None):
        self.requests.append(("POST", url, headers, {"json": json, "files": files, "data": data}))
        return self.responses.pop(0)

    async def get(self, url, headers=None):
        self.requests.append(("GET", url, headers, None))
        return self.responses.pop(0)


def _png_data_uri(width=64, height=64, color=(255, 0, 0, 255)):
    image = Image.new("RGBA", (width, height), color)
    buf = BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def test_build_input_items_splits_system_messages():
    instructions, input_items = LLMClient.build_input_items(
        [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    )
    assert instructions == "be concise"
    assert input_items == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_build_request_payload_uses_responses_shape():
    client = LLMClient(_cfg())
    payload = client.build_request_payload(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "hello"},
        ],
        tools=[{"type": "web_search"}],
        options={"temperature": 0.2},
    )
    assert payload["model"] == "gpt-5-mini"
    assert payload["instructions"] == "be concise"
    assert payload["input"] == [{"role": "user", "content": "hello"}]
    assert payload["tools"] == [{"type": "web_search"}]
    assert payload["tool_choice"] == "auto"
    assert payload["temperature"] == 0.2


def test_is_chat_model_filters_only_requested_variants():
    client = LLMClient(_cfg())
    allowed = [
        "gpt-4o",
        "gpt-5-mini",
        "gpt-realtime",
        "o3-pro",
    ]
    blocked = [
        "gpt-4.1-2025-04-14",
        "gpt-4o-audio-preview",
        "computer-use-preview",
        "gpt-4o-mini-transcribe",
        "gpt-4-turbo-preview",
        "gpt-4o-mini-tts",
        "gpt-image-1",
    ]
    for model_id in allowed:
        assert client._is_chat_model("openai", model_id) is True, model_id
    for model_id in blocked:
        assert client._is_chat_model("openai", model_id) is False, model_id


def test_build_request_payload_keeps_system_messages_in_input_for_xai():
    client = LLMClient(_cfg("grok-4"))
    payload = client.build_request_payload(
        model="grok-4",
        messages=[
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "hello"},
        ],
        tools=[{"type": "x_search"}],
    )
    assert payload["model"] == "grok-4"
    assert "instructions" not in payload
    assert payload["input"] == [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "hello"},
    ]
    assert payload["tools"] == [{"type": "x_search"}]
    assert payload["tool_choice"] == "auto"
    assert payload["include"] == ["no_inline_citations"]


def test_build_request_payload_merges_xai_include_items():
    client = LLMClient(_cfg("grok-4"))
    payload = client.build_request_payload(
        model="grok-4",
        messages=[
            {"role": "user", "content": "hello"},
        ],
        tools=[{"type": "web_search"}],
        options={"include": ["reasoning.encrypted_content"]},
    )
    assert payload["include"] == ["reasoning.encrypted_content", "no_inline_citations"]


def test_is_chat_model_filters_xai_non_chat_variants():
    client = LLMClient(_cfg("grok-4"))
    assert client._is_chat_model("xai", "grok-4") is True
    assert client._is_chat_model("xai", "grok-3-mini") is True
    assert client._is_chat_model("xai", "grok-imagine-image") is False


def test_lmstudio_uses_configured_base_url_and_no_auth_header():
    cfg = AppConfig(
        llm=LLMConfig(
            models={"lmstudio": ["local-model"]},
            api_keys={"lmstudio": ""},
            base_urls={"lmstudio": "http://127.0.0.1:1234/v1"},
            default_model="local-model",
            personality="p",
            prompt=["you are ", "."],
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admin="a"),
    )
    client = LLMClient(cfg)
    assert client._base_url("lmstudio") == "http://127.0.0.1:1234/v1"
    assert client._headers("lmstudio") == {"Content-Type": "application/json"}
    assert client._is_chat_model("lmstudio", "local-model") is True
    assert client._is_chat_model("lmstudio", "text-embedding-nomic-embed-text-v1.5") is False


def test_build_request_payload_adds_lmstudio_fallback_user_turn_when_missing():
    cfg = AppConfig(
        llm=LLMConfig(
            models={"lmstudio": ["local-model"]},
            api_keys={"lmstudio": ""},
            base_urls={"lmstudio": "http://127.0.0.1:1234/v1"},
            default_model="local-model",
            personality="p",
            prompt=["you are ", "."],
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admin="a"),
    )
    client = LLMClient(cfg)
    payload = client.build_request_payload(
        model="local-model",
        messages=[
            {"role": "system", "content": "be concise"},
            {"role": "assistant", "content": "Earlier reply"},
        ],
    )
    assert payload["instructions"] == "be concise"
    assert payload["input"] == [
        {"role": "assistant", "content": "Earlier reply"},
        {"role": "user", "content": LLMClient.LMSTUDIO_FALLBACK_USER_PROMPT},
    ]


def test_build_request_payload_keeps_existing_lmstudio_user_turn():
    cfg = AppConfig(
        llm=LLMConfig(
            models={"lmstudio": ["local-model"]},
            api_keys={"lmstudio": ""},
            base_urls={"lmstudio": "http://127.0.0.1:1234/v1"},
            default_model="local-model",
            personality="p",
            prompt=["you are ", "."],
        ),
        matrix=MatrixConfig(server="s", username="u", password="p", channels=["!r"], admin="a"),
    )
    client = LLMClient(cfg)
    payload = client.build_request_payload(
        model="local-model",
        messages=[
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "hello"},
        ],
    )
    assert payload["input"] == [{"role": "user", "content": "hello"}]


# --- build_input_items edge cases ---

def test_build_input_items_include_system_true_puts_system_in_input():
    instructions, input_items = LLMClient.build_input_items(
        [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "hello"},
        ],
        include_system=True,
    )
    assert instructions is None  # no instructions extracted
    assert input_items == [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "hello"},
    ]


def test_build_input_items_skips_empty_role_or_content():
    instructions, input_items = LLMClient.build_input_items(
        [
            {"role": "", "content": "ignored"},
            {"role": "user", "content": ""},
            {"role": "user", "content": "real"},
        ]
    )
    assert input_items == [{"role": "user", "content": "real"}]
    assert instructions is None


def test_build_input_items_multiple_system_messages_joined():
    instructions, input_items = LLMClient.build_input_items(
        [
            {"role": "system", "content": "rule 1"},
            {"role": "system", "content": "rule 2"},
            {"role": "user", "content": "go"},
        ]
    )
    assert instructions == "rule 1\n\nrule 2"
    assert len(input_items) == 1


def test_build_input_items_ignores_unknown_roles():
    """Roles other than system/user/assistant are silently dropped."""
    instructions, input_items = LLMClient.build_input_items(
        [
            {"role": "function", "content": "data"},
            {"role": "user", "content": "hello"},
        ]
    )
    assert input_items == [{"role": "user", "content": "hello"}]


# --- build_request_payload edge cases ---

def test_build_request_payload_previous_response_id_suppresses_instructions():
    client = LLMClient(_cfg())
    payload = client.build_request_payload(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "instructions here"},
            {"role": "user", "content": "hello"},
        ],
        previous_response_id="resp_123",
    )
    assert "instructions" not in payload
    assert payload["previous_response_id"] == "resp_123"


def test_build_request_payload_no_tools_omits_tool_keys():
    client = LLMClient(_cfg())
    payload = client.build_request_payload(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert "tools" not in payload
    assert "tool_choice" not in payload


def test_build_request_payload_with_input_items_instead_of_messages():
    client = LLMClient(_cfg())
    payload = client.build_request_payload(
        model="gpt-5-mini",
        input_items=[
            {"type": "mcp_approval_response", "approve": True},
        ],
        previous_response_id="resp_1",
    )
    assert payload["input"] == [{"type": "mcp_approval_response", "approve": True}]
    assert payload["previous_response_id"] == "resp_1"


def test_build_request_payload_accepts_explicit_instructions_with_input_items():
    client = LLMClient(_cfg())
    payload = client.build_request_payload(
        model="gpt-5-mini",
        input_items=[
            {"role": "user", "content": "hello"},
        ],
        instructions="be concise",
    )
    assert payload["instructions"] == "be concise"
    assert payload["input"] == [{"role": "user", "content": "hello"}]


def test_generate_video_reports_status_updates_during_polling(monkeypatch):
    client = LLMClient(_cfg("gpt-5-mini"))
    client.VIDEO_POLL_INTERVAL_SECONDS = 0
    RecordingAsyncClient.responses = [
        FakeResponse({"id": "vid_1", "status": "queued"}),
        FakeResponse({"id": "vid_1", "status": "processing"}),
        FakeResponse({"id": "vid_1", "status": "completed", "url": "https://example.com/video.mp4"}),
    ]
    RecordingAsyncClient.requests = []
    status_updates = []
    monkeypatch.setattr("agent_smithers.llm_client.httpx.AsyncClient", RecordingAsyncClient)

    result = asyncio.run(
        client.generate_video(
            prompt="hello",
            model="gpt-5-mini",
            backend="sora",
            on_status=status_updates.append,
        )
    )

    assert result["id"] == "vid_1"
    assert status_updates == [
        "Generating video with Sora [queued]",
        "Generating video with Sora [processing]",
        "Generating video with Sora [completed]",
    ]


def test_build_request_payload_options_none_values_omitted():
    client = LLMClient(_cfg())
    payload = client.build_request_payload(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "hi"}],
        options={"temperature": 0.5, "top_p": None},
    )
    assert payload["temperature"] == 0.5
    assert "top_p" not in payload


# --- _is_chat_model edge cases ---

def test_is_chat_model_openai_date_suffix_blocked():
    assert LLMClient._is_chat_model("openai", "gpt-4.1-2025-04-14") is False


def test_is_chat_model_openai_non_gpt_prefix_blocked():
    assert LLMClient._is_chat_model("openai", "dall-e-3") is False
    assert LLMClient._is_chat_model("openai", "whisper-1") is False


def test_is_chat_model_xai_non_grok_blocked():
    assert LLMClient._is_chat_model("xai", "something-else") is False


def test_is_chat_model_xai_vision_blocked():
    assert LLMClient._is_chat_model("xai", "grok-vision-beta") is False


def test_is_chat_model_lmstudio_empty_string():
    assert LLMClient._is_chat_model("lmstudio", "") is False
    assert LLMClient._is_chat_model("lmstudio", "   ") is False


# --- _fallback_base_url ---

def test_fallback_base_url_values():
    assert LLMClient._fallback_base_url("lmstudio") == "http://127.0.0.1:1234/v1"
    assert LLMClient._fallback_base_url("xai") == "https://api.x.ai/v1"
    assert LLMClient._fallback_base_url("openai") == "https://api.openai.com/v1"


# --- _has_user_message ---

def test_has_user_message_true():
    assert LLMClient._has_user_message([{"role": "user", "content": "hi"}]) is True


def test_has_user_message_false_empty_content():
    assert LLMClient._has_user_message([{"role": "user", "content": ""}]) is False


def test_has_user_message_false_no_items():
    assert LLMClient._has_user_message([]) is False


def test_has_user_message_skips_non_dict():
    assert LLMClient._has_user_message(["not a dict", None]) is False


# --- _headers ---

def test_headers_includes_bearer_when_key_present():
    client = LLMClient(_cfg())
    headers = client._headers("openai")
    assert headers["Authorization"] == "Bearer O"


def test_headers_no_auth_when_key_empty():
    client = LLMClient(_cfg())
    headers = client._headers("lmstudio")
    assert "Authorization" not in headers


# --- _merge_include_items ---

def test_merge_include_items_deduplicates():
    result = LLMClient._merge_include_items(
        ["a", "b"],
        ["b", "c"],
    )
    assert result == ["a", "b", "c"]


def test_merge_include_items_non_list_existing():
    result = LLMClient._merge_include_items(None, ["a"])
    assert result == ["a"]


def test_edit_image_posts_to_xai_edits_endpoint(monkeypatch):
    RecordingAsyncClient.requests = []
    RecordingAsyncClient.responses = [FakeResponse({"data": [{"b64_json": "abc"}]})]
    monkeypatch.setattr("agent_smithers.llm_client.httpx.AsyncClient", RecordingAsyncClient)
    client = LLMClient(_cfg("grok-4"))
    result = asyncio.run(
        client.edit_image(
            prompt="remove the background",
            image_urls=["https://example.com/source.png"],
            model="grok-4",
            resolution="2k",
        )
    )
    assert result["data"][0]["b64_json"] == "abc"
    method, url, headers, request = RecordingAsyncClient.requests[0]
    payload = request["json"]
    assert method == "POST"
    assert url == "https://api.x.ai/v1/images/edits"
    assert headers["Authorization"] == "Bearer X"
    assert payload["model"] == "grok-imagine-image"
    assert payload["prompt"] == "remove the background"
    assert payload["image"]["url"] == "https://example.com/source.png"
    assert payload["image"]["type"] == "image_url"
    assert payload["resolution"] == "2k"


def test_generate_video_polls_until_completed(monkeypatch):
    RecordingAsyncClient.requests = []
    RecordingAsyncClient.responses = [
        FakeResponse({"id": "vid_1", "status": "pending"}),
        FakeResponse({"id": "vid_1", "status": "completed", "url": "https://cdn.example.com/out.mp4"}),
    ]
    monkeypatch.setattr("agent_smithers.llm_client.httpx.AsyncClient", RecordingAsyncClient)

    async def _noop_sleep(_seconds):
        return None

    monkeypatch.setattr("agent_smithers.llm_client.asyncio.sleep", _noop_sleep)
    client = LLMClient(_cfg("grok-4"))
    result = asyncio.run(
        client.generate_video(
            prompt="slow dolly shot",
            model="grok-4",
            image_url="https://example.com/source.png",
            duration=5,
            aspect_ratio="16:9",
            resolution="720p",
        )
    )
    assert result["url"] == "https://cdn.example.com/out.mp4"
    post_method, post_url, _, request = RecordingAsyncClient.requests[0]
    post_payload = request["json"]
    get_method, get_url, _, _ = RecordingAsyncClient.requests[1]
    assert post_method == "POST"
    assert post_url == "https://api.x.ai/v1/videos/generations"
    assert post_payload["image"]["url"] == "https://example.com/source.png"
    assert post_payload["duration"] == 5
    assert post_payload["aspect_ratio"] == "16:9"
    assert post_payload["resolution"] == "720p"
    assert get_method == "GET"
    assert get_url == "https://api.x.ai/v1/videos/vid_1"


def test_generate_video_uses_edits_endpoint_for_existing_video(monkeypatch):
    RecordingAsyncClient.requests = []
    RecordingAsyncClient.responses = [
        FakeResponse({"id": "vid_2", "status": "completed", "url": "https://cdn.example.com/edited.mp4"})
    ]
    monkeypatch.setattr("agent_smithers.llm_client.httpx.AsyncClient", RecordingAsyncClient)
    client = LLMClient(_cfg("grok-4"))
    result = asyncio.run(
        client.generate_video(
            prompt="add dramatic lighting",
            model="grok-4",
            video_url="https://example.com/input.mp4",
        )
    )
    assert result["url"] == "https://cdn.example.com/edited.mp4"
    method, url, _, request = RecordingAsyncClient.requests[0]
    payload = request["json"]
    assert method == "POST"
    assert url == "https://api.x.ai/v1/videos/generations"
    assert payload["video_url"] == "https://example.com/input.mp4"
    assert "duration" not in payload


def test_generate_video_uses_openai_sora_multipart(monkeypatch):
    RecordingAsyncClient.requests = []
    RecordingAsyncClient.responses = [
        FakeResponse({"id": "vid_openai", "status": "queued"}),
        FakeResponse({"id": "vid_openai", "status": "completed"}),
        FakeResponse(content=b"video-bytes"),
    ]
    monkeypatch.setattr("agent_smithers.llm_client.httpx.AsyncClient", RecordingAsyncClient)

    async def _noop_sleep(_seconds):
        return None

    monkeypatch.setattr("agent_smithers.llm_client.asyncio.sleep", _noop_sleep)
    client = LLMClient(_cfg("gpt-5-mini"))
    result = asyncio.run(
        client.generate_video(
            prompt="camera orbit around the statue",
            model="gpt-5-mini",
            image_url=_png_data_uri(512, 512),
            seconds=8,
            size="1280x720",
        )
    )
    assert result["id"] == "vid_openai"
    post_method, post_url, post_headers, post_request = RecordingAsyncClient.requests[0]
    get_method, get_url, _, _ = RecordingAsyncClient.requests[1]
    assert post_method == "POST"
    assert post_url == "https://api.openai.com/v1/videos"
    assert "Content-Type" not in post_headers
    multipart = post_request["files"]
    assert ("model", (None, "sora-2", None)) in multipart
    assert ("prompt", (None, "camera orbit around the statue", None)) in multipart
    assert ("seconds", (None, "8", None)) in multipart
    assert ("size", (None, "1280x720", None)) in multipart
    file_part = next(item for item in multipart if item[0] == "input_reference")
    assert file_part[1][0].startswith("input_reference")
    assert file_part[1][2] == "image/png"
    prepared_image = Image.open(BytesIO(file_part[1][1]))
    assert prepared_image.size == (1280, 720)
    assert get_method == "GET"
    assert get_url == "https://api.openai.com/v1/videos/vid_openai"


def test_generate_video_infers_openai_sora_size_from_input_image(monkeypatch):
    RecordingAsyncClient.requests = []
    RecordingAsyncClient.responses = [
        FakeResponse({"id": "vid_openai", "status": "completed"}),
    ]
    monkeypatch.setattr("agent_smithers.llm_client.httpx.AsyncClient", RecordingAsyncClient)
    client = LLMClient(_cfg("gpt-5-mini"))
    result = asyncio.run(
        client.generate_video(
            prompt="portrait fly-through",
            model="gpt-5-mini",
            image_url=_png_data_uri(720, 1280),
            seconds=4,
        )
    )
    assert result["id"] == "vid_openai"
    method, url, headers, request = RecordingAsyncClient.requests[0]
    assert method == "POST"
    assert url == "https://api.openai.com/v1/videos"
    assert "Content-Type" not in headers
    multipart = request["files"]
    assert ("size", (None, "720x1280", None)) in multipart


def test_download_video_content_uses_openai_content_endpoint(monkeypatch):
    RecordingAsyncClient.requests = []
    RecordingAsyncClient.responses = [FakeResponse(content=b"video-bytes")]
    monkeypatch.setattr("agent_smithers.llm_client.httpx.AsyncClient", RecordingAsyncClient)
    client = LLMClient(_cfg("gpt-5-mini"))
    payload = asyncio.run(client.download_video_content("vid_openai", provider="openai"))
    assert payload == b"video-bytes"
    method, url, headers, _ = RecordingAsyncClient.requests[0]
    assert method == "GET"
    assert url == "https://api.openai.com/v1/videos/vid_openai/content"
    assert headers["Authorization"] == "Bearer O"
