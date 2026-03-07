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
