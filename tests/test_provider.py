"""Provider behaviour against a fake DashScope socket — no network, no credentials, no flakes."""

from __future__ import annotations

import time

import pytest
from conftest import requires_audio, requires_sdk

from transcription_dashscope import provider as provider_module

DEFAULT_MODEL = "qwen-audio-3.1-asr-flash-message"


# ── configuration resolution ──

def test_socket_urls_are_passed_through_http_bases_are_not():
    assert provider_module._ws_override("wss://a.test/api-ws/v1/inference") == "wss://a.test/api-ws/v1/inference"
    assert provider_module._ws_override("https://maas.qianwenaiapi.com/api/v1") is None


def test_defaults_are_the_maas_gateway_and_the_message_model():
    assert provider_module.DEFAULT_BASE_URL == "https://maas.qianwenaiapi.com/api/v1"
    assert provider_module.DEFAULT_MODEL == DEFAULT_MODEL


def test_config_wins_then_env_then_default(monkeypatch):
    monkeypatch.setattr(provider_module, "_section", lambda *a, **k: {"base_url": "https://cfg.test/api/v1"})
    monkeypatch.setattr(provider_module, "_env", lambda *a, **k: None)
    assert provider_module._configured_base_url() == "https://cfg.test/api/v1"

    monkeypatch.setattr(provider_module, "_section", lambda *a, **k: {})
    monkeypatch.setattr(
        provider_module, "_env",
        lambda name, *a, **k: "https://own.test/api/v1" if name == "TRANSCRIPTION_DASHSCOPE_BASE_URL" else None,
    )
    assert provider_module._configured_base_url() == "https://own.test/api/v1"


def test_plugin_namespace_wins_over_the_custom_and_vendor_variables(monkeypatch):
    monkeypatch.setattr(provider_module, "_section", lambda *a, **k: {})
    monkeypatch.setattr(provider_module, "_env", lambda name, *a, **k: f"https://{name}.test/api/v1")
    assert provider_module._configured_base_url() == "https://TRANSCRIPTION_DASHSCOPE_BASE_URL.test/api/v1"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("HERMES_CUSTOM_DASHSCOPE_BASE_URL", "https://dashscope.test/compatible-mode/v1"),
        ("DASHSCOPE_BASE_URL", "https://dashscope.test/compatible-mode/v1"),
    ],
)
def test_fallback_variables_are_honoured(monkeypatch, name, expected):
    monkeypatch.setattr(provider_module, "_section", lambda *a, **k: {})
    monkeypatch.setattr(provider_module, "_env", lambda key, *a, **k: expected if key == name else None)
    assert provider_module._configured_base_url() == expected


def test_the_default_gateway_is_used_when_nothing_is_set(monkeypatch):
    monkeypatch.setattr(provider_module, "_section", lambda *a, **k: {})
    monkeypatch.setattr(provider_module, "_env", lambda *a, **k: None)
    assert provider_module._configured_base_url() == provider_module.DEFAULT_BASE_URL


# ── parameter assembly ──

def test_parameters_map_language_and_prompt_onto_the_wire():
    file_params = provider_module._parameters("zh-CN", "百炼云, 通义千问", live=False)
    assert file_params["language_hints"] == ["zh"]
    assert file_params["vocabulary"] == {"百炼云": 5, "通义千问": 5}
    assert file_params["heartbeat"] is True
    assert "intermediate_result_enabled" not in file_params

    live_params = provider_module._parameters(None, None, live=True)
    assert live_params["intermediate_result_enabled"] is True
    assert "language_hints" not in live_params
    assert "vocabulary" not in live_params


def test_config_keys_use_the_wire_names():
    section = {
        "max_sentence_silence": 500,
        "vad_model": "near_meeting_16k",
        "speech_noise_threshold": 0.2,
        "keep_dialect": True,
        "heartbeat": False,  # a default we ship, still overridable
        "language_hints": ["zh", "en"],
        "parameters": {"multi_threshold_mode_enabled": True},
        # reserved: ours, or fixed by the provider
        "base_url": "https://x.test/api/v1",
        "language": "zh",
        "format": "mp3",
        "sample_rate": 8000,
        "disfluency_removal_enabled": True,
        "vocabulary_id": "vocab-123",
    }
    params = provider_module._parameters("zh", None, live=True, section=section)

    assert params["max_sentence_silence"] == 500
    assert params["vad_model"] == "near_meeting_16k"
    assert params["speech_noise_threshold"] == 0.2
    assert params["keep_dialect"] is True
    assert params["heartbeat"] is False
    assert params["language_hints"] == ["zh", "en"]
    assert params["multi_threshold_mode_enabled"] is True
    for reserved in ("base_url", "language", "format", "sample_rate", "parameters",
                     "disfluency_removal_enabled", "vocabulary_id"):
        assert reserved not in params


def test_named_kwargs_go_to_the_sdk_not_the_wire():
    section = {
        "vocabulary_id": "vocab-123",
        "disfluency_removal_enabled": True,
        "workspace": "ws-1",
        "ws_url": "",
        "max_sentence_silence": 500,
    }
    kwargs = provider_module._api_kwargs(section, model="m", timeout=30, ws_url="wss://x.test")

    assert kwargs["model"] == "m"
    assert kwargs["timeout"] == 30
    assert kwargs["vocabulary_id"] == "vocab-123"
    assert kwargs["disfluency_removal_enabled"] is True
    assert kwargs["workspace"] == "ws-1"
    assert kwargs["ws_url"] == "wss://x.test"
    assert "max_sentence_silence" not in kwargs


# ── provider identity and failure envelopes ──

@requires_sdk
def test_identity_and_capabilities(configure, fake_server):
    with fake_server() as server:
        instance = configure(server)
        assert instance.name == "dashscope"
        assert instance.display_name == "DashScope Qwen-Audio-ASR"
        assert instance.is_available() is True
        assert instance.streaming_capable is True
        assert instance.default_model() == DEFAULT_MODEL


def test_missing_sdk_is_reported_not_raised(monkeypatch):
    instance = provider_module.DashScopeProvider()
    monkeypatch.setattr(provider_module, "_sdk", lambda: None)
    monkeypatch.setattr(provider_module, "_api_key", lambda *a, **k: "k")

    assert instance.is_available() is False
    result = instance.transcribe(__file__)

    assert result["success"] is False
    assert "not importable" in result["error"]


@requires_sdk
def test_missing_key_degrades_to_an_error_envelope(configure, fake_server):
    with fake_server() as server:
        instance = configure(server, key=None)
        assert instance.is_available() is False
        result = instance.transcribe(__file__)

    assert result["success"] is False
    assert "TRANSCRIPTION_DASHSCOPE_API_KEY" in result["error"]


@requires_sdk
def test_transcribe_rejects_a_missing_file(configure, fake_server):
    with fake_server() as server:
        result = configure(server).transcribe("/nonexistent/voice.ogg")

    assert result["success"] is False
    assert "missing or empty" in result["error"]


# ── file path ──

@requires_sdk
@requires_audio
def test_transcribe_drives_one_file_task(configure, fake_server, clip):
    audio = clip(0.4)
    with fake_server() as server:
        result = configure(server).transcribe(audio, model=DEFAULT_MODEL, language="zh")

        assert result["success"] is True, result
        assert result["transcript"] == "你好，世界。"
        assert result["provider"] == "dashscope"
        assert server.audio_bytes >= 12000  # 0.4 s of 16 kHz mono s16le
        assert server.actions[:1] == ["run-task"]
        assert server.finished is True
        payload = server.run_task["payload"]
        assert payload["model"] == DEFAULT_MODEL
        assert payload["parameters"]["format"] == "wav"
        assert payload["parameters"]["sample_rate"] == 16000
        assert payload["parameters"]["language_hints"] == ["zh"]


@requires_sdk
@requires_audio
def test_section_options_reach_the_wire(configure, fake_server, clip):
    audio = clip(0.2)
    with fake_server() as server:
        instance = configure(server, section={
            "max_sentence_silence": 500,
            "disfluency_removal_enabled": True,
            "format": "mp3",
        })
        result = instance.transcribe(audio)

        assert result["success"] is True, result
        parameters = server.run_task["payload"]["parameters"]
        assert parameters["max_sentence_silence"] == 500
        assert parameters["format"] == "wav"  # fixed by the provider, not by config


@requires_sdk
@requires_audio
def test_transcribe_reports_a_rejected_task(configure, fake_server, clip):
    audio = clip(0.2)
    with fake_server(fail="unsupported audio format") as server:
        result = configure(server).transcribe(audio)

    assert result["success"] is False
    assert "unsupported audio format" in result["error"]


# ── live path ──

@requires_sdk
def test_live_session_streams_partials_then_a_final_transcript(configure, fake_server):
    with fake_server(mid_stream_interim="你好") as server:
        session = configure(server).open_stream_session(language="zh-CN", prompt="百炼云")
        for _ in range(6):
            session.push_audio(b"\x00\x00" * 1600)  # 100 ms of silence per frame
        deadline = time.monotonic() + 5
        while not session.partial_transcript() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert session.partial_transcript() == "你好"

        result = session.finalize()

        assert result["success"] is True, result
        assert result["transcript"] == "你好，世界。"
        assert session.partial_transcript() == "你好，世界。"
        parameters = server.run_task["payload"]["parameters"]
        assert parameters["intermediate_result_enabled"] is True
        assert parameters["vocabulary"] == {"百炼云": 5}
        assert server.finished is True


@requires_sdk
def test_push_audio_never_blocks_the_caller(configure, fake_server):
    with fake_server(sentences=()) as server:
        session = configure(server).open_stream_session()
        payload = b"\x00\x00" * 1600
        started = time.monotonic()
        for _ in range(2000):  # ~3.2 MB of audio, far more than a socket could take inline
            session.push_audio(payload)
        elapsed = time.monotonic() - started
        assert elapsed < 1.0, f"push_audio blocked for {elapsed:.2f}s"
        session.cancel()


@requires_sdk
def test_live_session_without_speech_succeeds_with_empty_text(configure, fake_server):
    with fake_server(sentences=()) as server:
        result = configure(server).open_stream_session().finalize()

    assert result["success"] is True, result
    assert result["transcript"] == ""


@requires_sdk
def test_live_session_requires_a_key(configure, fake_server):
    with fake_server() as server:
        instance = configure(server, key=None)
        with pytest.raises(RuntimeError):
            instance.open_stream_session()
