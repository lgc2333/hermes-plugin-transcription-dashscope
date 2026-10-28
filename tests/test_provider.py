"""Provider behaviour against a fake DashScope socket — no network, no credentials, no flakes."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import _bootstrap  # noqa: F401  (sys.path + PM environment for the declared dependency)
from _fake_server import FakeFunAsrServer
from transcription_dashscope import provider as provider_module

FFMPEG = shutil.which("ffmpeg")
HAS_SDK = provider_module._sdk() is not None
DEFAULT_MODEL = "qwen-audio-3.1-asr-flash-message"


def _wav(seconds: float = 0.4) -> str:
    """A real 16 kHz mono wav, produced the way a user's clip arrives."""
    handle, path = tempfile.mkstemp(suffix=".wav")
    os.close(handle)
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-ar", "16000", "-ac", "1", path],
        check=True, capture_output=True, timeout=60,
    )
    return path


class ConfigurationTests(unittest.TestCase):
    def test_socket_urls_are_passed_through_http_bases_are_not(self):
        self.assertEqual(provider_module._ws_override("wss://a.test/api-ws/v1/inference"),
                         "wss://a.test/api-ws/v1/inference")
        self.assertIsNone(provider_module._ws_override("https://maas.qianwenaiapi.com/api/v1"))

    def test_config_wins_then_env_then_default(self):
        with mock.patch.object(provider_module, "_section", return_value={"base_url": "https://cfg.test/api/v1"}):
            self.assertEqual(provider_module._configured_base_url(), "https://cfg.test/api/v1")
        with mock.patch.object(provider_module, "_section", return_value={}), \
                mock.patch.object(provider_module, "_env",
                                  side_effect=lambda name: "https://own.test/api/v1"
                                  if name == "TRANSCRIPTION_DASHSCOPE_BASE_URL" else None):
            self.assertEqual(provider_module._configured_base_url(), "https://own.test/api/v1")
        with mock.patch.object(provider_module, "_section", return_value={}), \
                mock.patch.object(provider_module, "_env",
                                  side_effect=lambda name: f"https://{name}.test/api/v1"):
            # Plugin namespace wins over the built-in pair, which wins over the vendor variable.
            self.assertEqual(provider_module._configured_base_url(),
                             "https://TRANSCRIPTION_DASHSCOPE_BASE_URL.test/api/v1")
        # The DashScope settings screen value is honoured too, after the STT-specific variable.
        with mock.patch.object(provider_module, "_section", return_value={}), \
                mock.patch.object(provider_module, "_env",
                                  side_effect=lambda name: "https://dashscope.test/compatible-mode/v1"
                                  if name == "HERMES_CUSTOM_DASHSCOPE_BASE_URL" else None):
            self.assertEqual(provider_module._configured_base_url(),
                             "https://dashscope.test/compatible-mode/v1")
        with mock.patch.object(provider_module, "_section", return_value={}), \
                mock.patch.object(provider_module, "_env", return_value=None):
            self.assertEqual(provider_module._configured_base_url(), provider_module.DEFAULT_BASE_URL)

    def test_defaults_are_the_maas_gateway_and_the_message_model(self):
        self.assertEqual(provider_module.DEFAULT_BASE_URL, "https://maas.qianwenaiapi.com/api/v1")
        self.assertEqual(provider_module.DEFAULT_MODEL, DEFAULT_MODEL)

    def test_parameters_map_language_and_prompt_onto_the_wire(self):
        file_params = provider_module._parameters("zh-CN", "百炼云, 通义千问", live=False)
        self.assertEqual(file_params["language_hints"], ["zh"])
        self.assertEqual(file_params["vocabulary"], {"百炼云": 5, "通义千问": 5})
        self.assertIs(file_params["heartbeat"], True)
        self.assertNotIn("intermediate_result_enabled", file_params)

        live_params = provider_module._parameters(None, None, live=True)
        self.assertIs(live_params["intermediate_result_enabled"], True)
        self.assertNotIn("language_hints", live_params)
        self.assertNotIn("vocabulary", live_params)

    def test_config_keys_use_the_wire_names(self):
        section = {
            "max_sentence_silence": 500,
            "vad_model": "near_meeting_16k",
            "speech_noise_threshold": 0.2,
            "keep_dialect": True,
            "heartbeat": False,          # a default we ship, still overridable
            "language_hints": ["zh", "en"],
            "parameters": {"multi_threshold_mode_enabled": True},
            # reserved: ours or fixed by the provider
            "base_url": "https://x.test/api/v1",
            "language": "zh",
            "format": "mp3",
            "sample_rate": 8000,
            "disfluency_removal_enabled": True,
            "vocabulary_id": "vocab-123",
        }
        params = provider_module._parameters("zh", None, live=True, section=section)
        self.assertEqual(params["max_sentence_silence"], 500)
        self.assertEqual(params["vad_model"], "near_meeting_16k")
        self.assertEqual(params["speech_noise_threshold"], 0.2)
        self.assertIs(params["keep_dialect"], True)
        self.assertIs(params["heartbeat"], False)
        self.assertEqual(params["language_hints"], ["zh", "en"])
        self.assertIs(params["multi_threshold_mode_enabled"], True)
        for reserved in ("base_url", "language", "format", "sample_rate", "parameters",
                         "disfluency_removal_enabled", "vocabulary_id"):
            self.assertNotIn(reserved, params)

    def test_named_kwargs_go_to_the_sdk_not_the_wire(self):
        section = {
            "vocabulary_id": "vocab-123",
            "disfluency_removal_enabled": True,
            "workspace": "ws-1",
            "ws_url": "",
            "max_sentence_silence": 500,
        }
        kwargs = provider_module._api_kwargs(section, model="m", timeout=30, ws_url="wss://x.test")
        self.assertEqual(kwargs["model"], "m")
        self.assertEqual(kwargs["timeout"], 30)
        self.assertEqual(kwargs["vocabulary_id"], "vocab-123")
        self.assertIs(kwargs["disfluency_removal_enabled"], True)
        self.assertEqual(kwargs["workspace"], "ws-1")
        self.assertEqual(kwargs["ws_url"], "wss://x.test")
        self.assertNotIn("max_sentence_silence", kwargs)


@unittest.skipIf(FFMPEG is None, "ffmpeg is required to decode audio")
@unittest.skipIf(not HAS_SDK, "the plugin's declared dependency (dashscope >=2) is not prepared")
class ProviderTests(unittest.TestCase):
    def setUp(self):
        self._tmp: list[str] = []
        self._stack = ExitStack()

    def tearDown(self):
        self._stack.close()
        for path in self._tmp:
            Path(path).unlink(missing_ok=True)

    def _configure(self, server: FakeFunAsrServer, key: str | None = "test-key", *,
                   base_url: str | None = None):
        self._stack.enter_context(mock.patch.object(provider_module, "_api_key", return_value=key))
        if base_url is None:
            self._stack.enter_context(mock.patch.object(
                provider_module, "_section", return_value={"base_url": server.url, "timeout": 30}))
        return provider_module.DashScopeProvider()

    # ── identity ──
    def test_identity_and_capabilities(self):
        with FakeFunAsrServer() as server:
            instance = self._configure(server)
            self.assertEqual(instance.name, "dashscope")
            self.assertEqual(instance.display_name, "DashScope Qwen-Audio-ASR")
            self.assertTrue(instance.is_available())
            self.assertTrue(instance.streaming_capable)
            self.assertEqual(instance.default_model(), DEFAULT_MODEL)

    def test_missing_sdk_is_reported_not_raised(self):
        instance = provider_module.DashScopeProvider()
        with mock.patch.object(provider_module, "_sdk", return_value=None), \
                mock.patch.object(provider_module, "_api_key", return_value="k"):
            self.assertFalse(instance.is_available())
            result = instance.transcribe(__file__)
        self.assertFalse(result["success"])
        self.assertIn("not importable", result["error"])

    def test_missing_key_degrades_to_an_error_envelope(self):
        with FakeFunAsrServer() as server:
            instance = self._configure(server, key=None)
            self.assertFalse(instance.is_available())
            result = instance.transcribe(__file__)
        self.assertFalse(result["success"])
        self.assertIn("HERMES_CUSTOM_DASHSCOPE_API_KEY", result["error"])

    def test_transcribe_rejects_a_missing_file(self):
        with FakeFunAsrServer() as server:
            instance = self._configure(server)
            result = instance.transcribe("/nonexistent/voice.ogg")
        self.assertFalse(result["success"])
        self.assertIn("missing or empty", result["error"])

    # ── file path ──
    def test_transcribe_drives_one_file_task(self):
        audio = _wav(0.4)
        self._tmp.append(audio)
        with FakeFunAsrServer() as server:
            instance = self._configure(server)
            result = instance.transcribe(audio, model=DEFAULT_MODEL, language="zh")
            self.assertTrue(result["success"], result)
            self.assertEqual(result["transcript"], "你好，世界。")
            self.assertEqual(result["provider"], "dashscope")
            self.assertGreaterEqual(server.audio_bytes, 12000)  # 0.4 s of 16 kHz mono s16le
            self.assertEqual(server.actions[:1], ["run-task"])
            self.assertTrue(server.finished)
            parameters = server.run_task["payload"]["parameters"]
            self.assertEqual(server.run_task["payload"]["model"], DEFAULT_MODEL)
            self.assertEqual(parameters["format"], "wav")
            self.assertEqual(parameters["sample_rate"], 16000)
            self.assertEqual(parameters["language_hints"], ["zh"])

    def test_section_options_reach_the_wire(self):
        audio = _wav(0.2)
        self._tmp.append(audio)
        with FakeFunAsrServer() as server:
            self._stack.enter_context(mock.patch.object(provider_module, "_api_key",
                                                        return_value="test-key"))
            self._stack.enter_context(mock.patch.object(provider_module, "_section", return_value={
                "base_url": server.url,
                "timeout": 30,
                "max_sentence_silence": 500,
                "disfluency_removal_enabled": True,
                "format": "mp3",
            }))
            instance = provider_module.DashScopeProvider()
            result = instance.transcribe(audio)
            self.assertTrue(result["success"], result)
            parameters = server.run_task["payload"]["parameters"]
            self.assertEqual(parameters["max_sentence_silence"], 500)
            self.assertEqual(parameters["format"], "wav")  # fixed by the provider, not by config

    def test_transcribe_reports_a_rejected_task(self):
        audio = _wav(0.2)
        self._tmp.append(audio)
        with FakeFunAsrServer(fail="unsupported audio format") as server:
            instance = self._configure(server)
            result = instance.transcribe(audio)
        self.assertFalse(result["success"])
        self.assertIn("unsupported audio format", result["error"])

    # ── live path ──
    def test_live_session_streams_partials_then_a_final_transcript(self):
        with FakeFunAsrServer(mid_stream_interim="你好") as server:
            instance = self._configure(server)
            session = instance.open_stream_session(language="zh-CN", prompt="百炼云")
            for _ in range(6):
                session.push_audio(b"\x00\x00" * 1600)  # 100 ms of silence per frame
            deadline = time.monotonic() + 5
            while not session.partial_transcript() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(session.partial_transcript(), "你好")
            result = session.finalize()
            self.assertTrue(result["success"], result)
            self.assertEqual(result["transcript"], "你好，世界。")
            self.assertEqual(session.partial_transcript(), "你好，世界。")
        parameters = server.run_task["payload"]["parameters"]
        self.assertIs(parameters["intermediate_result_enabled"], True)
        self.assertEqual(parameters["vocabulary"], {"百炼云": 5})
        self.assertTrue(server.finished)

    def test_push_audio_never_blocks_the_caller(self):
        with FakeFunAsrServer(sentences=()) as server:
            instance = self._configure(server)
            session = instance.open_stream_session()
            payload = b"\x00\x00" * 1600
            started = time.monotonic()
            for _ in range(2000):  # ~3.2 MB of audio, far more than a socket could take inline
                session.push_audio(payload)
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 1.0, f"push_audio blocked for {elapsed:.2f}s")
            session.cancel()

    def test_live_session_without_speech_succeeds_with_empty_text(self):
        with FakeFunAsrServer(sentences=()) as server:
            instance = self._configure(server)
            session = instance.open_stream_session()
            result = session.finalize()
        self.assertTrue(result["success"], result)
        self.assertEqual(result["transcript"], "")

    def test_live_session_requires_a_key(self):
        with FakeFunAsrServer() as server:
            instance = self._configure(server, key=None)
            with self.assertRaises(RuntimeError):
                instance.open_stream_session()


if __name__ == "__main__":
    unittest.main()
