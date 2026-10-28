"""Hermes STT provider built on the DashScope SDK (Qwen-Audio-ASR-Flash).

Registered through ``ctx.register_transcription_provider`` and selected by ``stt.provider:
dashscope``. One model serves both Hermes dispatch paths, and both go through the SDK's client
API, where the key and the endpoint belong to the client — no module globals:

* ``transcribe()``          — voice messages and desktop fallbacks. ffmpeg normalizes the clip to
                              16 kHz mono wav, then one ``client.audio.asr.recognize_file()`` call
                              drives the task to completion and yields the events.
* ``open_stream_session()`` — live dictation (``stt.streaming: true``). ``client.audio.asr.
                              recognize()`` opens a duplex session; ``intermediate_result_enabled``
                              makes the server emit partial text.

Threading contract (Hermes calls these from workers, and the desktop pushes mic frames straight
from its event loop, so nothing here may block the caller's thread):

* ``transcribe()`` and ``open_stream_session()`` are sync and blocking by design — Hermes invokes
  them through ``asyncio.to_thread`` / ``run_in_executor``.
* ``push_audio()`` never blocks: it appends to an internal queue and a feeder thread owns the SDK
  socket write, so an asyncio loop pushing frames is never stalled by network I/O.
* ``finalize()`` blocks (the host awaits it in an executor) until the task finishes or times out.

The SDK is declared in ``pyproject.toml`` (``dependencies``), so Hermes' PM layer prepares it when
the plugin is enabled. The import stays lazy: the plugin must load (and report an actionable error)
on an interpreter where that preparation has not happened yet.

Configuration (all optional; the endpoint also reads an environment variable):

    stt:
      streaming: true
      dashscope:
        model: qwen-audio-3.1-asr-flash-message
        base_url: https://maas.qianwenaiapi.com/api/v1
        timeout: 300

``base_url`` accepts the HTTP base and the SDK derives the task socket; a ``ws://``/``wss://`` URL
is passed through as the socket URL instead. Precedence: ``stt.dashscope.base_url`` →
``DASHSCOPE_STT_BASE_URL`` → ``HERMES_CUSTOM_DASHSCOPE_STT_BASE_URL`` → the default above.

The API key comes from the environment only — ``HERMES_CUSTOM_DASHSCOPE_API_KEY``, falling back to
``DASHSCOPE_API_KEY`` — so no secret ever lands in ``config.yaml``.
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from agent.transcription_provider import TranscriptionProvider, TranscriptionStreamSession

logger = logging.getLogger(__name__)

PROVIDER_NAME = "dashscope"
# Resolution order for both credentials and endpoint: this plugin's own namespace first, then the
# HERMES_CUSTOM_ pair (a user-defined custom provider entry, like the one this install uses), then
# the DASHSCOPE_ variables of the built-in provider. Hermes never reads TRANSCRIPTION_DASHSCOPE_*,
# so those names belong to us alone.
API_KEY_ENVS = ("TRANSCRIPTION_DASHSCOPE_API_KEY", "HERMES_CUSTOM_DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY")
BASE_URL_ENVS = ("TRANSCRIPTION_DASHSCOPE_BASE_URL", "HERMES_CUSTOM_DASHSCOPE_BASE_URL", "DASHSCOPE_BASE_URL")
# Workspace has no shared equivalent: the built-in provider does not use one.
WORKSPACE_ENVS = ("TRANSCRIPTION_DASHSCOPE_WORKSPACE",)
# The vendor's "message" model: built for voice messages and dictation, and the one that both
# transcribes whole clips and streams intermediate results (`intermediate_result_enabled`).
DEFAULT_MODEL = "qwen-audio-3.1-asr-flash-message"
DEFAULT_BASE_URL = "https://maas.qianwenaiapi.com/api/v1"
SAMPLE_RATE = 16000
DEFAULT_TIMEOUT_S = 300.0
FINALIZE_TIMEOUT_S = 20.0
# 100 ms of 16 kHz mono s16le — a frame size the server accepts and the mic naturally produces.
PCM_CHUNK_BYTES = 3200
# Instant hotwords: the documented vehicle for a word list, and the honest mapping of Hermes'
# `prompt` (documented as a vocabulary hint). Weight 5 is the strongest normal weight.
HOTWORD_WEIGHT = 5
MAX_HOTWORDS = 50


def _ok(transcript: str) -> Dict[str, Any]:
    return {"success": True, "transcript": transcript, "provider": PROVIDER_NAME}


def _error(message: str) -> Dict[str, Any]:
    return {"success": False, "transcript": "", "provider": PROVIDER_NAME, "error": message}


def _sdk():
    """The DashScope SDK module, or None when PM has not prepared the declared dependency."""
    try:
        import dashscope
    except Exception:  # health: allow BLE001 -- absence is an expected, reportable state
        logger.debug("DashScope SDK is not importable in this environment", exc_info=True)
        return None
    if not hasattr(dashscope, "Client"):
        # 1.x has no per-client credentials/endpoint surface; this plugin requires 2.x.
        logger.warning("DashScope SDK %s is too old for this plugin (need >=2,<3)",
                       getattr(dashscope, "__version__", "?"))
        return None
    return dashscope


def _env(name: str) -> Optional[str]:
    try:
        from hermes_cli.config import get_env_value
    except Exception:  # health: allow BLE001 -- fall back to the raw environment
        import os

        return os.environ.get(name)
    return get_env_value(name)


def _api_key() -> Optional[str]:
    for name in API_KEY_ENVS:
        value = _env(name)
        if value and value.strip():
            return value.strip()
    return None


def _section() -> Dict[str, Any]:
    """``stt.dashscope``, falling back to ``stt.providers.dashscope`` when that is not a command."""
    try:
        from hermes_cli.config import load_config_readonly
        stt = (load_config_readonly() or {}).get("stt") or {}
    except Exception:  # health: allow BLE001 -- config read failures fall back to defaults
        logger.debug("DashScope STT config read failed", exc_info=True)
        return {}
    section = stt.get(PROVIDER_NAME)
    if isinstance(section, dict):
        return section
    providers = stt.get("providers")
    if isinstance(providers, dict):
        candidate = providers.get(PROVIDER_NAME)
        if isinstance(candidate, dict) and str(candidate.get("type") or "").lower() != "command":
            return candidate
    return {}


def _configured_base_url(section: Optional[Dict[str, Any]] = None) -> str:
    """``stt.dashscope.base_url`` → ``DASHSCOPE_STT_BASE_URL`` → the default HTTP base."""
    section = _section() if section is None else section
    configured = str(section.get("base_url") or "").strip()
    if configured:
        return configured
    for name in BASE_URL_ENVS:
        value = (_env(name) or "").strip()
        if value:
            return value
    return DEFAULT_BASE_URL


def _configured_workspace(section: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """``stt.dashscope.workspace`` → ``DASHSCOPE_WORKSPACE`` → none (SDK's legacy fallback)."""
    section = _section() if section is None else section
    configured = str(section.get("workspace") or "").strip()
    if configured:
        return configured
    for name in WORKSPACE_ENVS:
        value = (_env(name) or "").strip()
        if value:
            return value
    return None


def _ws_override(base_url: str) -> Optional[str]:
    """A raw socket URL is handed to the API as ``ws_url``; an HTTP base lets the SDK derive it."""
    url = (base_url or "").strip()
    return url if url.startswith(("ws://", "wss://")) else None


def _config_number(section: Dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(section.get(key))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _primary_subtag(language: Optional[str]) -> Optional[str]:
    """``zh-CN``/``zh_CN`` -> ``zh``; the API takes bare language codes."""
    if not language:
        return None
    code = str(language).replace("_", "-").split("-")[0].strip().lower()
    return code or None


def _join_sentences(sentences: List[str]) -> str:
    """Join finalized sentences without gluing English words together."""
    out = ""
    for text in sentences:
        text = text.strip()
        if not text:
            continue
        if out and out[-1].isascii() and not out[-1].isspace():
            out += " "
        out += text
    return out.strip()


def _hotwords(prompt: Optional[str]) -> Dict[str, int]:
    """Hermes' ``prompt`` is a vocabulary hint; the wire calls that ``vocabulary`` (即时热词)."""
    words = [w for w in str(prompt or "").replace(",", " ").replace("，", " ").split() if w]
    return {word: HOTWORD_WEIGHT for word in list(dict.fromkeys(words))[:MAX_HOTWORDS]}


# `recognize()` / `recognize_file()` take these as named kwargs; every other key under
# `stt.dashscope` is a wire parameter and is forwarded verbatim inside `parameters`.
NAMED_API_KWARGS = frozenset({
    "model", "vocabulary_id", "disfluency_removal_enabled", "diarization_enabled",
    "phrase_id", "workspace", "ws_url", "timeout",
})
# Keys this plugin owns or deliberately fixes; they never reach the wire.
RESERVED_CONFIG_KEYS = frozenset({
    "base_url", "parameters", "language", "api_key", "format", "sample_rate",
})


def _parameters(language: Optional[str], prompt: Optional[str], *, live: bool,
                section: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Our defaults (language, prompt, partials, heartbeat) overlaid by the section's own keys."""
    section = section or {}
    parameters: Dict[str, Any] = {"heartbeat": True}
    hint = _primary_subtag(language)
    if hint:
        parameters["language_hints"] = [hint]
    if hotwords := _hotwords(prompt):
        parameters["vocabulary"] = hotwords
    if live:
        # Whole-clip calls do not need partial text; dictation cannot work without it.
        parameters["intermediate_result_enabled"] = True
    parameters.update({key: value for key, value in section.items()
                       if key not in NAMED_API_KWARGS and key not in RESERVED_CONFIG_KEYS})
    if isinstance(section.get("parameters"), dict):
        parameters.update(section["parameters"])
    return parameters


def _api_kwargs(section: Dict[str, Any], *, model: str, timeout: float,
                ws_url: Optional[str]) -> Dict[str, Any]:
    """Section keys the SDK takes as named kwargs, plus the resolved model/timeout/socket."""
    kwargs: Dict[str, Any] = {"model": model, "timeout": timeout}
    for key in NAMED_API_KWARGS:
        if key in ("model", "timeout") or key not in section:
            continue
        if section[key] is not None and section[key] != "":
            kwargs[key] = section[key]
    if ws_url:
        kwargs["ws_url"] = ws_url
    return kwargs


class _Collector:
    """Sentence assembly for one task, fed by the SDK's event stream."""

    def __init__(self) -> None:
        self.sentences: List[str] = []
        self.interim = ""
        self.failure: Optional[str] = None

    def absorb(self, event: Dict[str, Any]) -> None:
        payload = event.get("payload") or {}
        header = event.get("header") or {}
        if str(header.get("event") or "") == "task-failed":
            self.failure = f"{header.get('error_code') or 'TASK_FAILED'}: " \
                           f"{header.get('error_message') or 'recognition task failed'}"
            return
        sentence = (payload.get("output") or {}).get("sentence")
        for item in sentence if isinstance(sentence, list) else [sentence]:
            if not isinstance(item, dict) or item.get("heartbeat"):
                continue
            text = str(item.get("text") or "").strip()
            if item.get("sentence_end"):
                self.interim = ""  # a final result supersedes this sentence's interim text
                if text:
                    self.sentences.append(text)
            else:
                self.interim = text

    def absorb_error(self, exc: BaseException) -> None:
        self.failure = self.failure or str(exc)

    @property
    def text(self) -> str:
        return _join_sentences([*self.sentences, self.interim])

    def result(self, *, sent_audio: bool) -> Dict[str, Any]:
        if self.failure:
            return _error(self.failure)
        if not sent_audio:
            return _ok("")
        return _ok(self.text)


class _Client:
    """A ``dashscope.Client`` built per call, so key and endpoint stay call-scoped."""

    def __init__(self, api_key: str, base_url: str, timeout: float,
                 workspace: Optional[str] = None) -> None:
        sdk = _sdk()
        if sdk is None:
            raise RuntimeError(
                "DashScope SDK is not importable in this process — install it with `hermes plugins "
                "enable transcription-dashscope` and restart the gateway")
        socket_url = _ws_override(base_url)
        # Client takes an HTTP base; a raw ws URL is handed to each call as `ws_url` instead.
        http_base = (base_url or "").strip()
        if not http_base.startswith(("http://", "https://")):
            http_base = DEFAULT_BASE_URL
        kwargs: Dict[str, Any] = {"api_key": api_key, "base_url": http_base, "timeout": timeout}
        if workspace:
            kwargs["workspace"] = workspace
        self.client = sdk.Client(**kwargs)
        self.ws_url = socket_url


class DashScopeStreamSession(TranscriptionStreamSession):
    """Live dictation: mic frames are queued, a feeder thread owns every socket write."""

    def __init__(self, recognizer: Any, collector: _Collector,
                 finalize_timeout: float = FINALIZE_TIMEOUT_S) -> None:
        self._rec = recognizer
        self._collector = collector
        self._finalize_timeout = finalize_timeout
        self._audio: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._sent = threading.Event()      # the reader saw task-finished
        self._closed = threading.Event()    # both workers are done
        self._cancelled = threading.Event()
        self._done = threading.Event()
        self._result: Dict[str, Any] = _error("live transcription did not finish")
        self._audio_bytes = 0
        self._lock = threading.Lock()
        threading.Thread(target=self._read_events, name="stt-dashscope-events", daemon=True).start()
        threading.Thread(target=self._feed, name="stt-dashscope-feed", daemon=True).start()

    # ── caller API (never blocks) ──
    def push_audio(self, chunk: bytes) -> None:
        if chunk and not self._cancelled.is_set():
            self._audio.put(bytes(chunk))  # unbounded: a put here cannot block the caller

    # ── caller API (blocking, used from an executor) ──
    def finalize(self) -> Dict[str, Any]:
        self._audio.put(None)
        self._sent.wait(self._finalize_timeout)
        self._closed.wait(self._finalize_timeout)
        with self._lock:
            if not self._done.is_set():
                self._finish(self._collector.result(sent_audio=self._audio_bytes > 0))
            return dict(self._result)

    def partial_transcript(self) -> str:
        return self._collector.text

    def cancel(self) -> None:
        self._cancelled.set()
        self._audio.put(None)
        self._finish(_error("cancelled"))

    # ── workers ──
    def _finish(self, result: Dict[str, Any]) -> None:
        if not self._done.is_set():
            self._result = result
            self._done.set()

    def _feed(self) -> None:
        try:
            while True:
                chunk = self._audio.get()
                if chunk is None or self._cancelled.is_set():
                    break
                self._rec.send_audio_frame(chunk)
                self._audio_bytes += len(chunk)
            if not self._cancelled.is_set():
                self._rec.stop()  # end of input: keep reading until task-finished
        except Exception as exc:  # health: allow BLE001 -- worker boundary: report, never raise
            logger.warning("DashScope feed failed: %s", exc, exc_info=True)
            self._collector.absorb_error(exc)
            self._sent.set()
            self._safe_close()

    def _read_events(self) -> None:
        try:
            for event in self._rec.events():
                self._collector.absorb(event)
        except Exception as exc:  # health: allow BLE001 -- worker boundary: report, never raise
            logger.warning("DashScope event stream failed: %s", exc, exc_info=True)
            self._collector.absorb_error(exc)
        finally:
            self._sent.set()
            if self._audio_bytes or self._cancelled.is_set():
                self._safe_close()
            self._closed.set()

    def _safe_close(self) -> None:
        try:
            self._rec.close()
        except Exception:  # health: allow BLE001 -- already closed / socket gone
            logger.debug("DashScope session close failed", exc_info=True)


class DashScopeProvider(TranscriptionProvider):
    """DashScope Qwen-Audio-ASR (file + realtime), through the official SDK's client API."""

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def display_name(self) -> str:
        return "DashScope Qwen-Audio-ASR"

    def default_model(self) -> str:
        return str(_section().get("model") or DEFAULT_MODEL)

    def is_available(self) -> bool:
        return _sdk() is not None and _api_key() is not None

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "cn",
            "tag": "streaming",
            "env_vars": [{
                "key": API_KEY_ENVS[0],
                "prompt": "DashScope API key",
                "url": "https://bailian.console.aliyun.com/",
            }],
        }

    @property
    def streaming_capable(self) -> bool:
        return self.is_available()

    # ── file path (blocking; Hermes calls it in a worker) ──
    def transcribe(
        self, file_path: str, *, model: Optional[str] = None, language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        if _sdk() is None:
            return _error("DashScope SDK is not importable in this process — install it with `hermes "
                          "plugins enable transcription-dashscope` and restart the gateway")
        api_key = _api_key()
        if api_key is None:
            return _error(f"no DashScope API key: set {API_KEY_ENVS[0]} "
                          f"(or {' / '.join(API_KEY_ENVS[1:])})")
        audio = Path(file_path)
        if not audio.exists() or audio.stat().st_size == 0:
            return _error(f"audio file is missing or empty: {file_path}")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return _error("ffmpeg is required to decode audio for DashScope; install ffmpeg on PATH")

        section = _section()
        timeout = _config_number(section, "timeout", DEFAULT_TIMEOUT_S)
        collector = _Collector()
        wav = None
        try:
            handle, wav = tempfile.mkstemp(suffix=".wav", prefix="hermes-stt-")
            Path(wav).unlink()  # let ffmpeg create it with its own permissions
            import os

            os.close(handle)
            _to_wav(str(audio), wav, ffmpeg)
            wiring = _Client(api_key, _configured_base_url(section), timeout,
                             _configured_workspace(section))
            events: Iterator[Dict[str, Any]] = wiring.client.audio.asr.recognize_file(
                wav,
                format="wav",
                sample_rate=SAMPLE_RATE,
                parameters=_parameters(language, extra.get("prompt"), live=False, section=section),
                **_api_kwargs(section, model=str(model or section.get("model") or DEFAULT_MODEL),
                              timeout=timeout, ws_url=wiring.ws_url),
            )
            for event in events:
                collector.absorb(event)
        except Exception as exc:  # health: allow BLE001 -- must return the envelope, never raise
            logger.warning("DashScope STT failed for %s: %s", file_path, exc, exc_info=True)
            collector.absorb_error(exc)
            return _error(f"dashscope transcription failed: {exc}")
        finally:
            if wav:
                Path(wav).unlink(missing_ok=True)
        return collector.result(sent_audio=audio.stat().st_size > 0)

    # ── live path (blocking open; frames are queued afterwards) ──
    def open_stream_session(
        self, *, language: Optional[str] = None, prompt: Optional[str] = None,
    ) -> TranscriptionStreamSession:
        api_key = _api_key()
        if api_key is None:
            raise RuntimeError(f"no DashScope API key: set {API_KEY_ENVS[0]} "
                               f"(or {' / '.join(API_KEY_ENVS[1:])})")
        section = _section()
        wiring = _Client(api_key, _configured_base_url(section),
                         _config_number(section, "timeout", DEFAULT_TIMEOUT_S),
                         _configured_workspace(section))
        collector = _Collector()
        recognizer = wiring.client.audio.asr.recognize(
            format="pcm",
            sample_rate=SAMPLE_RATE,
            parameters=_parameters(language, prompt, live=True, section=section),
            **_api_kwargs(section, model=str(section.get("model") or DEFAULT_MODEL),
                          timeout=_config_number(section, "timeout", DEFAULT_TIMEOUT_S),
                          ws_url=wiring.ws_url),
        )
        recognizer.start()
        return DashScopeStreamSession(recognizer, collector)


def _to_wav(source: str, target: str, ffmpeg: str) -> None:
    """Normalize any ffmpeg-readable container to the 16 kHz mono PCM wav the API expects."""
    subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-y", "-i", source,
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", target],
        check=True, capture_output=True, timeout=300,
    )
