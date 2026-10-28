"""Pytest setup and shared fixtures.

Import paths first: the plugin root goes on ``sys.path`` so ``transcription_dashscope`` imports the
way Hermes imports it, ``tests/`` goes on so helper modules resolve under any import mode, and the
Hermes install directory goes on because the provider subclasses ``agent.transcription_provider``
(set ``HERMES_INSTALL_DIR`` when Hermes lives elsewhere; CI clones the upstream tag).

The plugin's declared dependency (``dashscope``) is prepared by Hermes' PM layer when the plugin is
enabled; when the interpreter running the tests does not have it, that environment is used instead.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
PLUGIN_ROOT = TESTS_DIR.parent

HERMES_INSTALL = Path(os.environ.get("HERMES_INSTALL_DIR", "/opt/hermes"))

for _entry in (str(PLUGIN_ROOT), str(TESTS_DIR), str(HERMES_INSTALL)):
    if _entry not in sys.path and Path(_entry).is_dir():
        sys.path.insert(0, _entry)

FFMPEG = shutil.which("ffmpeg")
if not (HERMES_INSTALL / "agent" / "transcription_provider.py").is_file():
    pytest.exit(
        f"Hermes sources not found at {HERMES_INSTALL}. The provider subclasses "
        "agent.transcription_provider, so the tests need them:\n"
        "    git clone --depth 1 https://github.com/NousResearch/hermes-agent /opt/hermes\n"
        "or point HERMES_INSTALL_DIR at an existing install.",
        returncode=2,
    )


def _ensure_plugin_dependency() -> None:
    """Fall back to the PM-managed environment when this interpreter lacks ``dashscope``."""
    try:
        import dashscope  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    if not (HERMES_INSTALL / "pm").is_dir():
        return
    try:
        from pm.environments import selected_venv, site_packages

        site = str(site_packages(selected_venv(HERMES_INSTALL)))
        if site not in sys.path:
            sys.path.append(site)
    except Exception:  # health: allow BLE001 -- the tests then skip with a clear reason
        pass


_ensure_plugin_dependency()

requires_audio = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is required to decode audio")


def _sdk_available() -> bool:
    from transcription_dashscope import provider

    return provider._sdk() is not None


requires_sdk = pytest.mark.skipif(
    not _sdk_available(), reason="the plugin's declared dependency (dashscope >=2) is not prepared"
)


@pytest.fixture
def clip(tmp_path):
    """Build a real 16 kHz mono wav, the way a user's voice clip arrives."""
    assert FFMPEG is not None  # guarded by requires_audio
    ffmpeg = FFMPEG

    def make(seconds: float = 0.4) -> str:
        path = tmp_path / f"clip-{seconds}.wav"
        subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-f", "lavfi",
             "-i", f"sine=frequency=440:duration={seconds}",
             "-ar", "16000", "-ac", "1", str(path)],
            check=True, capture_output=True, timeout=60,
        )
        return str(path)

    return make


@pytest.fixture
def fake_server():
    """The DashScope task-socket stand-in, imported lazily so collection never needs ``websockets``."""
    from _fake_server import FakeFunAsrServer

    return FakeFunAsrServer


@pytest.fixture
def configure(monkeypatch):
    """Point the provider at a fake server with a dummy key; ``section`` overrides on top."""
    from transcription_dashscope import provider

    def make(server, key: str | None = "test-key", section: dict | None = None):
        monkeypatch.setattr(provider, "_api_key", lambda *args, **kwargs: key)
        base = {"base_url": server.url, "timeout": 30}
        monkeypatch.setattr(provider, "_section", lambda *args, **kwargs: {**base, **(section or {})})
        return provider.DashScopeProvider()

    return make
