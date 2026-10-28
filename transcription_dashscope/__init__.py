"""DashScope Fun-ASR-Realtime STT provider for Hermes (file transcription + live dictation).

Selected with ``stt.provider: dashscope``; live partials need ``stt.streaming: true``.
"""

from .provider import DashScopeProvider

__all__ = ["DashScopeProvider", "register"]


def register(ctx) -> None:
    ctx.register_transcription_provider(DashScopeProvider())
