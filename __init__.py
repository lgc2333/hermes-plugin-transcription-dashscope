"""Hermes STT plugin entry point: DashScope Fun-ASR-Realtime.

Hermes calls ``register(ctx)`` on the plugin root. All implementation lives in the
``transcription_dashscope`` subpackage so this root stays free of top-level modules that could
shadow Hermes' own imports when the plugin directory lands on ``sys.path``.
"""

if __package__:
    from .transcription_dashscope import register
else:  # imported as a bare top-level module: test collection walks up from tests/ and imports
    # this file without the plugin directory on sys.path, so add it here.
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from transcription_dashscope import register  # noqa: E402

__all__ = ["register"]
