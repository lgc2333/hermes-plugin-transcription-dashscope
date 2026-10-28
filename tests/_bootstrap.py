"""Shared test bootstrap.

The plugin root goes on ``sys.path`` so the ``transcription_dashscope`` subpackage imports exactly
the way Hermes imports it, and Hermes' own tree is added for the provider ABCs. The plugin's
declared dependency (``dashscope``) is prepared by Hermes' PM layer into the selected environment,
so when the interpreter running the tests does not have it, that environment's site-packages is
added too.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
HERMES_INSTALL = Path(os.environ.get("HERMES_INSTALL_DIR", "/opt/hermes"))

for _entry in (str(PLUGIN_ROOT), str(HERMES_INSTALL)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


def _ensure_plugin_dependency() -> None:
    """Put the PM-managed environment on ``sys.path`` when this interpreter lacks the dependency."""
    try:
        import dashscope  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    try:
        from pm.environments import selected_venv, site_packages

        for entry in (site_packages(selected_venv(HERMES_INSTALL)),):
            if str(entry) not in sys.path:
                sys.path.append(str(entry))
    except Exception:  # health: allow BLE001 -- tests then fail with a clear ImportError
        pass


_ensure_plugin_dependency()

provider = importlib.import_module("transcription_dashscope.provider")
