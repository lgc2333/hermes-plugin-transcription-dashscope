"""A stand-in for the DashScope task socket: enough of the protocol to drive the real SDK client."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from typing import Any

from websockets.sync.server import serve

DEFAULT_SENTENCES: Sequence[tuple[str, bool]] = (("你好", False), ("你好，世界。", True))


class FakeFunAsrServer:
    """Serves one recognition task per connection and records every frame the client sent."""

    def __init__(
        self,
        sentences: Sequence[tuple[str, bool]] = DEFAULT_SENTENCES,
        fail: str | None = None,
        heartbeat_first: bool = False,
        mid_stream_interim: str | None = None,
        on_run_task: Any | None = None,
    ) -> None:
        self.sentences = sentences
        self.fail = fail
        self.heartbeat_first = heartbeat_first
        self.mid_stream_interim = mid_stream_interim
        self.on_run_task = on_run_task
        self._interim_sent = False
        self.received: list[Any] = []
        self._server: Any = None

    def __enter__(self) -> FakeFunAsrServer:
        self._server = serve(self._handle, "127.0.0.1", 0)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._server.shutdown()

    @property
    def url(self) -> str:
        host, port = self._server.socket.getsockname()[:2]
        return f"ws://{host}:{port}/api-ws/v1/inference"

    # ── helpers for assertions ──
    @property
    def run_task(self) -> dict:
        return next(msg for msg in self.received if msg.get("header", {}).get("action") == "run-task")

    @property
    def actions(self) -> list[str]:
        return [msg.get("header", {}).get("action") for msg in self.received if isinstance(msg, dict)]

    @property
    def audio_bytes(self) -> int:
        return sum(item[1] for item in self.received if isinstance(item, tuple))

    @property
    def finished(self) -> bool:
        return "finish-task" in self.actions

    def _handle(self, ws: Any) -> None:
        first = json.loads(ws.recv())
        self.received.append(first)
        if self.on_run_task is not None:
            self.on_run_task(first)
        task_id = first["header"]["task_id"]
        if self.fail:
            ws.send(json.dumps({
                "header": {"task_id": task_id, "event": "task-failed",
                           "error_code": "CLIENT_ERROR", "error_message": self.fail},
                "payload": {},
            }))
            return
        ws.send(json.dumps({"header": {"task_id": task_id, "event": "task-started", "attributes": {}},
                            "payload": {}}))
        if self.heartbeat_first:
            ws.send(json.dumps({
                "header": {"task_id": task_id, "event": "result-generated", "attributes": {}},
                "payload": {"output": {"sentence": {"text": "", "heartbeat": True, "sentence_id": 0}},
                            "usage": None},
            }))
        for message in ws:
            if isinstance(message, (bytes, bytearray)):
                self.received.append(("audio", len(message)))
                if self.mid_stream_interim and not self._interim_sent:
                    self._interim_sent = True
                    self._emit(ws, task_id, self.mid_stream_interim, sentence_end=False)
                continue
            event = json.loads(message)
            self.received.append(event)
            if event.get("header", {}).get("action") == "finish-task":
                self._emit_results(ws, task_id)
                return

    def _emit_results(self, ws: Any, task_id: str) -> None:
        for text, sentence_end in self.sentences:
            self._emit(ws, task_id, text, sentence_end=sentence_end)
        ws.send(json.dumps({"header": {"task_id": task_id, "event": "task-finished", "attributes": {}},
                            "payload": {}}))
        # A real server keeps the task socket open until the client closes it; the SDK treats an
        # early close as `UnexpectedMessageReceived`, so drain instead of returning.
        for _message in ws:
            pass

    @staticmethod
    def _emit(ws: Any, task_id: str, text: str, *, sentence_end: bool) -> None:
        ws.send(json.dumps({
            "header": {"task_id": task_id, "event": "result-generated", "attributes": {}},
            "payload": {"output": {"sentence": {"text": text, "sentence_end": sentence_end,
                                                "sentence_id": 1}}, "usage": None},
        }))
