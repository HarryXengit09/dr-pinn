"""Run the API + dashboard inside a notebook (Colab or Jupyter) without blocking the kernel."""

from __future__ import annotations

import threading
import time
from pathlib import Path

_SERVERS: dict[int, object] = {}


def serve_in_background(project: str | Path, port: int = 8000, **config) -> object:
    """Start uvicorn in a daemon thread (stopping an earlier one on the same port); returns the server."""
    import uvicorn

    from .app import create_app
    from .service import ServiceConfig

    old = _SERVERS.pop(port, None)
    if old is not None:
        old.should_exit = True
        for _ in range(50):
            if not getattr(old, "started", False):
                break
            time.sleep(0.1)
    app = create_app(config=ServiceConfig(project=Path(project), **config))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    if not server.started:
        raise RuntimeError(f"the server did not start on port {port}")
    _SERVERS[port] = server
    return server
