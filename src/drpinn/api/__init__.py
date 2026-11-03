"""Milestones 9-10: analysis service, FastAPI backend and the dashboard it serves.

`service` and `disc` need only the core stack; `app` (FastAPI) and `colab` are
imported on demand so the rest of the package works without them.
"""

from .disc import DISC_TO_FOV, estimate_disc
from .service import Analyser, ServiceConfig

__all__ = ["DISC_TO_FOV", "Analyser", "ServiceConfig", "estimate_disc"]
