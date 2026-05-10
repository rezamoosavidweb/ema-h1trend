"""Compatibility shim — implementation lives in ``strategies.ema_trend.crypto_core``."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from strategies.ema_trend.crypto_core import *  # noqa: F403
