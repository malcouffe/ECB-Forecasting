#!/usr/bin/env python3
"""CLI wrapper around ecb_forecast.compare."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecb_forecast.compare import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
