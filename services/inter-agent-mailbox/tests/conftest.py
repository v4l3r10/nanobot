"""Pytest config for the inter-agent mailbox service tests.

Adds ``src/`` to ``sys.path`` so individual tests can import
``nanobot_mailbox`` without a prior editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
