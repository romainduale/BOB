"""Parkinsons package root.

Exposes ``PROJECT_ROOT``: the absolute path to the project root,
derived from this file's location so experiment scripts run correctly
from any working directory.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
