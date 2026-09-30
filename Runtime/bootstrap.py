"""Expose the vendored Platoon namespace without depending on an application."""
from pathlib import Path
import sys

RUNTIME_ROOT = Path(__file__).resolve().parent


def bootstrap() -> None:
    vendor = str(RUNTIME_ROOT / "vendor")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
