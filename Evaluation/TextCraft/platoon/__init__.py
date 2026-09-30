"""Compatibility namespace; all Platoon implementation lives in Runtime."""
from pathlib import Path

_repo = Path(__file__).resolve().parents[3]
__path__ = [str(_repo / "Runtime/vendor/platoon")]
