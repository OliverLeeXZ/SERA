"""Training-only extensions over the shared Runtime Platoon execution core."""
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
__path__ = [str(Path(__file__).parent), str(_root / "Runtime/vendor/platoon")]
