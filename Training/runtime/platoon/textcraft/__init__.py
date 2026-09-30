from pathlib import Path
import platoon

__path__ = [str(Path(__file__).parent), str(Path(platoon.__path__[1]) / "textcraft")]
