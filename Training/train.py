"""AReaL local/Ray launcher entry point (main(args) is invoked on each trainer)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sera_training.bootstrap import bootstrap

bootstrap()


def main(args):
    from sera_training.trainer import main as train
    train(args)


if __name__ == "__main__":
    main(sys.argv[1:])
