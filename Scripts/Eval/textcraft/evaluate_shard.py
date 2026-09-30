#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Evaluation"))
from evaluation_common import cli

if __name__ == "__main__":
    cli("TextCraft", "evaluate_shard")
