"""Best-of-N progress rates count error records as failures."""
from textworld_parallel.summary import summarize_records as _summarize, write_json_atomic


def summarize_records(records, manifest):
    report = _summarize(records, manifest)
    for bucket in [report["overall"], *report["by_game"].values(), *report["by_difficulty"].values()]:
        bucket["valid_accuracy"] = bucket["accuracy"]
        bucket["failed"] = bucket["completed"] - bucket["successful"]
        bucket["accuracy"] = bucket["successful"] / bucket["completed"] if bucket["completed"] else 0.0
    return report
