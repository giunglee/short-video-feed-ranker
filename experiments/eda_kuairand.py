"""Configured raw-source audit with no dependency on historical census outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from feed_ranker.artifacts import collect_run_provenance, sha256, write_json
from feed_ranker.data.kuairand import TRAIN_END, VAL_END
from feed_ranker.data.profiling import profile_events, profile_fields, read_csv_sample


def audit(
    events: list[Path],
    directory: Path,
    *,
    tables: dict[str, Path] | None = None,
    max_rows: int = 100_000,
    train_end: int = TRAIN_END,
    val_end: int = VAL_END,
) -> dict[str, Any]:
    if directory.exists():
        raise FileExistsError(directory)
    sources = {"events": events, **{name: [path] for name, path in (tables or {}).items()}}
    if tables and "events" in tables:
        raise ValueError("events is reserved for raw event logs")
    result: dict[str, Any] = {
        "version": "raw-audit-v1",
        "max_rows_per_table": max_rows,
        "train_end": train_end,
        "val_end": val_end,
        "assumptions": [
            "Descriptive prefix audit when bounded; no population-performance estimate",
            "Snapshot historical availability is unverified",
            "No model fitting, selection or final-window scoring",
        ],
        "provenance": collect_run_provenance(),
        "audit_source_sha256": sha256(Path(__file__)),
        "tables": {},
    }
    for name, paths in sources.items():
        frame, truncated = read_csv_sample(paths, max_rows)
        result["tables"][name] = {
            "source_sha256": {str(p): sha256(p) for p in paths},
            "truncated": truncated,
            "rows_read": len(frame),
            "fields": profile_fields(frame),
        }
        if name == "events":
            result["events"] = profile_events(frame, train_end, val_end)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "audit.json", result)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = result["events"]["windows"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(windows), [w["warm_item_events"] for w in windows.values()], label="Warm item")
    ax.bar(
        list(windows),
        [w["cold_item_events"] for w in windows.values()],
        bottom=[w["warm_item_events"] for w in windows.values()],
        label="Cold item",
    )
    ax.set_ylabel("Observed events in audited prefix")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "cohorts.png", dpi=140)
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--table", action="append", default=[], metavar="NAME=CSV")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100_000,
        help="Per-table prefix cap; 0 reads all rows into memory",
    )
    parser.add_argument("--train-end", type=int, default=TRAIN_END)
    parser.add_argument("--val-end", type=int, default=VAL_END)
    args = parser.parse_args()
    tables = {}
    for entry in args.table:
        name, path = entry.split("=", 1)
        if not name or name in tables:
            parser.error("Require unique table names")
        tables[name] = Path(path)
    audit(
        args.events,
        args.out,
        tables=tables,
        max_rows=args.max_rows,
        train_end=args.train_end,
        val_end=args.val_end,
    )


if __name__ == "__main__":
    main()
