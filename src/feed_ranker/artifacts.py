"""Atomic run artifacts and source provenance for reproducible offline experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch


def write_json(path: Path, value: dict[str, Any] | list[Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def collect_run_provenance() -> dict[str, Any]:
    return {
        "commit": _git_output("rev-parse", "HEAD"),
        "working_diff": _git_output("diff", "HEAD"),
        "working_status": _git_output("status", "--short"),
        "torch_version": str(torch.__version__),
        "source_sha256": {
            str(path): sha256(path) for path in Path(__file__).resolve().parent.rglob("*.py")
        },
    }


def _git_output(*args: str) -> str:
    # A CLI may be launched elsewhere with this source tree on PYTHONPATH.
    source_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["git", *args], cwd=source_root, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def sha256(path: Path) -> str:
    """Fingerprint file bytes so runs can detect changed data or source code."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
