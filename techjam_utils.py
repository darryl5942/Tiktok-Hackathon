"""Shared helpers for the TikTok TechJam AIGC detector project.

These functions stay intentionally lightweight and dependency-free so they can
be imported from tests and small CLI entrypoints without pulling in the full
model stack.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence


def env_flag(name: str, default: bool = False) -> bool:
    """Reads a boolean-like environment variable."""
    import os

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def portable_identifier(path: str | Path, anchor: str | Path | None = None) -> str:
    """Creates a path label that avoids machine-specific absolute prefixes when possible.

    Always uses forward slashes, even on Windows, so identifiers are stable
    across machines/OSes rather than depending on os.sep.
    """
    p = Path(path)
    if anchor is not None:
        try:
            return p.resolve().relative_to(Path(anchor).resolve()).as_posix()
        except Exception:
            pass
    try:
        return p.resolve().name
    except Exception:
        return p.name


def load_labeled_csv_rows(csv_path: str | Path) -> list[dict]:
    """Loads a benchmark CSV and validates the required columns."""
    csv_path = Path(csv_path)
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    validate_labeled_csv_rows(rows, source=str(csv_path))
    return rows


def validate_labeled_csv_rows(rows: Sequence[dict], source: str = "csv") -> None:
    """Ensures a benchmark CSV has the expected schema and labels."""
    required = {"image_path", "label"}
    if not rows:
        raise ValueError(f"{source} is empty")
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    for i, row in enumerate(rows, start=1):
        if "image_path" not in row or "label" not in row:
            raise ValueError(f"{source} row {i} is missing required fields")
        try:
            label = int(row["label"])
        except Exception as exc:
            raise ValueError(f"{source} row {i} has a non-integer label: {row['label']!r}") from exc
        if label not in {0, 1}:
            raise ValueError(f"{source} row {i} has invalid label {label}; expected 0 or 1")


def benchmark_rows_to_samples(rows: Iterable[dict]) -> list[tuple[str, int]]:
    """Converts benchmark rows into (image_path, label) tuples."""
    samples = []
    for row in rows:
        samples.append((row["image_path"], int(row["label"])))
    return samples


def find_best_threshold(labels: Sequence[int], scores: Sequence[float]) -> tuple[float, float]:
    """Finds the threshold that maximizes binary accuracy on labeled scores."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if not labels:
        raise ValueError("cannot calibrate threshold on an empty set")

    paired = sorted(zip(scores, labels))
    candidate_thresholds = {0.5}
    candidate_thresholds.update(score for score, _ in paired)

    for i in range(len(paired) - 1):
        left = paired[i][0]
        right = paired[i + 1][0]
        candidate_thresholds.add((left + right) / 2.0)

    best_threshold = 0.5
    best_acc = -1.0
    total = len(labels)
    for threshold in sorted(candidate_thresholds):
        correct = sum(int((score > threshold) == bool(label)) for score, label in zip(scores, labels))
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    return best_threshold, best_acc
