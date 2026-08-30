#!/usr/bin/env python3
"""Small CLI wrapper for the AIGC detector project.

The main detector script still performs the full pipeline, but this wrapper
makes the common modes explicit and easier to discover:

- train: run training/resume, then evaluation
- infer: skip training and run inference/evaluation on `inference_images/`
- eval: skip training and run inference/evaluation, plus optional benchmark CSV
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT = PROJECT_ROOT / "aigc_detector_3.py"


def _run_script(env_overrides: dict[str, str]) -> int:
    env = os.environ.copy()
    env.update(env_overrides)
    proc = subprocess.run([sys.executable, str(SCRIPT)], cwd=str(PROJECT_ROOT), env=env)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="TikTok TechJam AIGC detector CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train or resume the detector")
    train.add_argument("--wildfake-labels-csv", default=None, help="Optional OOD benchmark CSV")

    infer = sub.add_parser("infer", help="Skip training and run inference/eval")
    infer.add_argument("--input-dir", default=None, help="Directory of images to score")
    infer.add_argument("--wildfake-labels-csv", default=None, help="Optional OOD benchmark CSV")

    eval_cmd = sub.add_parser("eval", help="Skip training and run evaluation/reporting")
    eval_cmd.add_argument("--wildfake-labels-csv", default=None, help="Optional OOD benchmark CSV")

    args = parser.parse_args()

    if args.command == "train":
        overrides = {"AIGC_SKIP_TRAINING": "0"}
        if args.wildfake_labels_csv:
            overrides["WILDFAKE_LABELS_CSV"] = args.wildfake_labels_csv
        return _run_script(overrides)

    if args.command == "infer":
        overrides = {"AIGC_SKIP_TRAINING": "1"}
        if args.input_dir:
            overrides["INFERENCE_INPUT_DIR"] = args.input_dir
        if args.wildfake_labels_csv:
            overrides["WILDFAKE_LABELS_CSV"] = args.wildfake_labels_csv
        return _run_script(overrides)

    if args.command == "eval":
        overrides = {"AIGC_SKIP_TRAINING": "1"}
        if args.wildfake_labels_csv:
            overrides["WILDFAKE_LABELS_CSV"] = args.wildfake_labels_csv
        return _run_script(overrides)

    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
