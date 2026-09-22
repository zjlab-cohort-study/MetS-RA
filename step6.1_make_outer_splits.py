#!/usr/bin/env python3
"""Create event-stratified repeated outer-fold assignments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import read_config, read_table, target_cohort  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = read_config(args.config)
    data = target_cohort(read_table(cfg), cfg)
    event = data[cfg["event_column"]].astype(int).to_numpy()
    participant = data[cfg["id_column"]].to_numpy()
    repeats = int(cfg.get("outer_repeats", 5))
    folds = int(cfg.get("outer_folds", 5))
    seed = int(cfg.get("random_seed", 20260812))
    rows = []
    for repeat_id in range(1, repeats + 1):
        rng = np.random.default_rng(seed + repeat_id)
        assignment = np.empty(len(data), dtype=int)
        for value in (0, 1):
            idx = np.flatnonzero(event == value)
            rng.shuffle(idx)
            assignment[idx] = np.resize(np.arange(1, folds + 1), len(idx))
        rows.append(pd.DataFrame({
            cfg["id_column"]: participant,
            "repeat_id": repeat_id,
            "outer_fold": assignment,
        }))
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(out / "outer_fold_assignments.csv", index=False)
    balance = result.merge(
        data[[cfg["id_column"], cfg["event_column"]]], on=cfg["id_column"], validate="many_to_one"
    ).groupby(["repeat_id", "outer_fold"])[cfg["event_column"]].agg(["size", "sum"])
    print(balance.to_string())


if __name__ == "__main__":
    main()

