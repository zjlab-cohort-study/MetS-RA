"""Shared, disclosure-safe utilities for nested survival prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sksurv.util import Surv


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = [
        "data_path", "output_dir", "id_column", "event_column", "time_column",
        "target_population_column", "protein_columns_file",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration entries: {missing}")
    return config


def read_table(config: dict[str, Any]) -> pd.DataFrame:
    path = Path(config["data_path"])
    fmt = config.get("data_format", path.suffix.lstrip(".")).lower()
    if fmt in {"parquet", "pq"}:
        return pd.read_parquet(path)
    if fmt in {"csv", "csv.gz"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data format: {fmt}")


def target_cohort(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    column = config["target_population_column"]
    value = config.get("target_population_value", 1)
    cohort = data.loc[data[column] == value].copy().reset_index(drop=True)
    id_col = config["id_column"]
    event_col = config["event_column"]
    time_col = config["time_column"]
    if cohort[id_col].isna().any() or cohort[id_col].duplicated().any():
        raise ValueError("Participant identifier must be complete and unique")
    if not set(cohort[event_col].dropna().unique()).issubset({0, 1, False, True}):
        raise ValueError("Event indicator must be binary")
    if (cohort[time_col] <= 0).any():
        raise ValueError("Observed follow-up time must be positive")
    return cohort


def read_protein_columns(config: dict[str, Any]) -> list[str]:
    proteins = [x.strip() for x in Path(config["protein_columns_file"]).read_text().splitlines() if x.strip()]
    if len(proteins) != len(set(proteins)):
        raise ValueError("Protein column list contains duplicates")
    return proteins


def add_configured_transforms(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = data.copy()
    for source, target in config.get("log1p_source_columns", {}).items():
        out[target] = np.log1p(pd.to_numeric(out[source], errors="coerce").clip(lower=0))
    return out


def encode_categorical(data: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    out = data.copy()
    encoded_parts = []
    encoded_names: list[str] = []
    for column in config.get("categorical_predictors", []):
        levels = config.get("categorical_levels", {}).get(column)
        if not levels:
            raise ValueError(f"Prespecified categorical_levels are required for {column}")
        categorical = pd.Categorical(out[column], categories=levels, ordered=False)
        dummy = pd.get_dummies(categorical, prefix=column, drop_first=True, dtype=float)
        encoded_parts.append(dummy.set_axis(out.index))
        encoded_names.extend(dummy.columns.tolist())
    if config.get("categorical_predictors"):
        out = out.drop(columns=config["categorical_predictors"])
    if encoded_parts:
        out = pd.concat([out] + encoded_parts, axis=1)
    return out, encoded_names


def make_survival(frame: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    return Surv.from_arrays(
        frame[config["event_column"]].astype(bool).to_numpy(),
        frame[config["time_column"]].astype(float).to_numpy(),
    )


def administratively_censor(y: np.ndarray, tau: float) -> np.ndarray:
    event = y["event"].astype(bool) & (y["time"].astype(float) <= tau)
    time = np.minimum(y["time"].astype(float), tau + 1e-3)
    return Surv.from_arrays(event, time)


class TrainingTransformer:
    """Median imputation, optional winsorisation, and standardisation."""

    def __init__(self, features: list[str], winsor_features: set[str], lower: float, upper: float):
        self.features = features
        self.winsor_features = winsor_features
        self.lower_quantile = lower
        self.upper_quantile = upper

    def fit(self, frame: pd.DataFrame) -> "TrainingTransformer":
        x = frame[self.features].to_numpy(dtype=float, copy=True)
        self.median_ = np.nanmedian(x, axis=0)
        self.median_[~np.isfinite(self.median_)] = 0.0
        x = np.where(np.isfinite(x), x, self.median_)
        self.lower_ = np.full(x.shape[1], -np.inf)
        self.upper_ = np.full(x.shape[1], np.inf)
        for j, name in enumerate(self.features):
            if name in self.winsor_features:
                self.lower_[j], self.upper_[j] = np.quantile(
                    x[:, j], [self.lower_quantile, self.upper_quantile]
                )
        x = np.clip(x, self.lower_, self.upper_)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0, ddof=0)
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ < 1e-12)] = 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.features].to_numpy(dtype=float, copy=True)
        x = np.where(np.isfinite(x), x, self.median_)
        x = np.clip(x, self.lower_, self.upper_)
        return np.ascontiguousarray((x - self.mean_) / self.scale_, dtype=np.float64)

