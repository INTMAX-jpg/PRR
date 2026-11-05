#!/usr/bin/env python3
"""Shared, auditable utilities for Experiment E.

The helpers here intentionally keep metric definitions and feature loading in
one place so that E_01--E_04 cannot silently disagree.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


LABELS = ("A", "B", "C", "D")


class ShrunkenDiffMean:
    """One-vs-rest difference in means with soft-thresholded directions."""

    def __init__(self, n_classes: int, threshold: float = 0.0):
        self.n_classes = int(n_classes)
        self.threshold = float(threshold)
        self.scaler = StandardScaler()
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        z = self.scaler.fit_transform(x)
        self.W = np.zeros((self.n_classes, z.shape[1]), dtype=np.float64)
        self.b = np.zeros(self.n_classes, dtype=np.float64)
        for label in range(self.n_classes):
            positive = z[y == label]
            negative = z[y != label]
            if not len(positive) or not len(negative):
                raise ValueError(f"Missing positive/negative rows for class {label}")
            mu_positive = positive.mean(axis=0)
            mu_negative = negative.mean(axis=0)
            raw_direction = mu_positive - mu_negative
            direction = np.sign(raw_direction) * np.maximum(
                np.abs(raw_direction) - self.threshold, 0.0
            )
            self.W[label] = direction
            self.b[label] = -0.5 * float(
                np.dot(direction, mu_positive + mu_negative)
            )
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.W is None or self.b is None:
            raise RuntimeError("Model is not fitted")
        return self.scaler.transform(x) @ self.W.T + self.b

    def raw_weights(self) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("Model is not fitted")
        return self.W / np.maximum(self.scaler.scale_[None, :], 1e-12)


def json_load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_fingerprint(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def command_record() -> str:
    return " ".join([sys.executable, *sys.argv])


def label_to_index(labels: Sequence[str]) -> Dict[str, int]:
    return {str(label): index for index, label in enumerate(labels)}


def labels_as_indices(values: pd.Series, mapping: Dict[str, int]) -> np.ndarray:
    normalized = {str(key): int(value) for key, value in mapping.items()}
    converted = values.astype(str).map(normalized)
    if converted.isna().any():
        bad = sorted(values[converted.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unknown labels: {bad}")
    return converted.astype(int).to_numpy()


def feature_key(site: str, layer: int) -> str:
    return f"{site}_L{int(layer)}"


def parse_feature_key(key: str) -> tuple[str, int]:
    if "_L" not in key:
        raise ValueError(f"Feature key must look like resid_pre_L22, got {key!r}")
    site, layer = key.rsplit("_L", 1)
    return site, int(layer)


def resolve_feature_path(run_dir: Path, manifest: Dict[str, Any], key: str) -> Path:
    candidates: list[Path] = []
    saved = manifest.get("feature_paths", {}).get(key)
    if saved:
        saved_path = Path(saved)
        candidates.extend(
            [
                saved_path,
                run_dir / saved_path,
                run_dir / "activations" / saved_path.name,
            ]
        )
    candidates.append(run_dir / "activations" / f"{key}.npy")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve activation {key}. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def assert_row_alignment(meta: pd.DataFrame, n_rows: int) -> None:
    if len(meta) != n_rows:
        raise ValueError(f"metadata rows={len(meta)} but activations rows={n_rows}")
    expected = np.arange(n_rows)
    actual = meta["row_idx"].astype(int).to_numpy()
    if not np.array_equal(actual, expected):
        raise ValueError("metadata.row_idx must be exactly 0..n_rows-1")


def clean_t0_frame(
    meta: pd.DataFrame,
    trajectory_type: str = "wrong_pressure",
    turn_index: int = 0,
) -> pd.DataFrame:
    clean = meta[
        meta["trajectory_type"].astype(str).eq(str(trajectory_type))
        & meta["turn_index"].astype(int).eq(int(turn_index))
    ].copy()
    if clean.empty:
        raise ValueError(
            f"No clean rows for trajectory_type={trajectory_type!r}, turn={turn_index}"
        )
    duplicates = clean.duplicated("base_id", keep=False)
    if duplicates.any():
        examples = clean.loc[duplicates, "base_id"].astype(str).head(10).tolist()
        raise ValueError(f"Clean T0 must contain one row per base_id; duplicates={examples}")
    return clean


def assert_split_disjoint(frame: pd.DataFrame, group_col: str = "base_id") -> None:
    counts = frame.groupby(group_col)["split"].nunique()
    bad = counts[counts > 1]
    if len(bad):
        raise ValueError(
            f"{len(bad)} {group_col} values occur in more than one split; "
            f"examples={bad.index.astype(str).tolist()[:10]}"
        )


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64) / max(float(temperature), 1e-8)
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.maximum(exp_x.sum(axis=1, keepdims=True), 1e-12)


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    pred = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        lo, hi = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(ece)


def multiclass_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    temperature: float = 1.0,
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    score_array = np.asarray(scores, dtype=np.float64)
    pred = score_array.argmax(axis=1)
    probabilities = softmax(score_array, temperature=temperature)
    gold_probability = probabilities[np.arange(len(y)), y]
    one_hot = np.eye(score_array.shape[1], dtype=np.float64)[y]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "nll": float(-np.log(np.maximum(gold_probability, 1e-12)).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_10bin": expected_calibration_error(y, probabilities, n_bins=10),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }


def gold_margin(scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y_true, dtype=int)
    gold = x[np.arange(len(y)), y]
    other = x.copy()
    other[np.arange(len(y)), y] = -np.inf
    return gold - other.max(axis=1)


def pair_margin(
    scores: np.ndarray, y_true: np.ndarray, wrong_index: np.ndarray
) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y_true, dtype=int)
    wrong = np.asarray(wrong_index, dtype=int)
    row = np.arange(len(y))
    return x[row, y] - x[row, wrong]


def select_temperature(y_dev: np.ndarray, dev_scores: np.ndarray) -> float:
    grid = np.geomspace(0.05, 20.0, 81)
    losses = []
    for temperature in grid:
        probabilities = softmax(dev_scores, temperature=float(temperature))
        gold = probabilities[np.arange(len(y_dev)), y_dev]
        losses.append(float(-np.log(np.maximum(gold, 1e-12)).mean()))
    best = int(np.argmin(losses))
    return float(grid[best])


def cluster_bootstrap_mean(
    values: np.ndarray,
    groups: Iterable[Any],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    value_array = np.asarray(values, dtype=np.float64)
    group_array = np.asarray(list(groups))
    if len(value_array) != len(group_array):
        raise ValueError("values and groups must have equal length")
    valid = np.isfinite(value_array)
    value_array = value_array[valid]
    group_array = group_array[valid]
    if not len(value_array):
        return {"estimate": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    unique_groups = np.unique(group_array)
    group_values = {
        group: value_array[group_array == group] for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(n_bootstrap), dtype=np.float64)
    for index in range(int(n_bootstrap)):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        chunks = [group_values[group] for group in sampled]
        estimates[index] = float(np.concatenate(chunks).mean())
    return {
        "estimate": float(value_array.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


def module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def get_final_norm(model) -> torch.nn.Module:
    candidates = [
        ("model", "norm"),
        ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"),
    ]
    for parent_name, child_name in candidates:
        parent = getattr(model, parent_name, None)
        child = getattr(parent, child_name, None) if parent is not None else None
        if child is not None:
            return child
    raise ValueError("Could not locate final normalization module")


def get_output_head(model) -> torch.nn.Module:
    head = model.get_output_embeddings()
    if head is None:
        head = getattr(model, "lm_head", None)
    if head is None:
        raise ValueError("Could not locate language-model output head")
    return head


def project_label_logits(
    model,
    hidden: torch.Tensor,
    label_token_ids: Sequence[int],
    apply_final_norm: bool,
) -> torch.Tensor:
    """Project hidden vectors to only the requested vocabulary logits."""
    norm = get_final_norm(model)
    head = get_output_head(model)
    norm_device = module_device(norm)
    head_device = module_device(head)
    x = hidden.to(norm_device)
    with torch.inference_mode():
        if apply_final_norm:
            x = norm(x)
        x = x.to(head_device)
        weight = head.weight[label_token_ids]
        logits = x.to(weight.dtype) @ weight.transpose(0, 1)
        bias = getattr(head, "bias", None)
        if bias is not None:
            logits = logits + bias[label_token_ids]
    return logits


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    if valid.sum() < 3:
        return math.nan
    x_rank = pd.Series(x_array[valid]).rank(method="average").to_numpy()
    y_rank = pd.Series(y_array[valid]).rank(method="average").to_numpy()
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return math.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def through_origin_slope(x: np.ndarray, y: np.ndarray) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array, y_array = x_array[valid], y_array[valid]
    denominator = float(np.dot(x_array, x_array))
    if denominator <= 1e-12:
        return math.nan
    return float(np.dot(x_array, y_array) / denominator)
