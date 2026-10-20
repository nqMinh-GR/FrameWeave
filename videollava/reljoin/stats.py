"""Paired statistical tests for method comparisons."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def paired_bootstrap_ci(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    a = np.asarray(correct_a, dtype=np.float64)
    b = np.asarray(correct_b, dtype=np.float64)
    assert a.shape == b.shape
    n = len(a)
    if n == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    delta = float(b.mean() - a.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return {
        "delta": delta,
        "ci_low": lo,
        "ci_high": hi,
        "n": int(n),
        "n_resamples": n_resamples,
    }


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    """Exact McNemar test on paired dichotomous outcomes (b vs a)."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    # b correct, a wrong
    n01 = int((~a & b).sum())
    # a correct, b wrong
    n10 = int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "statistic": 0.0, "p_value": 1.0}
    try:
        from scipy.stats import binomtest

        p = float(binomtest(n01, n=n, p=0.5, alternative="two-sided").pvalue)
    except ImportError:
        from math import exp, lgamma, log

        p = 0.0
        log_half_n = n * log(0.5)
        for k in range(n + 1):
            if abs(k - n / 2) >= abs(n01 - n / 2):
                logc = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
                p += exp(logc + log_half_n)
        p = min(1.0, p)
    return {
        "n01": n01,
        "n10": n10,
        "statistic": float((abs(n01 - n10) - 1) ** 2 / n) if n else 0.0,
        "p_value": min(1.0, p),
    }


def win_tie_loss(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    return {
        "wins_b": int((b & ~a).sum()),
        "wins_a": int((a & ~b).sum()),
        "ties": int((a == b).sum()),
    }


def compare_methods(
    rows_a: list[dict],
    rows_b: list[dict],
    id_key: str = "sample_id",
    correct_key: str = "is_correct",
) -> dict:
    map_a = {str(r[id_key]): bool(r[correct_key]) for r in rows_a}
    map_b = {str(r[id_key]): bool(r[correct_key]) for r in rows_b}
    common = sorted(set(map_a) & set(map_b))
    ca = [map_a[i] for i in common]
    cb = [map_b[i] for i in common]
    return {
        "n_paired": len(common),
        "bootstrap": paired_bootstrap_ci(ca, cb),
        "mcnemar": mcnemar_exact(ca, cb),
        "win_tie_loss": win_tie_loss(ca, cb),
        "acc_a": float(np.mean(ca)) if ca else 0.0,
        "acc_b": float(np.mean(cb)) if cb else 0.0,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, same order as input."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: float(p_values[i]))
    adj = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        holm = min(1.0, float(p_values[i]) * (m - rank))
        running = max(running, holm)
        adj[i] = running
    return adj


def cluster_bootstrap_delta(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Paired cluster bootstrap of mean(b) - mean(a).

    scores_a, scores_b: [N] or [N, R] (rotations averaged per sample first).
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.ndim == 2:
        a = a.mean(axis=1)
    if b.ndim == 2:
        b = b.mean(axis=1)
    if a.shape != b.shape:
        raise ValueError(f"cluster bootstrap shape mismatch {a.shape} vs {b.shape}")
    n = int(a.size)
    if n == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "n": 0}
    delta = float(b.mean() - a.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    frac_le = float((boot <= 0.0).mean())
    frac_ge = float((boot >= 0.0).mean())
    p_value = float(min(1.0, 2.0 * min(frac_le, frac_ge)))
    return {
        "delta": delta,
        "ci_low": float(np.quantile(boot, alpha / 2)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha / 2)),
        "p_value": p_value,
        "n": n,
        "n_resamples": int(n_resamples),
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap CI of a mean. `values` is [N] or [N, R] (mean over R first)."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    mean = float(x.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot = x[idx].mean(axis=1)
    return {
        "mean": mean,
        "ci_low": float(np.quantile(boot, alpha / 2)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha / 2)),
        "n": n,
        "n_resamples": int(n_resamples),
    }


def save_statistical_tests(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
