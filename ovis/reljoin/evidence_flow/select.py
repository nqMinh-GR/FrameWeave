"""Keyframe and token-set selection from real alpha and beta. No image content is used."""
from __future__ import annotations

import numpy as np

K_CANDIDATES = (4, 8, 12)
TOP1_MIN = 0.25
TOP2_MIN = 0.50


def normalized_entropy(weights) -> float:
    values = np.asarray(weights, dtype=np.float64)
    values = values[values > 1e-12]
    if values.size <= 1:
        return 0.0
    probs = values / values.sum()
    entropy = float(-(probs * np.log(probs)).sum())
    return entropy / float(np.log(values.size))


def incoming_profile(beta: np.ndarray) -> list[dict]:
    """``beta[i, j]`` is source ``j`` read by receiver ``i``."""
    matrix = np.asarray(beta, dtype=np.float64)
    k = int(matrix.shape[0])
    profiles = []
    for receiver in range(k):
        ranked = sorted(
            ((source, float(matrix[receiver, source])) for source in range(k) if source != receiver),
            key=lambda item: (-item[1], item[0]),
        )
        top1 = ranked[0][1] if ranked else 0.0
        top2 = top1 + (ranked[1][1] if len(ranked) > 1 else 0.0)
        entropy = normalized_entropy([weight for _, weight in ranked])
        sharp = top1 >= TOP1_MIN or (top2 >= TOP2_MIN and entropy <= 0.80)
        profiles.append(
            {
                "receiver": receiver,
                "ranked": [{"source": source, "beta": weight} for source, weight in ranked],
                "top1": top1,
                "top2_sum": top2,
                "entropy": entropy,
                "sharp": bool(sharp),
            }
        )
    return profiles


def select_keyframes(beta: np.ndarray, *, min_frames: int = 3, max_frames: int = 4) -> dict:
    matrix = np.asarray(beta, dtype=np.float64)
    k = int(matrix.shape[0])
    profiles = incoming_profile(matrix)
    edges = []
    for profile in profiles:
        if not profile["sharp"] or not profile["ranked"]:
            continue
        edges.append((profile["receiver"], profile["ranked"][0]["source"], profile["ranked"][0]["beta"]))
        if profile["top2_sum"] >= TOP2_MIN and len(profile["ranked"]) > 1:
            edges.append(
                (profile["receiver"], profile["ranked"][1]["source"], profile["ranked"][1]["beta"])
            )
    edges.sort(key=lambda item: (-item[2], item[0], item[1]))
    chosen = []
    for receiver, source, _weight in edges:
        for frame in (receiver, source):
            if frame not in chosen:
                chosen.append(frame)
        if len(chosen) >= max_frames:
            break
    if len(chosen) < min_frames:
        strength = matrix.sum(axis=0) + matrix.sum(axis=1)
        for frame in np.argsort(-strength):
            frame = int(frame)
            if frame not in chosen:
                chosen.append(frame)
            if len(chosen) >= min_frames:
                break
    chosen = sorted(chosen[:max_frames])
    displayed = []
    for receiver in chosen:
        incoming = [
            (source, float(matrix[receiver, source]))
            for source in chosen
            if source != receiver
        ]
        incoming.sort(key=lambda item: (-item[1], item[0]))
        keep = 1
        if len(incoming) > 1 and incoming[0][1] + incoming[1][1] >= TOP2_MIN:
            keep = 2
        for source, weight in incoming[:keep]:
            if weight <= 0:
                continue
            displayed.append({"receiver": receiver, "source": source, "beta": weight})
    off = float(matrix.sum() - np.trace(matrix))
    captured = float(sum(item["beta"] for item in displayed))
    return {
        "keyframes": chosen,
        "edges": displayed,
        "profiles": profiles,
        "sharp": any(profile["sharp"] for profile in profiles),
        "captured_beta": captured,
        "off_diagonal_beta": off,
        "captured_fraction": (captured / off) if off > 0 else 0.0,
        "n_frames": k,
    }


def top_alpha_tokens(alpha_row: np.ndarray, valid: np.ndarray, k: int) -> list[int]:
    scores = np.asarray(alpha_row, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    order = np.argsort(-np.where(mask, scores, -1.0))
    picked = [int(index) for index in order if mask[index]]
    return picked[: int(k)]


def token_mass(alpha_row: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    return float(np.asarray(alpha_row, dtype=np.float64)[indices].sum())
