"""Deterministic query selection. ROI is an analyst choice, not a model annotation."""
from __future__ import annotations

import numpy as np

from .geometry import is_verified


def select_queries(
    method: str,
    *,
    eta: np.ndarray,
    token_mask: np.ndarray,
    spatial_mask: np.ndarray,
    geometry: dict,
    max_queries: int,
    explicit: list[dict] | None = None,
    rois: list[dict] | None = None,
    seed: int = 0,
) -> dict:
    method = str(method or "top_eta")
    limit = max(1, int(max_queries))
    eta = np.asarray(eta, dtype=np.float64)
    token_mask = np.asarray(token_mask, dtype=bool)
    spatial_mask = np.asarray(spatial_mask, dtype=bool)
    if method == "explicit":
        pairs, notes = _explicit(explicit or [], token_mask, limit)
        rule = "explicit (frame_index, token_index) from the selection manifest"
    elif method == "roi":
        pairs, notes = _roi(rois or [], geometry, eta, token_mask, limit)
        rule = "ROI point on the original frame, mapped through confirmed token bboxes"
    elif method == "top_eta":
        pairs, notes = _top_eta(eta, token_mask, spatial_mask, limit)
        rule = (
            "deterministic top eta among real tokens, preferring one token per frame "
            "then a minimum token-index gap; no random sampling"
        )
    else:
        raise ValueError(f"unknown query method {method!r}")
    return {
        "method": method,
        "seed": int(seed),
        "rule": rule,
        "max_queries": limit,
        "queries": [{"frame_index": int(i), "token_index": int(p)} for i, p in pairs],
        "notes": notes,
        "mapping_status": geometry.get("mapping_status"),
    }


def _explicit(items, token_mask, limit):
    notes = []
    pairs = []
    k, pmax = token_mask.shape
    for item in items:
        frame_i = int(item["frame_index"] if "frame_index" in item else item["frame"])
        token_p = int(item["token_index"] if "token_index" in item else item["token"])
        if not (0 <= frame_i < k and 0 <= token_p < pmax and token_mask[frame_i, token_p]):
            notes.append(f"dropped invalid explicit query {(frame_i, token_p)}")
            continue
        pairs.append((frame_i, token_p))
        if len(pairs) >= limit:
            break
    if not pairs:
        notes.append("no valid explicit queries")
    return pairs, notes


def _roi(points, geometry, eta, token_mask, limit):
    notes = []
    if not is_verified(str(geometry.get("mapping_status"))):
        notes.append("ROI was not mapped because spatial_mapping_unverified")
        return [], notes
    hits = []
    for point in points:
        frame_i = int(point["frame_index"] if "frame_index" in point else point["frame"])
        x = float(point["x"])
        y = float(point["y"])
        contained = []
        for token in geometry["tokens"]:
            if int(token["frame_index"]) != frame_i or not token["spatial"]:
                continue
            box = token["bbox"]
            if box is None:
                continue
            if box[0] <= x < box[2] and box[1] <= y < box[3]:
                token_p = int(token["token_index"])
                area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
                contained.append((area, -float(eta[frame_i, token_p]), token_p))
        if not contained:
            notes.append(f"ROI {(frame_i, x, y)} missed every spatial token")
            continue
        contained.sort()
        hits.append((frame_i, int(contained[0][2])))
        if len(hits) >= limit:
            break
    return hits, notes


def _top_eta(eta, token_mask, spatial_mask, limit):
    notes = []
    use_spatial = bool(spatial_mask.any())
    mask = spatial_mask if use_spatial else token_mask
    if use_spatial:
        notes.append("top eta is restricted to spatial tokens")
    else:
        notes.append("no spatial tokens are confirmed; top eta uses real token indices")
    k, pmax = eta.shape
    candidates = []
    for frame_i in range(k):
        for token_p in range(pmax):
            if not mask[frame_i, token_p]:
                continue
            candidates.append((-float(eta[frame_i, token_p]), frame_i, token_p))
    candidates.sort()
    picked: list[tuple[int, int]] = []
    used_frames: set[int] = set()
    for _score, frame_i, token_p in candidates:
        if frame_i in used_frames:
            continue
        picked.append((frame_i, token_p))
        used_frames.add(frame_i)
        if len(picked) >= limit:
            return picked, notes
    gap = max(1, pmax // 16)
    picked_set = set(picked)
    for _score, frame_i, token_p in candidates:
        if (frame_i, token_p) in picked_set:
            continue
        if any(pf == frame_i and abs(pp - token_p) < gap for pf, pp in picked):
            continue
        picked.append((frame_i, token_p))
        picked_set.add((frame_i, token_p))
        if len(picked) >= limit:
            break
    return picked, notes
