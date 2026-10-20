"""Shared train and validation schedule.

Training draws K and option order with ``random.Random(seed + epoch)`` in
manifest order, matching ``v6_ovis`` ``_train_jobs``. Validation matches
``_val_jobs``: permute key ``val::{sample_id}`` and
``K = min(8, Uniform{3..8})`` from ``stable_int_seed``. Frame indices use the
same uniform formula as ``sample_rgb_frames``.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .common import ensure_v6_on_path, read_json, sha256_json, write_json
from .config import AblationConfig
from .paths import ResolvedPaths

ensure_v6_on_path()

from reljoin.data_perception import (  # noqa: E402
    deterministic_permute,
    permute_choices,
    resolve_video_path,
    stable_int_seed,
)
from reljoin.train import sample_K  # noqa: E402

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def uniform_plan(fps: float, total: int, k: int) -> tuple[list[float], list[int]]:
    """Match ``frame_io.sample_rgb_frames`` index and timestamp formula."""
    if k < 1:
        raise ValueError(f"K must be >= 1, got {k}")
    fps = float(fps or 25.0) or 25.0
    total = int(total or 1) or 1
    duration = total / fps
    times = [duration * i / max(k - 1, 1) for i in range(k)]
    indices = [min(total - 1, max(0, int(round(t * fps)))) for t in times]
    return times, indices


def load_split(path: str | Path) -> list[dict]:
    payload = read_json(Path(path))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError(f"Manifest has no samples: {path}")
    if int(payload.get("n", len(samples))) != len(samples):
        raise RuntimeError(f"Manifest n does not match samples in {path}")
    return samples


def video_counts(samples: list[dict]) -> set[str]:
    return {str(sample["video_id"]) for sample in samples}


def resolve_sample_video(sample: dict, videos_dir: str | None) -> str | None:
    recorded = Path(str(sample.get("video_path") or ""))
    if recorded.is_file():
        return str(recorded.resolve())
    if not videos_dir:
        return None
    stem = str(sample.get("video_id") or recorded.stem)
    hit = resolve_video_path(f"{stem}.mp4", Path(videos_dir))
    if hit is None and recorded.name:
        hit = resolve_video_path(recorded.name, Path(videos_dir))
    return str(hit) if hit is not None else None


def _probe_geometry(path: str) -> tuple[float, int]:
    import cv2

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0) or 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1) or 1
    finally:
        capture.release()
    return fps, total


def _row_base(sample: dict, *, split: str, order_index: int, epoch: int | None) -> dict:
    return {
        "split": split,
        "epoch": epoch,
        "order_index": order_index,
        "sample_id": str(sample["sample_id"]),
        "video_id": str(sample["video_id"]),
        "question": str(sample["question"]),
        "original_choices": list(sample["choices"]),
        "original_answer": str(sample["answer"]),
        "original_answer_index": int(sample["answer_index"]),
    }


def _with_permutation(row: dict, choices: list[str], answer_index: int, perm: list[int], k: int) -> dict:
    row = dict(row)
    row["choices"] = list(choices)
    row["answer_index"] = int(answer_index)
    row["answer_letter"] = LETTERS[int(answer_index)]
    row["permutation"] = list(perm)
    row["K"] = int(k)
    row["frame_indices"] = None
    row["timestamps"] = None
    row["video_path"] = None
    row["frames_ready"] = False
    row["frame_error"] = ""
    return row


def build_train_rows(samples: list[dict], seed: int, epochs: int) -> list[dict]:
    rows: list[dict] = []
    for epoch in range(1, int(epochs) + 1):
        rng = random.Random(int(seed) + epoch)
        for order, sample in enumerate(samples):
            k = int(sample_K(rng))
            choices, answer_index, perm = permute_choices(
                list(sample["choices"]), int(sample["answer_index"]), rng
            )
            row = _with_permutation(
                _row_base(sample, split="train", order_index=order, epoch=epoch),
                choices,
                answer_index,
                perm,
                k,
            )
            rows.append(row)
    return rows


def build_val_rows(samples: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for order, sample in enumerate(samples):
        choices, answer_index, perm = deterministic_permute(
            list(sample["choices"]),
            int(sample["answer_index"]),
            f"val::{sample['sample_id']}",
        )
        k = min(8, int(sample_K(random.Random(stable_int_seed(str(sample["sample_id"]))))))
        row = _with_permutation(
            _row_base(sample, split="val", order_index=order, epoch=None),
            choices,
            answer_index,
            perm,
            k,
        )
        rows.append(row)
    return rows


def attach_frame_plans(rows: list[dict], videos_dir: str | None, samples_by_id: dict[str, dict]) -> dict:
    """Fill indices from local videos. Missing files stay explicit."""
    geometry: dict[str, tuple[float, int] | str] = {}
    missing: dict[str, str] = {}
    ready = 0
    for row in rows:
        sample = samples_by_id[row["sample_id"]]
        resolved = resolve_sample_video(sample, videos_dir)
        if resolved is None:
            row["frames_ready"] = False
            row["frame_error"] = "missing_video"
            missing.setdefault(row["video_id"], str(sample.get("video_path") or ""))
            continue
        row["video_path"] = resolved
        cached = geometry.get(resolved)
        if isinstance(cached, str):
            row["frames_ready"] = False
            row["frame_error"] = cached
            missing.setdefault(row["video_id"], cached)
            continue
        if cached is None:
            try:
                cached = _probe_geometry(resolved)
            except Exception as exc:  # noqa: BLE001 - recorded per video, not swallowed silently later
                message = f"{type(exc).__name__}: {exc}"
                geometry[resolved] = message
                row["frames_ready"] = False
                row["frame_error"] = message
                missing.setdefault(row["video_id"], message)
                continue
            geometry[resolved] = cached
        fps, total = cached
        times, indices = uniform_plan(fps, total, int(row["K"]))
        row["timestamps"] = times
        row["frame_indices"] = indices
        row["frames_ready"] = True
        row["frame_error"] = ""
        ready += 1
    return {
        "rows_ready": ready,
        "rows_total": len(rows),
        "missing_videos": missing,
    }


def assert_canonical_split(cfg: AblationConfig, train: list[dict], val: list[dict]) -> None:
    train_videos = video_counts(train)
    val_videos = video_counts(val)
    if len(train) != int(cfg.expected_train_questions) or len(train_videos) != int(cfg.expected_train_videos):
        raise RuntimeError(
            f"Train split is {len(train)} questions / {len(train_videos)} videos, "
            f"expected {cfg.expected_train_questions}/{cfg.expected_train_videos}."
        )
    if len(val) != int(cfg.expected_val_questions) or len(val_videos) != int(cfg.expected_val_videos):
        raise RuntimeError(
            f"Val split is {len(val)} questions / {len(val_videos)} videos, "
            f"expected {cfg.expected_val_questions}/{cfg.expected_val_videos}."
        )
    overlap = train_videos & val_videos
    if overlap:
        raise RuntimeError(f"Train and val videos intersect: {sorted(overlap)[:8]}")
    train_ids = [row["sample_id"] for row in train]
    val_ids = [row["sample_id"] for row in val]
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
        raise RuntimeError("Duplicate sample ids in a split")
    if set(train_ids) & set(val_ids):
        raise RuntimeError("Train and val sample ids intersect")


def build_schedules(
    cfg: AblationConfig,
    resolved: ResolvedPaths,
    *,
    probe_frames: bool,
) -> dict[str, Any]:
    if not resolved.train_manifest or not resolved.val_manifest:
        raise FileNotFoundError("Canonical train/val manifests were not found. See resolved_paths.json.")
    train = load_split(resolved.train_manifest)
    val = load_split(resolved.val_manifest)
    assert_canonical_split(cfg, train, val)
    train_rows = build_train_rows(train, cfg.seed, cfg.max_epochs)
    val_rows = build_val_rows(val)
    samples = {str(sample["sample_id"]): sample for sample in train + val}
    frame_report = {
        "rows_ready": 0,
        "rows_total": len(train_rows) + len(val_rows),
        "missing_videos": {},
        "probed": False,
    }
    if probe_frames:
        frame_report = attach_frame_plans(train_rows + val_rows, resolved.videos_dir, samples)
        frame_report["probed"] = True
    identity = [
        {
            "split": row["split"],
            "epoch": row["epoch"],
            "order_index": row["order_index"],
            "sample_id": row["sample_id"],
            "K": row["K"],
            "permutation": row["permutation"],
            "answer_index": row["answer_index"],
            "answer_letter": row["answer_letter"],
            "frame_indices": row["frame_indices"],
            "timestamps": row["timestamps"],
        }
        for row in train_rows + val_rows
    ]
    meta = {
        "seed": cfg.seed,
        "epochs": cfg.max_epochs,
        "val_policy": cfg.val_policy,
        "train_questions": len(train),
        "train_videos": len(video_counts(train)),
        "val_questions": len(val),
        "val_videos": len(video_counts(val)),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "question_order": "manifest order; not reshuffled",
        "train_rng": "random.Random(seed + epoch), then sample_K and permute_choices per sample",
        "val_rng": "deterministic_permute val::{sample_id}; K=min(8, sample_K(Random(stable_int_seed)))",
        "frame_formula": "sample_rgb_frames uniform times and rounded indices",
        "frames": frame_report,
        "schedule_sha256": sha256_json(identity),
        "train_manifest": resolved.train_manifest,
        "val_manifest": resolved.val_manifest,
    }
    return {"train_rows": train_rows, "val_rows": val_rows, "meta": meta}


def write_schedules(cfg: AblationConfig, bundle: dict[str, Any]) -> Path:
    shared = cfg.output_dir() / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    train_path = shared / "train_schedule.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for row in bundle["train_rows"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(shared / "val_schedule.json", bundle["val_rows"])
    write_json(shared / "schedule_meta.json", bundle["meta"])
    missing = bundle["meta"]["frames"].get("missing_videos") or {}
    write_json(shared / "missing_videos.json", {"n": len(missing), "videos": missing})
    return shared


def load_schedule_files(cfg: AblationConfig) -> dict[str, Any]:
    shared = cfg.output_dir() / "shared"
    meta = read_json(shared / "schedule_meta.json")
    train_rows = []
    for line in (shared / "train_schedule.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            train_rows.append(json.loads(line))
    val_rows = read_json(shared / "val_schedule.json")
    return {"train_rows": train_rows, "val_rows": val_rows, "meta": meta}
