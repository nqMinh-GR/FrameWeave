"""HERBench oracle frame selection — no Video-LLaVA / DTR dependency.

Default eval config is lite_v2. Config ``full`` loads 27,631 questions and
resolves videos by relative path (no Drive FUSE ``rglob``).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module_from_file(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_time_token(token: str) -> float | None:
    token = token.strip()
    if not token or token.lower() == "all":
        return None
    m = re.match(
        r"^(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:[.:](\d+))?$",
        token,
    )
    if m:
        hours = int(m.group(1) or 0)
        minutes = int(m.group(2))
        seconds = int(m.group(3))
        frac = m.group(4) or "0"
        frac_sec = int(frac) / (10 ** len(frac)) if frac else 0.0
        return hours * 3600 + minutes * 60 + seconds + frac_sec
    try:
        return float(token)
    except ValueError:
        return None


def parse_span(span) -> tuple[float, float] | None:
    if span is None:
        return None
    if isinstance(span, (int, float)):
        t = float(span)
        return (t, t)
    if not isinstance(span, str):
        return None
    s = span.strip()
    if not s or s.lower() == "all":
        return None
    if " - " in s:
        left, right = s.split(" - ", 1)
    else:
        m = re.match(r"^(\d[\d:.]*)\s*-\s*(\d[\d:.]*)$", s)
        if m:
            left, right = m.group(1), m.group(2)
        else:
            t = _parse_time_token(s)
            return (t, t) if t is not None else None
    a = _parse_time_token(left.strip())
    b = _parse_time_token(right.strip())
    if a is None and b is None:
        return None
    if a is None:
        a = b
    if b is None:
        b = a
    if a > b:
        a, b = b, a
    return (a, b)


def extract_evidence_spans(metadata: dict) -> tuple[list[tuple[float, float]], str]:
    if not metadata:
        return [], "missing_metadata"
    for key in ("timestamps", "required_timestamps"):
        raw = metadata.get(key)
        if raw is None:
            continue
        if isinstance(raw, str) and raw.strip().lower() == "all":
            return [], "uniform_fallback"
        spans: list[tuple[float, float]] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, list):
                    continue
                span = parse_span(item)
                if span is not None:
                    spans.append(span)
        else:
            span = parse_span(raw)
            if span is not None:
                spans.append(span)
        if spans:
            return spans, "oracle"
    return [], "uniform_fallback"


def choose_oracle_times(
    spans: list[tuple[float, float]],
    duration: float,
    num_frames: int = 8,
) -> tuple[list[float], bool, str]:
    duration = max(float(duration or 1.0), 1e-3)
    if not spans:
        times = [duration * i / max(num_frames - 1, 1) for i in range(num_frames)]
        return times, False, "uniform_fallback"

    candidates = [max(0.0, min(duration, 0.5 * (a + b))) for a, b in spans]
    candidates = sorted(set(round(t, 4) for t in candidates))
    if len(candidates) >= num_frames:
        idxs = np.linspace(0, len(candidates) - 1, num_frames)
        times = [candidates[int(round(i))] for i in idxs]
        return times, False, "oracle"

    times = list(candidates)
    padded = False
    span_i = 0
    while len(times) < num_frames and spans:
        a, b = spans[span_i % len(spans)]
        if b - a < 1e-3:
            t = max(0.0, min(duration, a + 0.04 * (1 + len(times))))
        else:
            frac = ((len(times) + 1) % 5 + 1) / 6.0
            t = a + frac * (b - a)
            t = max(0.0, min(duration, t))
        times.append(round(t, 4))
        times = sorted(set(times))
        span_i += 1
        padded = True
        if span_i > num_frames * 20:
            break
    seen = set(times)
    guard = 0
    while len(times) < num_frames and guard < num_frames * 50:
        guard += 1
        base = duration * len(times) / max(num_frames - 1, 1)
        t = round(max(0.0, min(duration, base)), 4)
        nudge = 0
        while t in seen and nudge < 100000:
            nudge += 1
            step = 0.0001 * nudge
            forward = round(min(duration, base + step), 4)
            backward = round(max(0.0, base - step), 4)
            if forward not in seen:
                t = forward
            elif backward not in seen:
                t = backward
        if t in seen:
            break
        seen.add(t)
        times.append(t)
        times = sorted(times)
        padded = True
    if len(times) > num_frames:
        idxs = np.linspace(0, len(times) - 1, num_frames)
        times = [times[int(round(i))] for i in idxs]
    return times[:num_frames], padded, "oracle"


def times_to_frame_indices(times: list[float], fps: float, total_frames: int) -> list[int]:
    fps = max(float(fps or 1.0), 1e-6)
    total_frames = max(int(total_frames), 1)
    indices = []
    for t in times:
        idx = int(round(t * fps))
        idx = max(0, min(total_frames - 1, idx))
        indices.append(idx)
    seen = set()
    fixed = []
    for idx in indices:
        cur = idx
        while cur in seen and cur < total_frames - 1:
            cur += 1
        while cur in seen and cur > 0:
            cur -= 1
        seen.add(cur)
        fixed.append(cur)
    return fixed


def extract_frames_rgb(video_path: Path, frame_indices: list[int]):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames = []
    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def load_oracle_pil_frames(
    video_path: Path,
    times: list[float],
    duration: float,
) -> tuple[list, list[int], dict]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or 1
    if duration <= 0:
        duration = total / fps
    cap.release()
    indices = times_to_frame_indices(times, fps, total)
    rgb = extract_frames_rgb(video_path, indices)
    meta = {
        "fps": fps,
        "total_frames": total,
        "duration": duration,
        "frame_indices": indices,
        "frame_times": times,
    }
    from .frame_io import frames_to_pil

    return frames_to_pil(rgb), indices, meta


PARQUET_BY_CONFIG = {
    "full": "herbench_full.parquet",
    "lite_v2": "herbench_lite_v2.parquet",
    "lite": "herbench_lite.parquet",
}
PACK_GAP_STEM = "uid_vid_00109.mp4"
EXPECTED_FULL_QUESTIONS = 27631
EXPECTED_FULL_MISSING = 65


def _rel_under_videos(rel: Path) -> Path:
    parts = rel.parts
    if parts and parts[0] == "videos":
        return Path(*parts[1:]) if len(parts) > 1 else Path()
    return rel


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def resolve_video(annotation_path: str, herbench_root: Path, videos_root: Path) -> Path | None:
    rel = Path(str(annotation_path))
    under = _rel_under_videos(rel)
    candidates = [
        herbench_root / rel,
        videos_root / under,
        videos_root / rel,
        videos_root / rel.name,
    ]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if _is_file(candidate):
            return candidate.resolve()
    return None


def _coerce_choices(raw) -> list[str]:
    if raw is None:
        items = []
    elif hasattr(raw, "tolist"):
        items = list(raw.tolist())
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = [raw]
    choices = [str(c).strip() for c in items if c is not None]
    while len(choices) < 5:
        choices.append("")
    return choices


def _row_from_item(item: dict, herbench_root: Path, videos_root: Path) -> dict:
    row = dict(item)
    row["question_id"] = str(row.get("question_id", ""))
    row["question"] = str(row.get("question") or "").strip()
    row["choices"] = _coerce_choices(row.get("choices"))
    row["answer"] = str(row.get("answer") or "").strip().upper()
    meta_raw = row.get("metadata_json") or row.get("metadata") or "{}"
    if isinstance(meta_raw, str):
        try:
            row["metadata"] = json.loads(meta_raw)
        except json.JSONDecodeError:
            row["metadata"] = {}
    else:
        row["metadata"] = meta_raw or {}
    resolved = resolve_video(row.get("video_path", ""), herbench_root, videos_root)
    row["resolved_video_path"] = str(resolved) if resolved else ""
    duration = float(row.get("duration") or row["metadata"].get("duration") or 0.0)
    row["duration"] = duration
    return row


def load_rows(
    herbench_root: Path,
    videos_root: Path,
    revision: str,
    config: str = "lite_v2",
) -> list[dict]:
    config = str(config or "lite_v2").strip() or "lite_v2"
    parquet_name = PARQUET_BY_CONFIG.get(config, f"herbench_{config}.parquet")
    local_parquet = Path(herbench_root) / "data" / parquet_name
    items: list[dict]
    if local_parquet.is_file():
        import pandas as pd

        print(f"Loading HERBench {config} from {local_parquet}", flush=True)
        frame = pd.read_parquet(local_parquet)
        items = frame.to_dict(orient="records")
    else:
        from datasets import load_dataset

        print(
            f"Loading HERBench {config} via HuggingFace "
            f"(revision={revision}, no local {local_parquet.name})",
            flush=True,
        )
        dataset = load_dataset(
            "DanBenAmi/HERBench",
            config,
            split="test",
            revision=revision,
        )
        items = [dict(item) for item in dataset]
    return [_row_from_item(item, herbench_root, videos_root) for item in items]


TSO_TASK = "Temporal Shot Ordering"
EXPECTED_TSO_N = 152


def is_tso_task(task_type: str) -> bool:
    t = str(task_type or "").strip().lower()
    return t in {"temporal shot ordering", "tso"} or "temporal shot ordering" in t
