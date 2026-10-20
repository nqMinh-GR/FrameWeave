"""NExT-GQA frame protocols: uniform full-video and oracle evidence, K=8.

Do not reuse HERBench ``choose_oracle_times``: that path pads with frames
outside labelled spans. Fail the question instead.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .nextgqa import NUM_FRAMES

SPAN_EPS = 1e-4


def uniform_times(duration: float, k: int = NUM_FRAMES) -> list[float]:
    duration = max(float(duration or 0.0), 1e-6)
    k = int(k)
    if k < 1:
        raise ValueError(f"K must be >= 1, got {k}")
    return [duration * i / max(k - 1, 1) for i in range(k)]


def clip_spans(spans: Sequence[Sequence[float]], duration: float) -> list[tuple[float, float]]:
    duration = max(float(duration or 0.0), 0.0)
    out: list[tuple[float, float]] = []
    for span in spans or []:
        if not span or len(span) < 2:
            continue
        a, b = float(span[0]), float(span[1])
        if a > b:
            a, b = b, a
        a = max(0.0, min(duration, a)) if duration else max(0.0, a)
        b = max(0.0, min(duration, b)) if duration else max(0.0, b)
        if b < a:
            a, b = b, a
        out.append((a, b))
    out.sort(key=lambda s: (s[0], s[1]))
    return out


def largest_remainder_counts(durations: list[float], remaining: int) -> list[int]:
    n = len(durations)
    extra = [0] * n
    if remaining <= 0 or n == 0:
        return extra
    total = float(sum(max(0.0, d) for d in durations))
    if total <= SPAN_EPS:
        return extra
    quotas = [remaining * max(0.0, d) / total for d in durations]
    floors = [int(q) for q in quotas]
    leftover = remaining - sum(floors)
    order = sorted(range(n), key=lambda i: (-(quotas[i] - floors[i]), i))
    for i in order[: max(0, leftover)]:
        extra[i] = 1
    return [floors[i] + extra[i] for i in range(n)]


def allocate_oracle_counts(spans: list[tuple[float, float]], k: int) -> tuple[list[int], str]:
    k = int(k)
    n = len(spans)
    if n == 0:
        return [], "missing_spans"
    if n > k:
        return [], "too_many_spans"
    durations = [round(max(0.0, b - a), 6) for a, b in spans]
    extra = largest_remainder_counts(durations, k - n)
    counts = [1 + e for e in extra]
    if sum(counts) != k:
        leftover = k - sum(counts)
        if leftover > 0 and sum(durations) <= SPAN_EPS:
            return counts, "insufficient_distinct_frames"
        i = 0
        while leftover > 0 and durations:
            if durations[i % n] > SPAN_EPS:
                counts[i % n] += 1
                leftover -= 1
            i += 1
            if i > k * 20:
                return counts, "insufficient_distinct_frames"
    for (a, b), c in zip(spans, counts):
        if c > 1 and (b - a) <= SPAN_EPS:
            return counts, "insufficient_distinct_frames"
    return counts, ""


def times_in_span(a: float, b: float, n: int) -> list[float]:
    if n <= 0:
        return []
    if n == 1 or (b - a) <= SPAN_EPS:
        return [float(a)]
    return [a + (j + 0.5) / n * (b - a) for j in range(n)]


def oracle_times(
    spans: Sequence[Sequence[float]],
    duration: float,
    k: int = NUM_FRAMES,
) -> tuple[list[float], list[int], str]:
    clipped = clip_spans(spans, duration)
    counts, err = allocate_oracle_counts(clipped, k)
    if err:
        return [], [], err
    pairs: list[tuple[float, int]] = []
    for i, ((a, b), n) in enumerate(zip(clipped, counts)):
        for t in times_in_span(a, b, n):
            t = min(max(0.0, t), max(duration, b))
            pairs.append((t, i))
    pairs.sort(key=lambda p: (p[0], p[1]))
    if len(pairs) != int(k):
        return [], [], "insufficient_distinct_frames"
    times = [p[0] for p in pairs]
    assigned = [p[1] for p in pairs]
    return times, assigned, ""


def times_to_indices(times: Sequence[float], fps: float, total: int) -> list[int]:
    fps = max(float(fps or 1.0), 1e-6)
    total = max(int(total), 1)
    out = []
    for t in times:
        idx = int(round(float(t) * fps))
        out.append(max(0, min(total - 1, idx)))
    return out


def actual_time(index: int, fps: float) -> float:
    return float(index) / max(float(fps or 1.0), 1e-6)


def span_contains(t: float, a: float, b: float, fps: float) -> bool:
    tol = 0.5 / max(float(fps or 1.0), 1e-6)
    return (a - tol) <= t <= (b + tol)


def frame_hash(arr) -> str:
    import numpy as np

    data = np.ascontiguousarray(arr)
    return hashlib.sha256(data.tobytes()).hexdigest()[:16]


@dataclass
class DecodeResult:
    status: str
    error: str = ""
    frames: list = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    actual_times: list[float] = field(default_factory=list)
    assigned_spans: list[int] = field(default_factory=list)
    frame_hashes: list[str] = field(default_factory=list)
    fps: float = 0.0
    duration: float = 0.0
    total_frames: int = 0
    duplicate_indices: bool = False


def check_indices(
    indices: list[int],
    times: list[float],
    assigned: list[int],
    spans: list[tuple[float, float]],
    fps: float,
    mode: str,
) -> str:
    if len(indices) != len(times):
        return "insufficient_distinct_frames"
    if len(set(indices)) != len(indices):
        return "insufficient_distinct_frames"
    if mode != "oracle":
        return ""
    for idx, span_i in zip(indices, assigned):
        t_hat = actual_time(idx, fps)
        a, b = spans[span_i]
        if not span_contains(t_hat, a, b, fps):
            return "decode_outside_span"
    return ""


def decode_from_arrays(
    frames: Sequence,
    fps: float,
    times: Sequence[float],
    *,
    mode: str = "uniform",
    spans: Sequence[Sequence[float]] | None = None,
    assigned: Sequence[int] | None = None,
    duration: float | None = None,
) -> DecodeResult:
    import numpy as np

    total = len(frames)
    if total == 0:
        return DecodeResult(status="decode_failure", error="empty_video")
    duration = float(duration if duration is not None else (total / max(float(fps or 1.0), 1e-6)))
    clipped = clip_spans(spans or [], duration)
    idxs = times_to_indices(times, fps, total)
    err = check_indices(list(idxs), list(times), list(assigned or []), clipped, fps, mode)
    if err:
        return DecodeResult(
            status=err,
            error=err,
            timestamps=list(times),
            indices=list(idxs),
            assigned_spans=list(assigned or []),
            fps=float(fps),
            duration=duration,
            total_frames=total,
            duplicate_indices=len(set(idxs)) != len(idxs),
        )
    out_frames = [np.ascontiguousarray(frames[i]) for i in idxs]
    actual = [actual_time(i, fps) for i in idxs]
    hashes = [frame_hash(fr) for fr in out_frames]
    return DecodeResult(
        status="ok",
        frames=out_frames,
        timestamps=list(times),
        indices=list(idxs),
        actual_times=actual,
        assigned_spans=list(assigned or []),
        frame_hashes=hashes,
        fps=float(fps),
        duration=duration,
        total_frames=total,
    )


def _open_capture(video_path: Path):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or 1
    duration = total / fps if fps else 0.0
    return cap, fps, total, duration


def extract_rgb(video_path: Path, indices: list[int]) -> list:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    frames = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
            frames.append(np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()
    return frames


def plan_times(row: dict, mode: str, k: int = NUM_FRAMES) -> tuple[list[float], list[int], str, float]:
    duration = float(row.get("duration") or 0.0)
    if duration <= 0 and row.get("fps") and row.get("frame_count"):
        duration = float(row["frame_count"]) / max(float(row["fps"]), 1e-6)
    if mode == "uniform":
        if duration <= 0:
            duration = 1.0
        return uniform_times(duration, k), [], "", duration
    times, assigned, err = oracle_times(row.get("spans") or [], duration or 1.0, k)
    return times, assigned, err, duration


def decode_row(row: dict, mode: str, k: int = NUM_FRAMES) -> DecodeResult:
    path = Path(row.get("resolved_video_path") or "")
    if not path.is_file():
        return DecodeResult(status="missing_media", error="missing_video")
    try:
        cap, fps, total, cap_duration = _open_capture(path)
        cap.release()
    except Exception as exc:
        return DecodeResult(status="decode_failure", error=repr(exc))
    duration = float(row.get("duration") or 0.0) or cap_duration
    fps = float(row.get("fps") or 0.0) or fps
    row = dict(row)
    row["duration"] = duration
    row["fps"] = fps
    times, assigned, err, duration = plan_times(row, mode, k)
    if err:
        return DecodeResult(status=err, error=err, duration=duration, fps=fps, total_frames=total)
    idxs = times_to_indices(times, fps, total)
    check = check_indices(idxs, times, assigned, clip_spans(row.get("spans") or [], duration), fps, mode)
    if check:
        return DecodeResult(
            status=check,
            error=check,
            timestamps=times,
            indices=idxs,
            assigned_spans=assigned,
            fps=fps,
            duration=duration,
            total_frames=total,
            duplicate_indices=len(set(idxs)) != len(idxs),
        )
    try:
        rgb = extract_rgb(path, idxs)
    except Exception as exc:
        return DecodeResult(status="decode_failure", error=repr(exc), timestamps=times, indices=idxs)
    from .frame_io import frames_to_pil

    actual = [actual_time(i, fps) for i in idxs]
    hashes = [frame_hash(fr) for fr in rgb]
    return DecodeResult(
        status="ok",
        frames=frames_to_pil(rgb),
        timestamps=times,
        indices=idxs,
        actual_times=actual,
        assigned_spans=assigned,
        frame_hashes=hashes,
        fps=fps,
        duration=duration,
        total_frames=total,
    )
