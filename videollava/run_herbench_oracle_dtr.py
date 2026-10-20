#!/usr/bin/env python3
"""Paired Baseline vs DTR evaluation on HERBench lite_v2 oracle frames.

Designed to run inside /content/dtr_env (torch==2.0.1, transformers==4.31.0).
Do not import this from the Colab system kernel.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import random
import re
import shutil
import sys
import types
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# torchvision shim for pytorchvideo 0.1.5 (must run before videollava imports)
# ---------------------------------------------------------------------------
def patch_torchvision_functional_tensor() -> None:
    import torchvision.transforms as tv_transforms
    import torchvision.transforms.functional as tv_functional

    if "torchvision.transforms.functional_tensor" not in sys.modules:
        functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
        for name in dir(tv_functional):
            if not name.startswith("__"):
                setattr(functional_tensor, name, getattr(tv_functional, name))
        sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor
        tv_transforms.functional_tensor = functional_tensor


def assert_venv_packages(expected_prefix: str) -> dict:
    import torch as _torch
    import torchvision as _tv
    import transformers as _tf

    paths = {
        "torch": Path(_torch.__file__).resolve(),
        "torchvision": Path(_tv.__file__).resolve(),
        "transformers": Path(_tf.__file__).resolve(),
    }
    for name, path in paths.items():
        if expected_prefix and expected_prefix not in str(path):
            raise RuntimeError(
                f"{name} loaded from {path}, expected under {expected_prefix}"
            )
    return {
        "torch": _torch.__version__,
        "torchvision": _tv.__version__,
        "transformers": _tf.__version__,
        "paths": {k: str(v) for k, v in paths.items()},
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Timestamp / oracle frame helpers
# ---------------------------------------------------------------------------
_TIME_TOKEN = re.compile(
    r"(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:[.:](\d+))?|"
    r"(\d+(?:\.\d+)?)"
)


def _parse_time_token(token: str) -> float | None:
    token = token.strip()
    if not token or token.lower() == "all":
        return None
    # ranges like 00:05-00:08 handled by caller
    m = re.match(
        r"^(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:[.:](\d+))?$",
        token,
    )
    if m:
        hours = int(m.group(1) or 0)
        minutes = int(m.group(2))
        seconds = int(m.group(3))
        frac = m.group(4) or "0"
        # treat trailing digits as fractional seconds (variable precision)
        frac_sec = int(frac) / (10 ** len(frac)) if frac else 0.0
        return hours * 3600 + minutes * 60 + seconds + frac_sec
    try:
        return float(token)
    except ValueError:
        return None


def parse_span(span) -> tuple[float, float] | None:
    """Parse a timestamp span into (start_sec, end_sec)."""
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

    # Prefer explicit "start - end", then hyphen between two time-like tokens.
    if " - " in s:
        left, right = s.split(" - ", 1)
    else:
        m = re.match(
            r"^(\d[\d:.]*)\s*-\s*(\d[\d:.]*)$",
            s,
        )
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
    """Return evidence spans and mode hint from HERBench metadata_json."""
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
                    # nested option timestamps — use first-level items only if they are spans
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
    """Select exactly num_frames times (seconds). Returns times, padded, mode."""
    duration = max(float(duration or 1.0), 1e-3)

    if not spans:
        times = [duration * i / max(num_frames - 1, 1) for i in range(num_frames)]
        return times, False, "uniform_fallback"

    # Midpoints of each evidence span as primary candidates
    candidates = [max(0.0, min(duration, 0.5 * (a + b))) for a, b in spans]
    candidates = sorted(set(round(t, 4) for t in candidates))

    if len(candidates) >= num_frames:
        idxs = np.linspace(0, len(candidates) - 1, num_frames)
        times = [candidates[int(round(i))] for i in idxs]
        return times, False, "oracle"

    # Pad by denser sampling inside evidence spans
    times = list(candidates)
    padded = False
    span_i = 0
    while len(times) < num_frames and spans:
        a, b = spans[span_i % len(spans)]
        if b - a < 1e-3:
            # degenerate span: nudge neighbor samples
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

    while len(times) < num_frames:
        t = duration * len(times) / max(num_frames - 1, 1)
        times.append(round(t, 4))
        times = sorted(set(times))
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
    # ensure uniqueness while keeping length by shifting duplicates
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


def extract_frames_bgr(video_path: Path, frame_indices: list[int]) -> list[np.ndarray]:
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
            frames.append(frame)
    finally:
        cap.release()
    return frames


def write_temp_mp4(frames_bgr: list[np.ndarray], out_path: Path, fps: float = 1.0) -> None:
    if not frames_bgr:
        raise RuntimeError("No frames to write")
    h, w = frames_bgr[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")
    try:
        for frame in frames_bgr:
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
    finally:
        writer.release()


class OracleClipCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._meta: dict[str, dict] = {}

    def get_or_build(
        self,
        video_path: Path,
        video_id: str,
        times: list[float],
        duration: float,
    ) -> tuple[Path, list[int], dict]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or 1
        if duration <= 0:
            duration = total / fps
        cap.release()

        indices = times_to_frame_indices(times, fps, total)
        key = hashlib.sha1(
            f"{video_id}|{','.join(map(str, indices))}".encode()
        ).hexdigest()[:16]
        clip_path = self.cache_dir / f"{video_id}_{key}.mp4"
        meta = {
            "fps": fps,
            "total_frames": total,
            "duration": duration,
            "frame_indices": indices,
            "frame_times": times,
            "clip_path": str(clip_path),
        }
        if not clip_path.exists():
            frames = extract_frames_bgr(video_path, indices)
            write_temp_mp4(frames, clip_path, fps=1.0)
        self._meta[key] = meta
        return clip_path, indices, meta


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def resolve_video(annotation_path: str, herbench_root: Path, videos_root: Path) -> Path | None:
    rel = Path(str(annotation_path))
    candidates = [
        herbench_root / rel,
        videos_root / rel,
        videos_root / rel.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = list(videos_root.rglob(rel.name))
    return matches[0].resolve() if len(matches) == 1 else None


def load_rows(herbench_root: Path, videos_root: Path, revision: str) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(
        "DanBenAmi/HERBench",
        "lite_v2",
        split="test",
        revision=revision,
    )
    rows = []
    for item in dataset:
        row = dict(item)
        row["question_id"] = str(row.get("question_id", ""))
        row["question"] = str(row.get("question") or "").strip()
        raw_choices = row.get("choices") or []
        row["choices"] = [str(c).strip() for c in raw_choices if c is not None]
        while len(row["choices"]) < 5:
            row["choices"].append("")
        row["answer"] = str(row.get("answer") or "").strip().upper()
        meta_raw = row.get("metadata_json") or "{}"
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
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Model / paired inference
# ---------------------------------------------------------------------------
def set_attention_condition(model, dtr, vanilla_fn, dtr_fn, use_dtr: bool) -> None:
    modifier = dtr.FrameAttentionModifier
    if modifier._instance is None:
        modifier()
    modifier._instance.enabled = bool(use_dtr)
    for layer_idx in range(18, 32):
        attention = model.model.layers[layer_idx].self_attn
        if use_dtr:
            attention.use_frame_attn = True
            attention.layer_idx = layer_idx
            attention.forward = types.MethodType(dtr_fn, attention)
        else:
            attention.use_frame_attn = False
            attention.forward = types.MethodType(vanilla_fn, attention)


def generate_once(
    model,
    tokenizer,
    video_tensor,
    prompt: str,
    num_video_frames: int,
    dtr,
    vanilla_fn,
    dtr_fn,
    use_dtr: bool,
    max_new_tokens: int = 32,
) -> str:
    from videollava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from videollava.conversation import SeparatorStyle, conv_templates
    from videollava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token

    set_attention_condition(model, dtr, vanilla_fn, dtr_fn, use_dtr=use_dtr)

    conv = conv_templates["llava_v1"].copy()
    instruction = "".join([DEFAULT_IMAGE_TOKEN] * num_video_frames) + "\n" + prompt
    conv.append_message(conv.roles[0], instruction)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=video_tensor,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    text = tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
    if text.endswith(stop_str):
        text = text[: -len(stop_str)].strip()
    return text


def prepare_video_tensor(video_processor, clip_path: Path, model, seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pixels = video_processor(str(clip_path), return_tensors="pt")["pixel_values"]
    if isinstance(pixels, list):
        return [t.to(model.device, dtype=torch.float16) for t in pixels]
    return pixels.to(model.device, dtype=torch.float16)


FIELD_ORDER = [
    "run_fingerprint",
    "question_id",
    "video_id",
    "video_path",
    "task_type",
    "question",
    "choices_json",
    "answer",
    "answer_index",
    "num_frames",
    "oracle_mode",
    "oracle_padded",
    "oracle_frame_times",
    "oracle_frame_indices",
    "baseline_prediction",
    "dtr_prediction",
    "baseline_raw_output",
    "dtr_raw_output",
    "baseline_correct",
    "dtr_correct",
    "baseline_parse_ok",
    "dtr_parse_ok",
    "status",
    "error",
]


def append_record(path: Path, record: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_ORDER)
        if not exists:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in FIELD_ORDER})


def make_herbench_predictions(frame: pd.DataFrame, condition: str) -> dict:
    out = {}
    pred_col = f"{condition}_prediction"
    raw_col = f"{condition}_raw_output"
    for _, row in frame.iterrows():
        out[str(row["question_id"])] = {
            "predicted_choice": str(row[pred_col]),
            "is_correct": bool(row[f"{condition}_correct"]),
            "response": str(row[raw_col]),
            "frames_used": int(row["num_frames"]),
            "gt_answer": str(row["answer"]),
            "task_type": str(row["task_type"]),
        }
    return out


def summarize_pairs(pairs: pd.DataFrame) -> dict:
    def block(frame: pd.DataFrame) -> dict:
        n = len(frame)
        b_ok = frame["baseline_parse_ok"].astype(bool)
        d_ok = frame["dtr_parse_ok"].astype(bool)
        b_cor = frame["baseline_correct"].astype(bool)
        d_cor = frame["dtr_correct"].astype(bool)
        return {
            "n": int(n),
            "baseline_strict_acc": float(b_cor.mean()) if n else 0.0,
            "dtr_strict_acc": float(d_cor.mean()) if n else 0.0,
            "baseline_valid_acc": float(b_cor[b_ok].mean()) if b_ok.any() else 0.0,
            "dtr_valid_acc": float(d_cor[d_ok].mean()) if d_ok.any() else 0.0,
            "baseline_parse_errors": int((~b_ok).sum()),
            "dtr_parse_errors": int((~d_ok).sum()),
            "delta_strict": float(d_cor.mean() - b_cor.mean()) if n else 0.0,
        }

    overall = block(pairs)
    by_task = {
        task: block(g) for task, g in pairs.groupby("task_type", dropna=False)
    }
    by_oracle = {
        mode: block(g) for mode, g in pairs.groupby("oracle_mode", dropna=False)
    }
    return {"overall": overall, "by_task": by_task, "by_oracle_mode": by_oracle}


def run(args: argparse.Namespace) -> None:
    expected_prefix = str(Path(args.venv_site).resolve()) if args.venv_site else ""
    patch_torchvision_functional_tensor()
    versions = assert_venv_packages(expected_prefix)
    print("Package check OK:", json.dumps(versions, indent=2))

    root = Path(args.root)
    dtr_repo = Path(args.dtr_repo)
    herbench_root = Path(args.herbench_root)
    videos_root = Path(args.videos_root)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    persist_dir = Path(args.persist_dir) if args.persist_dir else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    persist_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "oracle_clips"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dtr_source = dtr_repo / "run_inference_dtr.py"
    if sha256_file(dtr_source) != args.dtr_sha256:
        raise RuntimeError(
            f"DTR source SHA256 mismatch: {sha256_file(dtr_source)} != {args.dtr_sha256}"
        )
    base_src = Path(args.herbench_base_vlm)
    acc_src = Path(args.herbench_accuracy)
    if sha256_file(base_src) != args.herbench_base_sha256:
        raise RuntimeError("HERBench base_vlm.py SHA256 mismatch")
    if sha256_file(acc_src) != args.herbench_acc_sha256:
        raise RuntimeError("HERBench calculate_accuracy.py SHA256 mismatch")

    sys.path.insert(0, str(dtr_repo.resolve()))
    import run_inference_dtr as dtr
    from transformers.models.llama.modeling_llama import LlamaAttention

    base_mod = load_module_from_file("herbench_base_pinned", base_src)
    acc_mod = load_module_from_file("herbench_accuracy_pinned", acc_src)

    rows = load_rows(herbench_root, videos_root, args.herbench_revision)
    missing = [r["question_id"] for r in rows if not r["resolved_video_path"]]
    if missing:
        raise RuntimeError(f"{len(missing)} questions missing videos; first={missing[:5]}")

    vanilla_fn = LlamaAttention.forward
    model, tokenizer, video_processor = dtr.load_model_with_global_rebalance(
        model_path=str(model_dir),
        device=0,
        start_layer=18,
        end_layer=31,
        alpha=0.5,
        beta=0.4,
    )
    model.eval()
    try:
        import decord  # noqa: F401
    except ImportError:
        video_processor.config.vision_config.video_decode_backend = "opencv"

    dtr_fn = dtr.llama_new_forward_with_global_rebalance
    set_attention_condition(model, dtr, vanilla_fn, dtr_fn, use_dtr=False)
    num_video_frames = int(model.get_video_tower().config.num_frames)
    if num_video_frames != 8:
        raise RuntimeError(f"Expected 8 native frames, found {num_video_frames}")

    def format_prompt(row: dict) -> str:
        return base_mod.BaseVLM._format_prompt(None, row["question"], row["choices"])

    def parse_choice(text: str) -> str:
        return base_mod.BaseVLM.extract_answer_choice(None, str(text))

    fingerprint_payload = {
        "protocol": "herbench_lite_v2_oracle8_baseline_dtr",
        "dataset_revision": args.herbench_revision,
        "dtr_commit": args.dtr_commit,
        "dtr_sha256": args.dtr_sha256,
        "model_revision": args.model_revision,
        "num_frames": num_video_frames,
        "alpha": 0.5,
        "beta": 0.4,
        "layers": [18, 31],
        "torch": versions["torch"],
        "transformers": versions["transformers"],
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    tag = f"oracle_{fingerprint}"
    pairs_csv = out_dir / f"pairs_{tag}.csv"
    persist_csv = persist_dir / pairs_csv.name
    summary_json = out_dir / f"summary_{tag}.json"
    manifest_json = out_dir / f"manifest_{tag}.json"

    if persist_csv.exists() and (
        not pairs_csv.exists() or persist_csv.stat().st_mtime > pairs_csv.stat().st_mtime
    ):
        shutil.copy2(persist_csv, pairs_csv)

    completed: set[str] = set()
    if pairs_csv.exists():
        prev = pd.read_csv(pairs_csv, on_bad_lines="error")
        if "run_fingerprint" not in prev.columns:
            raise RuntimeError("Resume CSV missing run_fingerprint")
        if not prev["run_fingerprint"].astype(str).eq(fingerprint).all():
            raise RuntimeError("Resume fingerprint mismatch; refusing to mix protocols")
        resumable = prev[prev["status"].astype(str) != "runtime_error"]
        completed = set(resumable["question_id"].astype(str))
        print(f"Resuming {len(completed)}/{len(rows)} completed pairs")

    pending = [r for r in rows if r["question_id"] not in completed]
    pending.sort(key=lambda r: (r["resolved_video_path"], r["question_id"]))

    if args.smoke:
        # two questions from distinct videos when possible
        smoke = []
        seen_v = set()
        for r in pending or rows:
            if r["resolved_video_path"] not in seen_v:
                smoke.append(r)
                seen_v.add(r["resolved_video_path"])
            if len(smoke) == 2:
                break
        if len(smoke) < 2:
            smoke = (pending or rows)[:2]
        pending = smoke
        print(f"SMOKE mode: {len(pending)} questions")

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in pending:
        groups[r["resolved_video_path"]].append(r)

    clip_cache = OracleClipCache(cache_dir)
    oracle_seed_base = 20260811

    manifest_json.write_text(
        json.dumps(
            {
                **fingerprint_payload,
                "run_fingerprint": fingerprint,
                "n_questions": len(rows),
                "n_pending": len(pending),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "package_versions": {
                    pkg: importlib.metadata.version(pkg)
                    for pkg in [
                        "torch",
                        "torchvision",
                        "transformers",
                        "tokenizers",
                        "accelerate",
                        "pytorchvideo",
                    ]
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    processed = 0
    for video_number, (video_path, video_rows) in enumerate(groups.items(), start=1):
        print(
            f"Video {video_number}/{len(groups)}: {Path(video_path).name} "
            f"({len(video_rows)} questions)"
        )
        for row in video_rows:
            record = {
                "run_fingerprint": fingerprint,
                "question_id": row["question_id"],
                "video_id": row.get("video_id", ""),
                "video_path": row.get("video_path", ""),
                "task_type": row.get("task_type", ""),
                "question": row["question"],
                "choices_json": json.dumps(row["choices"], ensure_ascii=False),
                "answer": row["answer"],
                "answer_index": row.get("answer_index", ""),
                "num_frames": num_video_frames,
                "oracle_mode": "",
                "oracle_padded": "",
                "oracle_frame_times": "",
                "oracle_frame_indices": "",
                "baseline_prediction": "",
                "dtr_prediction": "",
                "baseline_raw_output": "",
                "dtr_raw_output": "",
                "baseline_correct": "",
                "dtr_correct": "",
                "baseline_parse_ok": "",
                "dtr_parse_ok": "",
                "status": "runtime_error",
                "error": "",
            }
            video_tensor = None
            try:
                spans, mode_hint = extract_evidence_spans(row["metadata"])
                times, padded, mode = choose_oracle_times(
                    spans, row["duration"], num_frames=num_video_frames
                )
                if mode_hint == "uniform_fallback":
                    mode = "uniform_fallback"
                    padded = False
                clip_path, indices, meta = clip_cache.get_or_build(
                    Path(video_path),
                    str(row.get("video_id") or Path(video_path).stem),
                    times,
                    row["duration"],
                )
                seed = oracle_seed_base + int(
                    hashlib.sha256(row["question_id"].encode()).hexdigest()[:8], 16
                )
                video_tensor = prepare_video_tensor(
                    video_processor, clip_path, model, seed=seed
                )
                prompt = format_prompt(row)
                baseline_raw = generate_once(
                    model,
                    tokenizer,
                    video_tensor,
                    prompt,
                    num_video_frames,
                    dtr,
                    vanilla_fn,
                    dtr_fn,
                    use_dtr=False,
                )
                dtr_raw = generate_once(
                    model,
                    tokenizer,
                    video_tensor,
                    prompt,
                    num_video_frames,
                    dtr,
                    vanilla_fn,
                    dtr_fn,
                    use_dtr=True,
                )
                baseline_pred = parse_choice(baseline_raw)
                dtr_pred = parse_choice(dtr_raw)
                answer = row["answer"]
                record.update(
                    {
                        "oracle_mode": mode,
                        "oracle_padded": bool(padded),
                        "oracle_frame_times": json.dumps(times),
                        "oracle_frame_indices": json.dumps(indices),
                        "baseline_prediction": baseline_pred,
                        "dtr_prediction": dtr_pred,
                        "baseline_raw_output": baseline_raw,
                        "dtr_raw_output": dtr_raw,
                        "baseline_correct": baseline_pred == answer and baseline_pred != "ERROR",
                        "dtr_correct": dtr_pred == answer and dtr_pred != "ERROR",
                        "baseline_parse_ok": baseline_pred != "ERROR",
                        "dtr_parse_ok": dtr_pred != "ERROR",
                        "status": "ok",
                        "error": "",
                    }
                )
            except Exception as exc:
                record["error"] = repr(exc)
                record["status"] = "runtime_error"
                append_record(pairs_csv, record)
                shutil.copy2(pairs_csv, persist_csv)
                raise
            finally:
                if video_tensor is not None:
                    del video_tensor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            append_record(pairs_csv, record)
            processed += 1
            if processed % 10 == 0:
                shutil.copy2(pairs_csv, persist_csv)
                print(f"  checkpoint {processed} rows -> {persist_csv}")

        shutil.copy2(pairs_csv, persist_csv)

    pairs = pd.read_csv(pairs_csv)
    pairs = pairs[pairs["run_fingerprint"].astype(str) == fingerprint]
    pairs_ok = pairs[pairs["status"] == "ok"].copy()
    if args.smoke:
        smoke_path = out_dir / f"smoke_{tag}.json"
        smoke_path.write_text(
            pairs_ok.tail(len(pending)).to_json(orient="records", indent=2),
            encoding="utf-8",
        )
        if pairs_ok.empty:
            raise RuntimeError("Smoke produced no successful rows")
        if pairs_ok["baseline_prediction"].eq("ERROR").all():
            raise RuntimeError("All smoke baseline outputs failed parser")
        if pairs_ok["dtr_prediction"].eq("ERROR").all():
            raise RuntimeError("All smoke DTR outputs failed parser")
        print("SMOKE OK")
        print(pairs_ok[
            [
                "question_id",
                "oracle_mode",
                "baseline_prediction",
                "dtr_prediction",
                "answer",
            ]
        ].to_string(index=False))
        return

    summary = summarize_pairs(pairs_ok)
    baseline_preds = make_herbench_predictions(pairs_ok, "baseline")
    dtr_preds = make_herbench_predictions(pairs_ok, "dtr")
    baseline_official = acc_mod.calculate_accuracy(baseline_preds)
    dtr_official = acc_mod.calculate_accuracy(dtr_preds)
    payload = {
        "run_fingerprint": fingerprint,
        "manifest": fingerprint_payload,
        "summary": summary,
        "official_herbench_error_excluded": {
            "baseline": baseline_official,
            "dtr": dtr_official,
        },
        "n_ok": int(len(pairs_ok)),
        "n_runtime_error": int((pairs["status"] == "runtime_error").sum()),
        "pairs_csv": str(pairs_csv),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    shutil.copy2(summary_json, persist_dir / summary_json.name)
    shutil.copy2(pairs_csv, persist_csv)

    baseline_json = out_dir / f"predictions_baseline_{tag}.json"
    dtr_json = out_dir / f"predictions_dtr_{tag}.json"
    baseline_json.write_text(json.dumps(baseline_preds, indent=2), encoding="utf-8")
    dtr_json.write_text(json.dumps(dtr_preds, indent=2), encoding="utf-8")

    print("\n=== OVERALL ===")
    print(json.dumps(summary["overall"], indent=2))
    print("\n=== OFFICIAL HERBENCH (error-excluded) ===")
    print("baseline", baseline_official["overall"])
    print("dtr     ", dtr_official["overall"])
    print("\nSaved:", summary_json)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/content")
    p.add_argument("--venv-site", default="")
    p.add_argument("--dtr-repo", required=True)
    p.add_argument("--dtr-commit", required=True)
    p.add_argument("--dtr-sha256", required=True)
    p.add_argument("--herbench-root", required=True)
    p.add_argument("--videos-root", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--persist-dir", default="")
    p.add_argument("--herbench-revision", required=True)
    p.add_argument("--model-revision", required=True)
    p.add_argument("--herbench-base-vlm", required=True)
    p.add_argument("--herbench-accuracy", required=True)
    p.add_argument("--herbench-base-sha256", required=True)
    p.add_argument("--herbench-acc-sha256", required=True)
    p.add_argument("--smoke", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
