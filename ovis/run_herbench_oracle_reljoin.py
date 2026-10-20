#!/usr/bin/env python3
"""HERBench oracle evaluation for FrameWeave Hybrid v6.0 (V5.1) on Ovis2.5-9B.

Default config is lite_v2. Pass --herbench-config full --skip-unresolved for the
Drive full pack (27,566 / 27,631; uid_vid_00109 absent from the official tar).

Oracle evidence frames are passed as K separate images (not Ovis video packing).
Comparison is Vanilla (hook bypass) vs Hybrid (joint). DTR is not supported.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_PERSIST_DISABLED = False


def best_effort_copy(src: Path, dst: Path, timeout: float = 8.0) -> bool:
    global _PERSIST_DISABLED
    src, dst = Path(src), Path(dst)
    if _PERSIST_DISABLED or not src.exists():
        return False
    try:
        if dst.exists() and src.resolve() == dst.resolve():
            return True
    except OSError:
        return False
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"WARN: persist mkdir failed: {exc!r}", flush=True)
        return False
    err: list[BaseException] = []

    def _copy() -> None:
        try:
            tmp = dst.with_name(dst.name + ".tmp")
            shutil.copy2(src, tmp)
            tmp.replace(dst)
        except BaseException as exc:
            err.append(exc)

    worker = threading.Thread(target=_copy, daemon=True)
    worker.start()
    worker.join(timeout=timeout)
    if worker.is_alive():
        print(
            f"WARN: persist copy timed out after {timeout}s ({src} -> {dst}).",
            flush=True,
        )
        return False
    if err:
        print(f"WARN: persist copy failed: {err[0]!r}", flush=True)
        return False
    return True


os.environ.pop("PYTHONPATH", None)
os.environ.pop("PYTHONHOME", None)

import pandas as pd
import torch

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from reljoin.config import (
    METHOD_TO_VARIANT,
    VALID_VARIANTS,
    architecture_variant,
    variant_to_method,
)

VARIANT_TO_METHOD = {name: variant_to_method(name) for name in VALID_VARIANTS}
VARIANT_TO_METHOD["RelJoin"] = "hybrid"

FIELD_ORDER = [
    "run_fingerprint",
    "sample_id",
    "question_id",
    "video_id",
    "video_path",
    "task_type",
    "method",
    "question",
    "option_order",
    "choices_json",
    "ground_truth",
    "answer",
    "parsed_prediction",
    "raw_output",
    "is_valid",
    "is_correct",
    "num_frames",
    "frame_timestamps",
    "oracle_mode",
    "oracle_padded",
    "oracle_frame_times",
    "oracle_frame_indices",
    "tokens_per_frame",
    "runtime_seconds",
    "status",
    "error_message",
    "error",
    "eval_mode",
]


def csv_fields(path: Path) -> list[str]:
    path = Path(path)
    if path.is_file() and path.stat().st_size > 0:
        with path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header:
            return list(header)
    return list(FIELD_ORDER)


def append_record(path: Path, record: dict) -> None:
    path = Path(path)
    fields = csv_fields(path)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in fields})
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return os.path.abspath(a) == os.path.abspath(b)


def save_record(local_csv: Path, persist_csv: Path, record: dict) -> None:
    """Append one finished row locally and on Drive so a dead runtime can resume."""
    append_record(local_csv, record)
    if _same_path(local_csv, persist_csv):
        return
    t0 = time.time()
    append_record(persist_csv, record)
    dt = time.time() - t0
    if dt >= 2.0:
        print(
            f"  persist {record.get('question_id')} {record.get('method')} {dt:.1f}s",
            flush=True,
        )


def pending_methods(
    row: dict, methods: list[str], completed: set[tuple[str, str]]
) -> list[str]:
    qid = str(row["question_id"])
    return [method for method in methods if (qid, method) not in completed]


def stage_one_video(src: Path, stage_dir: Path) -> Path:
    """Copy one mp4 onto local disk. Caller deletes it after the video finishes."""
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(src)
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    for old in list(stage_dir.iterdir()):
        if old.is_file() or old.is_symlink():
            old.unlink()
    dst = stage_dir / "current.mp4"
    total = src.stat().st_size
    free = shutil.disk_usage(stage_dir).free
    margin = 512 * 1024 * 1024
    if free < total + margin:
        raise RuntimeError(
            f"Not enough local disk for {src.name}: file={total} bytes, free={free} bytes."
        )
    print(f"Staging {src} ({total} bytes) -> {dst}", flush=True)
    copied = 0
    next_report = 0
    t0 = time.time()
    try:
        with src.open("rb") as rf, dst.open("wb") as wf:
            while True:
                buf = rf.read(8 * 1024 * 1024)
                if not buf:
                    break
                wf.write(buf)
                copied += len(buf)
                if copied >= next_report or copied == total:
                    print(
                        f"  stage {copied}/{total} bytes ({time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    next_report = copied + 64 * 1024 * 1024
            wf.flush()
            os.fsync(wf.fileno())
    except Exception:
        if dst.exists():
            dst.unlink()
        raise
    got = dst.stat().st_size
    if got != total:
        dst.unlink()
        raise OSError(f"Short copy of {src.name}: {got} != {total}")
    return dst


def row_task_type(row: dict) -> str:
    for key in ("task_type", "task", "category", "question_type"):
        v = row.get(key)
        if v:
            return str(v)
    meta = row.get("metadata") or {}
    if isinstance(meta, dict):
        for key in ("task_type", "task", "category", "question_type"):
            if meta.get(key):
                return str(meta[key])
    return ""


def per_task_accuracy(df: pd.DataFrame) -> dict:
    out: dict[str, dict] = {}
    if df.empty or "method" not in df.columns:
        return out
    for method, g in df.groupby("method"):
        by_task = {}
        if "task_type" in g.columns:
            for task, t in g.groupby("task_type"):
                n = len(t)
                by_task[str(task)] = {
                    "n": int(n),
                    "strict_accuracy": float(t["is_correct"].mean()) if n else 0.0,
                }
        out[str(method)] = by_task
    return out


def discover_methods_from_ckpt_dir(ckpt_dir: Path) -> tuple[list[str], dict[str, Path]]:
    ckpt_dir = Path(ckpt_dir)
    found: dict[str, Path] = {}
    methods = ["vanilla"]
    if not ckpt_dir.is_dir():
        return methods, found
    for path in sorted(ckpt_dir.glob("*.pt")):
        stem = architecture_variant(path.stem)
        if stem not in VALID_VARIANTS:
            continue
        method = variant_to_method(stem)
        found[method] = path
        if method not in methods:
            methods.append(method)
    return methods, found


def find_extract_root(root: Path) -> Path:
    extract = Path(root) / "outputs_reljoin_v6_ovis"
    if extract.is_dir():
        return extract
    raise FileNotFoundError(
        f"Missing {extract}. Put folder outputs_reljoin_v6_ovis at {Path(root) / 'outputs_reljoin_v6_ovis'}."
    )


def find_reljoin_pt(extract: Path) -> Path:
    extract = Path(extract)
    names = ("Hybrid.pt", "GlobalTimeFix.pt", "RelJoin.pt")
    preferred = []
    for name in names:
        preferred.extend(
            [
                extract / "checkpoints" / "best_external_val" / name,
                extract / "checkpoints" / "last" / name,
                extract / "checkpoints" / name,
                extract / "best_external_val" / name,
                extract / "last" / name,
                extract / name,
            ]
        )
    for path in preferred:
        if path.is_file():
            return path
    hits = sorted(p for p in extract.rglob("*.pt") if p.is_file()) if extract.is_dir() else []
    if hits:
        print("Found FrameWeave ckpt at", hits[0])
        return hits[0]
    listing = []
    if extract.is_dir():
        listing = [p.name + ("/" if p.is_dir() else "") for p in sorted(extract.iterdir())]
        pts = [str(p.relative_to(extract)) for p in extract.rglob("*.pt")]
    else:
        pts = []
    raise FileNotFoundError(
        f"No Hybrid.pt / GlobalTimeFix.pt under {extract}.\n"
        f"Top-level: {listing or 'MISSING DIR'}\n"
        f"*.pt: {pts or 'none'}"
    )


def stage_ckpt_dir(extract_dir: Path) -> Path:
    reljoin_pt = find_reljoin_pt(extract_dir)
    ckpt_dir = reljoin_pt.parent
    last = Path(extract_dir) / "checkpoints" / "last"
    if ckpt_dir.resolve() != last.resolve() and last.is_dir():
        for name in ("Hybrid.pt", "GlobalTimeFix.pt", "RelJoin.pt"):
            if not (ckpt_dir / name).is_file() and (last / name).is_file():
                shutil.copy2(last / name, ckpt_dir / name)
                print(f"missing {name} → copy from last/")
    print("Using FrameWeave ckpt dir", ckpt_dir)
    return ckpt_dir


def load_reljoin_module(ckpt_path: Path, llm_dim: int, device, variant: str | None = None):
    from reljoin.config import METHOD_VERSION
    from reljoin.reljoin_modules import build_joining_module

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unexpected FrameWeave ckpt payload in {ckpt_path}: {type(ckpt)}")
    version = str(ckpt.get("method_version") or "")
    if version and not version.startswith("6."):
        raise RuntimeError(
            f"Refusing checkpoint method_version={version!r} from {ckpt_path}. "
            f"v6 eval requires method_version {METHOD_VERSION}."
        )
    state = ckpt.get("reljoin", ckpt)
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected RelJoin state in {ckpt_path}: {type(state)}")
    ckpt_variant = architecture_variant(variant or ckpt.get("variant") or Path(ckpt_path).stem)
    if "global_relation.W_Q.weight" not in state:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} is not FrameWeave v6 (missing global_relation.W_Q). "
            "Do not load v3/v4/v5 weights."
        )
    local_enabled = ckpt.get("local_enabled")
    module = build_joining_module(
        llm_dim,
        variant=ckpt_variant,
        local_enabled=local_enabled,
    ).to(device=device, dtype=torch.float32)
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"FrameWeave v6 state_dict mismatch from {ckpt_path}: {exc}"
        ) from exc
    print(
        f"Loaded FrameWeave v6 ckpt {ckpt_path} variant={ckpt_variant} "
        f"method_version={version or METHOD_VERSION}",
        flush=True,
    )
    module.eval()
    return module


from reljoin.ovis_generate import format_mc_prompt, generate_ovis


def run(args: argparse.Namespace) -> None:
    from reljoin.config import HERBENCH_ORACLE_FRAMES, MAX_PIXELS, MIN_PIXELS, METHOD_VERSION, MODEL_ID
    from reljoin.diagnostics import accuracy_reports, answer_position_bias, parse_choice_strict
    from reljoin.herbench_oracle import (
        EXPECTED_FULL_MISSING,
        EXPECTED_FULL_QUESTIONS,
        EXPECTED_TSO_N,
        PACK_GAP_STEM,
        choose_oracle_times,
        extract_evidence_spans,
        is_tso_task,
        load_module_from_file,
        load_oracle_pil_frames,
        load_rows,
        sha256_file,
    )
    from reljoin.ovis_hooks import (
        encode_question_hidden,
        freeze_backbone,
        get_hook_state,
        install_reljoin_hook,
        ovis_hidden_size,
    )
    from reljoin.stats import compare_methods, save_statistical_tests
    from reljoin.train import load_ovis
    from reljoin.config import ExperimentConfig

    root = Path(args.root)
    herbench_root = Path(args.herbench_root)
    videos_root = Path(args.videos_root)
    out_dir = Path(args.out_dir)
    persist_dir = Path(args.persist_dir) if args.persist_dir else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    persist_dir.mkdir(parents=True, exist_ok=True)
    stage_dir_arg = str(getattr(args, "stage_video_dir", "") or "")
    stage_dir = Path(stage_dir_arg) if stage_dir_arg else None
    if stage_dir is not None:
        stage_dir.mkdir(parents=True, exist_ok=True)
        for old in list(stage_dir.iterdir()):
            if old.is_file() or old.is_symlink():
                old.unlink()
                print(f"Removed leftover stage file {old}", flush=True)
        print(f"Staging one video at a time in {stage_dir}", flush=True)

    base_src = Path(args.herbench_base_vlm)
    acc_src = Path(args.herbench_accuracy)
    if base_src.is_file() and sha256_file(base_src) != args.herbench_base_sha256:
        raise RuntimeError("HERBench base_vlm.py SHA256 mismatch")
    if acc_src.is_file() and sha256_file(acc_src) != args.herbench_acc_sha256:
        raise RuntimeError("HERBench calculate_accuracy.py SHA256 mismatch")

    base_mod = load_module_from_file("herbench_base_pinned", base_src) if base_src.is_file() else None
    acc_mod = load_module_from_file("herbench_accuracy_pinned", acc_src) if acc_src.is_file() else None

    herbench_config = str(getattr(args, "herbench_config", "lite_v2") or "lite_v2").strip() or "lite_v2"
    skip_unresolved = bool(getattr(args, "skip_unresolved", False))
    print(
        f"Loading HERBench {herbench_config} rows ({args.herbench_revision}) ...",
        flush=True,
    )
    rows = load_rows(
        herbench_root, videos_root, args.herbench_revision, config=herbench_config
    )
    tso_only = bool(getattr(args, "tso_only", False))
    if tso_only:
        n_all = len(rows)
        rows = [r for r in rows if is_tso_task(row_task_type(r))]
        for r in rows:
            if not r.get("task_type"):
                r["task_type"] = "Temporal Shot Ordering"
        print(
            f"TSO-only: {len(rows)}/{n_all} questions (expect {EXPECTED_TSO_N})",
            flush=True,
        )
        if not rows:
            raise RuntimeError("TSO-only filter matched 0 questions.")
        if not args.smoke and herbench_config != "full" and len(rows) != EXPECTED_TSO_N:
            print(
                f"WARN: TSO n={len(rows)} != {EXPECTED_TSO_N}. "
                "Check HERBench revision / task_type labels.",
                flush=True,
            )
    missing_rows = [r for r in rows if not r["resolved_video_path"]]
    missing = [r["question_id"] for r in missing_rows]
    resolved_rows = [r for r in rows if r["resolved_video_path"]]
    pack_gap_stems = sorted(
        {Path(str(r.get("video_path") or "")).name for r in missing_rows}
    )
    print(
        f"Loaded {len(rows)} questions, {len(resolved_rows)} with local mp4, "
        f"{len(missing)} missing video",
        flush=True,
    )

    def _raise_missing_videos() -> None:
        raise RuntimeError(
            f"{len(missing)} questions missing videos; first={missing[:5]}. "
            f"--videos-root={videos_root}."
        )

    if missing and not args.smoke:
        if skip_unresolved:
            print(
                f"Skipping {len(missing)} unresolved questions; stems={pack_gap_stems[:10]}",
                flush=True,
            )
            if herbench_config == "full" and not tso_only:
                if PACK_GAP_STEM not in pack_gap_stems or len(missing) != EXPECTED_FULL_MISSING:
                    print(
                        f"WARN: full pack-gap expected {EXPECTED_FULL_MISSING} of "
                        f"{PACK_GAP_STEM}, got n={len(missing)} stems={pack_gap_stems}",
                        flush=True,
                    )
                if len(rows) != EXPECTED_FULL_QUESTIONS:
                    print(
                        f"WARN: full n={len(rows)} != {EXPECTED_FULL_QUESTIONS}",
                        flush=True,
                    )
        else:
            _raise_missing_videos()
    if args.smoke and not resolved_rows:
        _raise_missing_videos()

    ckpt_root = Path(args.reljoin_ckpt)
    if ckpt_root.suffix == ".pt":
        ckpt_root = ckpt_root.parent
    discovered, ckpt_by_method = discover_methods_from_ckpt_dir(ckpt_root)
    known_methods = {"vanilla", *METHOD_TO_VARIANT.keys(), *VARIANT_TO_METHOD.values()}
    if args.methods.strip():
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        methods = discovered
    methods = ["hybrid" if m == "reljoin" else m for m in methods]
    if "vanilla" not in methods:
        methods = ["vanilla"] + [m for m in methods if m != "vanilla"]
    if "dtr" in methods:
        raise ValueError("DTR is not supported on Ovis2.5. Use vanilla and FrameWeave variants.")
    unknown = [m for m in methods if m not in known_methods]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}. Known: {sorted(known_methods)}")
    needed = [m for m in methods if m != "vanilla"]
    if "hybrid_localoff" in needed and "hybrid_localoff" not in ckpt_by_method:
        if "global_timefix" in ckpt_by_method:
            ckpt_by_method["hybrid_localoff"] = ckpt_by_method["global_timefix"]
    missing_ckpt = [m for m in needed if m not in ckpt_by_method]
    if missing_ckpt:
        scanned = sorted(p.name for p in ckpt_root.glob("*.pt")) if ckpt_root.exists() else []
        raise FileNotFoundError(
            f"Missing checkpoints for {missing_ckpt} under {ckpt_root}. Found: {scanned or 'NO DIR'}"
        )
    print("Eval methods:", methods, flush=True)

    cfg = ExperimentConfig(run_mode="eval_herbench", root=root)
    if args.model_dir:
        cfg.model_dir = Path(args.model_dir)
    print(
        f"Loading {MODEL_ID} — several minutes of silence is expected.",
        flush=True,
    )
    model, tokenizer = load_ovis(cfg)
    print("Ovis2.5-9B loaded.", flush=True)
    model.eval()
    freeze_backbone(model)
    install_reljoin_hook(model)
    dtype = model.dtype
    llm_dim = ovis_hidden_size(model)
    joining_modules = {}
    for method in needed:
        joining_modules[method] = load_reljoin_module(
            ckpt_by_method[method],
            llm_dim,
            model.device,
            variant=METHOD_TO_VARIANT.get(method, method),
        )

    def format_prompt(row: dict) -> str:
        if base_mod is not None:
            return base_mod.BaseVLM._format_prompt(None, row["question"], row["choices"])
        return format_mc_prompt(row["question"], row["choices"])

    def parse_choice(text: str) -> str:
        if base_mod is not None:
            return base_mod.BaseVLM.extract_answer_choice(None, str(text))
        pred, ok = parse_choice_strict(text)
        return pred if ok else "ERROR"

    num_frames = int(getattr(args, "num_frames", HERBENCH_ORACLE_FRAMES) or HERBENCH_ORACLE_FRAMES)
    max_pixels = int(getattr(args, "max_pixels", MAX_PIXELS) or MAX_PIXELS)
    min_pixels = int(getattr(args, "min_pixels", MIN_PIXELS) or MIN_PIXELS)

    cfg_tag = "full" if herbench_config == "full" else "lite_v2"
    protocol = f"herbench_{cfg_tag}_oracle_frameweave_v6_ovis25"
    if tso_only:
        protocol = f"{protocol}_tso152"
    fingerprint_payload = {
        "protocol": protocol,
        "herbench_config": herbench_config,
        "skip_unresolved": skip_unresolved,
        "dataset_revision": args.herbench_revision,
        "model_id": MODEL_ID,
        "model_revision": args.model_revision,
        "num_frames": num_frames,
        "max_pixels": max_pixels,
        "methods": methods,
        "reljoin_ckpt": args.reljoin_ckpt,
        "method": "FrameWeave-Hybrid",
        "method_version": METHOD_VERSION,
        "formulation": "5.1.0",
        "backbone": "Ovis2.5-9B",
    }
    if tso_only:
        fingerprint_payload["tso_only"] = True
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    tag = f"reljoin_{fingerprint}"
    pairs_csv = out_dir / f"predictions_{tag}.csv"
    persist_csv = persist_dir / pairs_csv.name

    if persist_csv.exists() and (
        not pairs_csv.exists() or persist_csv.stat().st_mtime > pairs_csv.stat().st_mtime
    ):
        if best_effort_copy(persist_csv, pairs_csv):
            print(f"Restored resume CSV from persist: {persist_csv}", flush=True)

    completed: set[tuple[str, str]] = set()
    if pairs_csv.exists():
        prev = pd.read_csv(pairs_csv, on_bad_lines="error")
        for _, r in prev.iterrows():
            key = (str(r["question_id"]), str(r["method"]))
            if str(r.get("status")) == "runtime_error":
                completed.discard(key)
            else:
                completed.add(key)
        print(
            f"Resuming {len(completed)} completed (question, method) pairs from {pairs_csv}",
            flush=True,
        )

    pending = resolved_rows if skip_unresolved else rows
    if args.smoke:
        smoke = []
        seen_v = set()
        for r in resolved_rows:
            if r["resolved_video_path"] not in seen_v:
                smoke.append(r)
                seen_v.add(r["resolved_video_path"])
            if len(smoke) == 2:
                break
        if len(smoke) < 2:
            smoke = resolved_rows[:2]
        pending = smoke
        print(f"SMOKE mode: {len(pending)} questions × {len(methods)} methods", flush=True)

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in pending:
        vpath = r.get("resolved_video_path") or ""
        if not vpath:
            continue
        groups[vpath].append(r)

    n_left = 0
    n_done_videos = 0
    for video_rows in groups.values():
        left = sum(len(pending_methods(row, methods, completed)) for row in video_rows)
        n_left += left
        if left == 0:
            n_done_videos += 1
    print(
        f"Resume: {n_done_videos}/{len(groups)} videos already done, "
        f"{n_left} (question, method) pairs left",
        flush=True,
    )

    for video_number, (video_path, video_rows) in enumerate(groups.items(), start=1):
        video_name = Path(video_path).name
        left = sum(len(pending_methods(row, methods, completed)) for row in video_rows)
        if left == 0:
            print(f"Video {video_number}/{len(groups)}: {video_name} skip", flush=True)
            continue
        print(
            f"Video {video_number}/{len(groups)}: {video_name} pending={left}",
            flush=True,
        )
        staged: Path | None = None
        read_path = Path(video_path)
        try:
            if stage_dir is not None:
                read_path = stage_one_video(Path(video_path), stage_dir)
                staged = read_path
            for row in video_rows:
                todo = pending_methods(row, methods, completed)
                if not todo:
                    continue
                spans, mode_hint = extract_evidence_spans(row["metadata"])
                times, padded, mode = choose_oracle_times(
                    spans, row["duration"], num_frames=num_frames
                )
                if mode_hint == "uniform_fallback":
                    mode = "uniform_fallback"
                    padded = False
                frames, indices, _meta = load_oracle_pil_frames(
                    read_path, times, row["duration"]
                )
                prompt = format_prompt(row)
                answer = row["answer"]
                option_order = list("ABCDE"[: len([c for c in row["choices"] if c])])
                q_last = encode_question_hidden(
                    model,
                    tokenizer,
                    row["question"],
                    row["choices"],
                    model.device,
                    dtype,
                )
                timestamps = torch.tensor([times], device=model.device, dtype=dtype)
                frame_mask = torch.ones(1, len(frames), device=model.device, dtype=torch.bool)

                def _parse_pred(raw: str) -> tuple[str, bool]:
                    pred = parse_choice(raw)
                    is_valid = pred != "ERROR"
                    if not is_valid:
                        pred2, ok = parse_choice_strict(raw)
                        pred = pred2
                        is_valid = ok
                    return pred, is_valid

                for method in todo:
                    key = (str(row["question_id"]), method)
                    t0 = time.time()
                    record = {
                        "run_fingerprint": fingerprint,
                        "sample_id": row["question_id"],
                        "question_id": row["question_id"],
                        "video_id": row.get("video_id", ""),
                        "video_path": row.get("video_path", ""),
                        "task_type": row.get("task_type", ""),
                        "method": method,
                        "question": row["question"],
                        "option_order": json.dumps(option_order),
                        "choices_json": json.dumps(row["choices"], ensure_ascii=False),
                        "ground_truth": answer,
                        "answer": answer,
                        "num_frames": len(frames),
                        "frame_timestamps": json.dumps(times),
                        "oracle_mode": mode,
                        "oracle_padded": bool(padded),
                        "oracle_frame_times": json.dumps(times),
                        "oracle_frame_indices": json.dumps(indices),
                        "status": "runtime_error",
                        "error_message": "",
                        "error": "",
                        "eval_mode": "bypass" if method == "vanilla" else "full",
                    }
                    try:
                        if method == "vanilla":
                            raw = generate_ovis(
                                model,
                                frames,
                                prompt,
                                None,
                                q_last,
                                timestamps,
                                frame_mask,
                                bypass=True,
                                max_pixels=max_pixels,
                                min_pixels=min_pixels,
                            )
                        else:
                            raw = generate_ovis(
                                model,
                                frames,
                                prompt,
                                joining_modules[method],
                                q_last,
                                timestamps,
                                frame_mask,
                                bypass=False,
                                max_pixels=max_pixels,
                                min_pixels=min_pixels,
                            )
                        pred, is_valid = _parse_pred(raw)
                        is_correct = bool(is_valid and pred == answer)
                        record.update(
                            {
                                "parsed_prediction": pred,
                                "raw_output": raw,
                                "is_valid": is_valid,
                                "is_correct": is_correct,
                                "tokens_per_frame": json.dumps(get_hook_state().last_tokens_per_frame),
                                "runtime_seconds": time.time() - t0,
                                "status": "ok",
                            }
                        )
                        print(
                            f"  {method} {row['question_id']} pred={pred} gt={answer} "
                            f"ok={is_correct} {time.time() - t0:.1f}s "
                            f"P_i={get_hook_state().last_tokens_per_frame}",
                            flush=True,
                        )
                    except Exception as exc:
                        record["error_message"] = repr(exc)
                        record["error"] = repr(exc)
                        record["status"] = "runtime_error"
                        save_record(pairs_csv, persist_csv, record)
                        raise
                    save_record(pairs_csv, persist_csv, record)
                    completed.add(key)
        finally:
            if staged is not None and staged.exists():
                staged.unlink()
                print(f"Removed staged {video_name}", flush=True)

        gc.collect()
        torch.cuda.empty_cache()

    df = pd.read_csv(pairs_csv)
    if {"question_id", "method"}.issubset(df.columns):
        before = len(df)
        df = df.drop_duplicates(subset=["question_id", "method"], keep="last")
        if len(df) != before:
            print(f"Dropped {before - len(df)} duplicate CSV rows (kept last).", flush=True)
    if "is_correct" in df.columns:
        df["is_correct"] = df["is_correct"].astype(str).str.lower().isin(["true", "1", "yes"])
    if "is_valid" in df.columns:
        df["is_valid"] = df["is_valid"].astype(str).str.lower().isin(["true", "1", "yes"])
    df_ok = df[df["status"] == "ok"].copy()
    csv_methods = (
        [str(m) for m in df_ok["method"].astype(str).tolist()] if not df_ok.empty else []
    )
    csv_unique = list(dict.fromkeys(csv_methods))
    present = [m for m in methods if m in csv_unique]
    for m in csv_unique:
        if m not in present:
            present.append(m)

    summary = {}
    for method in present:
        sub = df_ok[df_ok["method"].astype(str) == method]
        rows_m = sub.to_dict(orient="records")
        summary[method] = accuracy_reports(rows_m)
        preds = {}
        for _, r in sub.iterrows():
            preds[str(r["question_id"])] = {
                "predicted_choice": str(r["parsed_prediction"]),
                "is_correct": bool(r["is_correct"]),
                "response": str(r["raw_output"]),
                "frames_used": int(r["num_frames"]),
                "gt_answer": str(r["ground_truth"]),
                "task_type": str(r["task_type"]),
            }
        official = acc_mod.calculate_accuracy(preds) if acc_mod is not None and preds else {}
        summary[method]["official"] = official
        sub.to_csv(out_dir / f"herbench_{method}.csv", index=False)

    by_task = per_task_accuracy(df_ok)
    for method, tasks in by_task.items():
        if method in summary:
            summary[method]["by_task_type"] = tasks

    stats_payload = {}
    if "vanilla" in present:
        base_rows = df_ok[df_ok["method"].astype(str) == "vanilla"].to_dict(orient="records")
        for method in present:
            if method == "vanilla":
                continue
            other = df_ok[df_ok["method"].astype(str) == method].to_dict(orient="records")
            if other:
                stats_payload[f"vanilla_vs_{method}"] = compare_methods(
                    base_rows, other, id_key="question_id", correct_key="is_correct"
                )
    save_statistical_tests(stats_payload, out_dir / "statistical_tests.json")

    payload = {
        "run_fingerprint": fingerprint,
        "manifest": fingerprint_payload,
        "herbench_config": herbench_config,
        "skip_unresolved": skip_unresolved,
        "n_questions": int(len(rows)),
        "n_resolved": int(len(resolved_rows)),
        "n_missing": int(len(missing)),
        "pack_gap_stems": pack_gap_stems,
        "tso_only": tso_only,
        "summary": summary,
        "stats": stats_payload,
        "n_ok": int(len(df_ok)),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "pairs_csv": str(pairs_csv),
    }
    summary_path = out_dir / f"summary_{tag}.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    bias = {}
    for method in present:
        sub = df_ok[df_ok["method"].astype(str) == method]
        bias[method] = answer_position_bias(sub.to_dict(orient="records"))
    bias_path = out_dir / "answer_position_bias.json"
    bias_path.write_text(json.dumps(bias, indent=2), encoding="utf-8")
    best_effort_copy(pairs_csv, persist_csv, timeout=180.0)
    best_effort_copy(summary_path, persist_dir / summary_path.name)
    best_effort_copy(out_dir / "statistical_tests.json", persist_dir / "statistical_tests.json")
    best_effort_copy(bias_path, persist_dir / bias_path.name)

    print(json.dumps(summary, indent=2), flush=True)
    if tso_only:
        print(f"TSO-only n={len(rows)}", flush=True)
    print("SMOKE OK" if args.smoke else "FULL OK", flush=True)


MAX_PIXELS_DEFAULT = 448 * 448
MIN_PIXELS_DEFAULT = 448 * 448


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/content")
    p.add_argument("--herbench-root", required=True)
    p.add_argument("--videos-root", required=True)
    p.add_argument("--model-dir", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--persist-dir", default="")
    p.add_argument("--herbench-revision", required=True)
    p.add_argument("--model-revision", required=True)
    p.add_argument("--herbench-base-vlm", default="")
    p.add_argument("--herbench-accuracy", default="")
    p.add_argument("--herbench-base-sha256", default="")
    p.add_argument("--herbench-acc-sha256", default="")
    p.add_argument("--methods", default="vanilla,hybrid")
    p.add_argument(
        "--reljoin-ckpt",
        default="/content/outputs_reljoin_v6_ovis/checkpoints/best_external_val",
    )
    p.add_argument("--max-pixels", type=int, default=MAX_PIXELS_DEFAULT)
    p.add_argument("--min-pixels", type=int, default=MIN_PIXELS_DEFAULT)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--tso-only",
        dest="tso_only",
        action="store_true",
        help="Evaluate only Temporal Shot Ordering. Separate CSV fingerprint.",
    )
    p.add_argument(
        "--herbench-config",
        default="lite_v2",
        help="HuggingFace / parquet config: lite_v2 (default) or full.",
    )
    p.add_argument(
        "--skip-unresolved",
        action="store_true",
        help="Skip questions whose mp4 is missing (full pack gap: uid_vid_00109).",
    )
    p.add_argument(
        "--stage-video-dir",
        default="",
        help="Copy one mp4 here, score it, then delete it. Empty reads videos in place.",
    )
    return p


if __name__ == "__main__":
    try:
        run(build_parser().parse_args())
    except Exception:
        import traceback

        traceback.print_exc()
        os._exit(1)
    os._exit(0)
