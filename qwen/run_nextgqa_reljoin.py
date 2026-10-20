#!/usr/bin/env python3
"""NExT-GQA evaluation for FrameWeave Hybrid v6.0 (V5.1) on Qwen2.5-VL-7B-Instruct.

Two protocols share K frames: oracle evidence (primary) and uniform full video.
Vanilla (hook bypass) and Hybrid read the same decoded frames. Evidence spans
never enter the prompt. Oracle accuracy is not Acc@GQA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_PERSIST_DISABLED = False
_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

os.environ.pop("PYTHONPATH", None)
os.environ.pop("PYTHONHOME", None)

BACKBONE = "Qwen2.5-VL-7B-Instruct"
OUTPUT_TREE = "outputs_reljoin_v6_qwen"
DEFAULT_CKPT = f"/content/{OUTPUT_TREE}/checkpoints/best_external_val"
DEFAULT_MODEL_DIR = "/content/checkpoints/qwen25vl_7b"
MAX_PIXELS_DEFAULT = 448 * 448
MIN_PIXELS_DEFAULT = 448 * 448


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
        print(f"WARN: persist copy timed out after {timeout}s ({src} -> {dst}).", flush=True)
        return False
    if err:
        print(f"WARN: persist copy failed: {err[0]!r}", flush=True)
        return False
    return True


def _exit_process(code: int = 0) -> None:
    """Leave immediately. Torch CUDA teardown otherwise keeps the PID alive after FULL OK."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return os.path.abspath(a) == os.path.abspath(b)


def persist_file(local: Path, persist: Path) -> None:
    if _same_path(local, persist):
        return
    best_effort_copy(local, persist, timeout=30.0)


def stage_one_video(src: Path, stage_dir: Path) -> Path:
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(src)
    dest = Path(stage_dir) / src.name
    shutil.copy2(src, dest)
    return dest


def parse_methods(raw: str) -> list[str]:
    methods = [m.strip() for m in str(raw or "vanilla,hybrid").split(",") if m.strip()]
    methods = ["hybrid" if m == "reljoin" else m for m in methods]
    if "vanilla" not in methods:
        methods = ["vanilla"] + methods
    if "dtr" in methods:
        raise ValueError("DTR is not supported. Use vanilla and hybrid.")
    return methods


def parse_modes(raw: str) -> list[str]:
    from reljoin.nextgqa import MODES

    if raw.strip().lower() == "both":
        return ["oracle", "uniform"]
    modes = [m.strip().lower() for m in raw.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise ValueError(f"Unknown modes {bad}. Use oracle, uniform, or both.")
    return modes or ["oracle"]


def cmd_prepare(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download

    from reljoin.nextgqa import DATASET_ID, DATASET_REVISION

    dest = Path(args.data_root)
    dest.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(dest).free / 2**30
    print(f"Preparing {DATASET_ID} @{DATASET_REVISION} -> {dest} (free {free_gb:.1f} GiB)", flush=True)
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=args.dataset_revision or DATASET_REVISION,
        local_dir=str(dest),
    )
    n_mp4 = len(list((dest / "videos").glob("*.mp4"))) if (dest / "videos").is_dir() else 0
    print({"videos_mp4": n_mp4, "val_csv": (dest / "annotations" / "val.csv").is_file()}, flush=True)


def cmd_audit(args: argparse.Namespace) -> None:
    from reljoin.nextgqa import atomic_write_json, audit_nextgqa

    split = args.split if args.split != "both" else None
    payload = audit_nextgqa(Path(args.data_root), split=split)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "coverage.json"
    atomic_write_json(path, payload)
    persist_file(path, Path(args.persist_dir or out) / "coverage.json")
    print(json.dumps(payload, indent=2), flush=True)


def _manifest_record(row: dict, mode: str, decoded, k: int) -> dict:
    return {
        "question_id": row["question_id"],
        "video_id": row["video_id"],
        "qid": row["qid"],
        "split": row["split"],
        "mode": mode,
        "k": int(k),
        "question": row["question"],
        "raw_type": row["raw_type"],
        "eval_type": row["eval_type"],
        "answer_letter": row["answer_letter"],
        "spans": row.get("spans") or [],
        "timestamps": list(decoded.timestamps),
        "actual_times": list(decoded.actual_times),
        "indices": list(decoded.indices),
        "frame_hashes": list(decoded.frame_hashes),
        "video_path": row.get("resolved_video_path") or "",
        "status": decoded.status,
        "error": decoded.error,
        "duration": decoded.duration or row.get("duration"),
        "fps": decoded.fps or row.get("fps"),
    }


def write_manifest(rows: list[dict], modes: list[str], k: int, path: Path) -> list[dict]:
    from reljoin.nextgqa import append_jsonl
    from reljoin.nextgqa_frames import decode_row, plan_times
    from reljoin.nextgqa_frames import DecodeResult

    if path.exists():
        path.unlink()
    records = []
    for row in rows:
        for mode in modes:
            if row.get("skip_reason"):
                decoded = DecodeResult(status=row["skip_reason"], error=row["skip_reason"])
            elif mode == "oracle" and row.get("oracle_skip_reason"):
                decoded = DecodeResult(
                    status=row["oracle_skip_reason"], error=row["oracle_skip_reason"]
                )
            else:
                times, assigned, err, duration = plan_times(row, mode, k)
                if err:
                    decoded = DecodeResult(status=err, error=err, timestamps=times, assigned_spans=assigned)
                else:
                    decoded = decode_row(row, mode, k)
            rec = _manifest_record(row, mode, decoded, k)
            append_jsonl(path, rec)
            records.append(rec)
    return records


def cmd_manifest(args: argparse.Namespace) -> None:
    from reljoin.nextgqa import load_nextgqa_rows

    rows, _ = load_nextgqa_rows(Path(args.data_root), args.split)
    modes = parse_modes(args.mode)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "frame_manifest.jsonl"
    recs = write_manifest(rows, modes, int(args.num_frames), path)
    persist_file(path, Path(args.persist_dir or out) / "frame_manifest.jsonl")
    n_ok = sum(1 for r in recs if r["status"] == "ok")
    print({"n_manifest": len(recs), "n_ok": n_ok, "path": str(path)}, flush=True)


def print_smoke_cases(rows: list[dict], k: int) -> None:
    from reljoin.nextgqa import format_nextgqa_prompt, prompt_has_leak, select_smoke_rows
    from reljoin.nextgqa_frames import decode_row, plan_times

    picks = select_smoke_rows(rows)
    print(f"SMOKE cases: {len(picks)}", flush=True)
    for row in picks:
        print("=" * 60, flush=True)
        print(f"video_path: {row.get('resolved_video_path')}", flush=True)
        print(f"(video_id, qid): ({row['video_id']}, {row['qid']})", flush=True)
        print(f"gold: {row['answer_letter']} | {row['answer']}", flush=True)
        print(f"type: raw={row['raw_type']} eval={row['eval_type']}", flush=True)
        print(f"spans: {row.get('spans')}", flush=True)
        prompt = format_nextgqa_prompt(row["question"], row["choices"])
        print(f"prompt_leak={prompt_has_leak(prompt, row)}", flush=True)
        for mode in ("uniform", "oracle"):
            times, assigned, err, _ = plan_times(row, mode, k)
            decoded = decode_row(row, mode, k) if row.get("resolved_video_path") else None
            print(f"{mode} plan_err={err or ''} timestamps={times}", flush=True)
            if decoded is not None:
                print(
                    f"{mode} decode status={decoded.status} hashes={decoded.frame_hashes} "
                    f"actual={decoded.actual_times}",
                    flush=True,
                )


def prediction_record(row: dict, mode: str, method: str, fingerprint: str, **extra) -> dict:
    rec = {
        "run_fingerprint": fingerprint,
        "question_id": row["question_id"],
        "video_id": row["video_id"],
        "qid": row["qid"],
        "split": row["split"],
        "mode": mode,
        "method": method,
        "backbone": BACKBONE,
        "raw_type": row["raw_type"],
        "eval_type": row["eval_type"],
        "task_type": row["eval_type"],
        "question": row["question"],
        "choices": row["choices"],
        "ground_truth": row["answer_letter"],
        "answer": row["answer"],
        "parsed_prediction": "",
        "raw_output": "",
        "is_valid": False,
        "is_correct": False,
        "status": extra.get("status", "error"),
        "error": extra.get("error", ""),
        "eval_mode": "bypass" if method == "vanilla" else "full",
        "not_acc_gqa": True,
    }
    rec.update(extra)
    return rec


def cmd_report(args: argparse.Namespace) -> None:
    from reljoin.nextgqa import (
        atomic_write_json,
        atomic_write_text,
        csv_table,
        markdown_table,
        read_jsonl,
        score_predictions,
    )

    out = Path(args.out_dir)
    pred_path = Path(args.predictions) if args.predictions else out / "predictions.jsonl"
    records = read_jsonl(pred_path)
    if not records:
        extra = []
        for p in sorted(out.glob("predictions_*.jsonl")):
            extra.extend(read_jsonl(p))
        records = extra
    metrics = score_predictions(records)
    atomic_write_json(out / "metrics.json", metrics)
    atomic_write_text(out / "metrics.md", markdown_table(metrics))
    atomic_write_text(out / "metrics.csv", csv_table(metrics))
    persist = Path(args.persist_dir or out)
    for name in ("metrics.json", "metrics.md", "metrics.csv"):
        persist_file(out / name, persist / name)
    print(markdown_table(metrics), flush=True)


def _load_model_and_join(args, methods: list[str], root: Path):
    import torch

    from reljoin.config import METHOD_TO_VARIANT, METHOD_VERSION, MODEL_ID, VALID_VARIANTS, architecture_variant, variant_to_method
    from reljoin.reljoin_modules import build_joining_module
    from reljoin.qwen_hooks import freeze_backbone, install_reljoin_hook, qwen_hidden_size
    from reljoin.train import load_qwen
    from reljoin.config import ExperimentConfig

    VARIANT_TO_METHOD = {name: variant_to_method(name) for name in VALID_VARIANTS}
    ckpt_root = Path(args.reljoin_ckpt)
    if ckpt_root.is_file():
        ckpt_root = ckpt_root.parent
    found: dict[str, Path] = {}
    if ckpt_root.is_dir():
        for path in sorted(ckpt_root.glob("*.pt")):
            stem = architecture_variant(path.stem)
            if stem not in VALID_VARIANTS:
                continue
            method = variant_to_method(stem)
            found[method] = path
    needed = [m for m in methods if m != "vanilla"]
    missing = [m for m in needed if m not in found]
    if missing:
        scanned = sorted(p.name for p in ckpt_root.glob("*.pt")) if ckpt_root.exists() else []
        raise FileNotFoundError(
            f"Missing checkpoints for {missing} under {ckpt_root}. Found: {scanned or 'NO DIR'}"
        )
    cfg = ExperimentConfig(run_mode="eval_herbench", root=root)
    if args.model_dir:
        cfg.model_dir = Path(args.model_dir)
    print(f"Loading {MODEL_ID} — several minutes of silence is expected.", flush=True)
    model, tokenizer = load_qwen(cfg)
    print(f"{BACKBONE} loaded.", flush=True)
    model.eval()
    freeze_backbone(model)
    install_reljoin_hook(model)
    llm_dim = qwen_hidden_size(model)
    joining = {}
    for method in needed:
        ckpt_path = found[method]
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            raise RuntimeError(f"Unexpected FrameWeave ckpt payload in {ckpt_path}")
        version = str(ckpt.get("method_version") or "")
        if version and not version.startswith("6."):
            raise RuntimeError(f"Refusing checkpoint method_version={version!r}")
        state = ckpt.get("reljoin", ckpt)
        if "global_relation.W_Q.weight" not in state:
            raise RuntimeError(f"Checkpoint {ckpt_path} is not FrameWeave v6")
        ckpt_variant = architecture_variant(METHOD_TO_VARIANT.get(method, method))
        module = build_joining_module(
            llm_dim, variant=ckpt_variant, local_enabled=ckpt.get("local_enabled")
        ).to(device=model.device, dtype=torch.float32)
        module.load_state_dict(state, strict=True)
        module.eval()
        print(
            f"Loaded FrameWeave v6 ckpt {ckpt_path} variant={ckpt_variant} "
            f"method_version={version or METHOD_VERSION}",
            flush=True,
        )
        joining[method] = module
    return model, tokenizer, joining


def run_eval(args: argparse.Namespace, smoke: bool) -> None:
    import gc

    import torch

    from reljoin.config import MAX_PIXELS, METHOD_VERSION, MIN_PIXELS, MODEL_ID, MODEL_REVISION
    from reljoin.diagnostics import parse_choice_strict
    from reljoin.nextgqa import (
        DATASET_REVISION,
        NUM_FRAMES,
        append_jsonl,
        atomic_write_json,
        atomic_write_text,
        audit_nextgqa,
        code_commit,
        csv_table,
        fingerprint_of,
        fingerprint_payload,
        format_nextgqa_prompt,
        load_nextgqa_rows,
        markdown_table,
        prompt_has_leak,
        read_jsonl,
        resume_completed_keys,
        score_predictions,
        select_smoke_rows,
    )
    from reljoin.nextgqa_frames import decode_row
    from reljoin.qwen_generate import generate_qwen
    from reljoin.qwen_hooks import encode_question_hidden, get_hook_state

    root = Path(args.root)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    persist_dir = Path(args.persist_dir) if args.persist_dir else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    persist_dir.mkdir(parents=True, exist_ok=True)
    k = int(args.num_frames or NUM_FRAMES)
    modes = parse_modes(args.mode)
    methods = parse_methods(args.methods)
    split = str(args.split).strip().lower()
    min_pixels = int(args.min_pixels or MIN_PIXELS)
    max_pixels = int(args.max_pixels or MAX_PIXELS)
    dataset_revision = args.dataset_revision or DATASET_REVISION
    model_revision = args.model_revision or MODEL_REVISION

    coverage = audit_nextgqa(data_root, split=split)
    coverage["code_commit"] = code_commit(_THIS)
    atomic_write_json(out_dir / "coverage.json", coverage)
    persist_file(out_dir / "coverage.json", persist_dir / "coverage.json")
    print(json.dumps(coverage["splits"].get(split, coverage["splits"]), indent=2), flush=True)

    rows, _ = load_nextgqa_rows(data_root, split)
    if smoke:
        rows = select_smoke_rows(rows)
        print_smoke_cases(rows, k)

    skip_model = bool(args.skip_model) or not torch.cuda.is_available()
    if skip_model:
        print("No GPU or --skip-model: writing coverage + smoke timestamps only.", flush=True)
        man_path = out_dir / "frame_manifest.jsonl"
        write_manifest(rows, modes, k, man_path)
        persist_file(man_path, persist_dir / "frame_manifest.jsonl")
        print("SMOKE OK" if smoke else "AUDIT OK", flush=True)
        _exit_process(0)

    fps = []
    payloads = {}
    for mode in modes:
        payload = fingerprint_payload(
            split=split,
            mode=mode,
            num_frames=k,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            model_id=MODEL_ID,
            model_revision=model_revision,
            dataset_revision=dataset_revision,
            ckpt=str(args.reljoin_ckpt),
            methods=methods,
            method_version=METHOD_VERSION,
            backbone=BACKBONE,
        )
        payloads[mode] = payload
        fps.append(fingerprint_of(payload))
    tag = fps[0] if len(fps) == 1 else hashlib.sha256("".join(fps).encode()).hexdigest()[:16]
    pred_path = out_dir / f"predictions_{split}_{tag}.jsonl"
    alias = out_dir / "predictions.jsonl"
    persist_pred = persist_dir / pred_path.name
    if persist_pred.exists() and (
        not pred_path.exists() or persist_pred.stat().st_mtime > pred_path.stat().st_mtime
    ):
        if best_effort_copy(persist_pred, pred_path):
            print(f"Restored resume jsonl from persist: {persist_pred}", flush=True)

    completed: set[tuple[str, str, str, str]] = set()
    for mode in modes:
        fp = fingerprint_of(payloads[mode])
        if pred_path.exists():
            completed |= resume_completed_keys(pred_path, fp) if len(modes) == 1 else set()
    if len(modes) > 1 and pred_path.exists():
        allowed = {fingerprint_of(payloads[m]) for m in modes}
        for rec in read_jsonl(pred_path):
            fp = str(rec.get("run_fingerprint") or "")
            if fp and fp not in allowed:
                raise RuntimeError(f"predictions have a different fingerprint {fp}")
            if str(rec.get("status") or "") == "runtime_error":
                continue
            completed.add(
                (
                    str(rec.get("split") or ""),
                    str(rec.get("mode") or ""),
                    str(rec.get("question_id") or ""),
                    str(rec.get("method") or ""),
                )
            )

    model, tokenizer, joining = _load_model_and_join(args, methods, root)
    dtype = model.dtype
    man_path = out_dir / "frame_manifest.jsonl"
    if man_path.exists():
        man_path.unlink()
    stage_dir = Path(args.stage_video_dir) if args.stage_video_dir else None
    if stage_dir is not None:
        stage_dir.mkdir(parents=True, exist_ok=True)

    n_left = 0
    for row in rows:
        for mode in modes:
            for method in methods:
                if (split, mode, row["question_id"], method) not in completed:
                    n_left += 1
    print(f"Pending (question, mode, method) pairs: {n_left}", flush=True)

    for row_i, row in enumerate(rows, start=1):
        staged = None
        read_row = dict(row)
        try:
            src = Path(row.get("resolved_video_path") or "")
            if stage_dir is not None and src.is_file():
                staged = stage_one_video(src, stage_dir)
                read_row["resolved_video_path"] = str(staged)
            for mode in modes:
                decoded = None
                if row.get("skip_reason") or (mode == "oracle" and row.get("oracle_skip_reason") == "missing_spans"):
                    status = row.get("skip_reason") or row.get("oracle_skip_reason")
                    man = {
                        "question_id": row["question_id"],
                        "video_id": row["video_id"],
                        "qid": row["qid"],
                        "split": split,
                        "mode": mode,
                        "k": k,
                        "spans": row.get("spans") or [],
                        "timestamps": [],
                        "video_path": row.get("resolved_video_path") or "",
                        "status": status,
                        "error": status,
                    }
                    append_jsonl(man_path, man)
                    for method in methods:
                        key = (split, mode, row["question_id"], method)
                        if key in completed:
                            continue
                        rec = prediction_record(row, mode, method, fingerprint_of(payloads[mode]), status=status, error=status)
                        append_jsonl(pred_path, rec)
                        persist_file(pred_path, persist_pred)
                        completed.add(key)
                    continue
                decoded = decode_row(read_row, mode, k)
                man = _manifest_record(row, mode, decoded, k)
                append_jsonl(man_path, man)
                if decoded.status != "ok":
                    for method in methods:
                        key = (split, mode, row["question_id"], method)
                        if key in completed:
                            continue
                        rec = prediction_record(
                            row,
                            mode,
                            method,
                            fingerprint_of(payloads[mode]),
                            status=decoded.status,
                            error=decoded.error,
                            frame_timestamps=decoded.timestamps,
                            frame_hashes=decoded.frame_hashes,
                        )
                        append_jsonl(pred_path, rec)
                        persist_file(pred_path, persist_pred)
                        completed.add(key)
                    continue
                prompt = format_nextgqa_prompt(row["question"], row["choices"])
                if prompt_has_leak(prompt, row):
                    raise RuntimeError(f"prompt leak on {row['question_id']}")
                hashes = list(decoded.frame_hashes)
                q_last = encode_question_hidden(
                    model,
                    tokenizer,
                    row["question"],
                    row["choices"],
                    model.device,
                    dtype,
                )
                timestamps = torch.tensor([decoded.timestamps], device=model.device, dtype=dtype)
                frame_mask = torch.ones(1, len(decoded.frames), device=model.device, dtype=torch.bool)
                for method in methods:
                    key = (split, mode, row["question_id"], method)
                    if key in completed:
                        continue
                    t0 = time.time()
                    fp = fingerprint_of(payloads[mode])
                    rec = prediction_record(
                        row,
                        mode,
                        method,
                        fp,
                        status="runtime_error",
                        frame_timestamps=decoded.timestamps,
                        frame_indices=decoded.indices,
                        frame_hashes=hashes,
                        num_frames=len(decoded.frames),
                    )
                    try:
                        bypass = method == "vanilla"
                        raw = generate_qwen(
                            model,
                            decoded.frames,
                            prompt,
                            None if bypass else joining[method],
                            q_last,
                            timestamps,
                            frame_mask,
                            bypass=bypass,
                            max_pixels=max_pixels,
                            min_pixels=min_pixels,
                        )
                        pred, is_valid = parse_choice_strict(raw, n_choices=5)
                        is_correct = bool(is_valid and pred == row["answer_letter"])
                        rec.update(
                            {
                                "parsed_prediction": pred,
                                "raw_output": raw,
                                "is_valid": is_valid,
                                "is_correct": is_correct,
                                "tokens_per_frame": json.dumps(get_hook_state().last_tokens_per_frame),
                                "runtime_seconds": time.time() - t0,
                                "status": "ok",
                                "error": "",
                            }
                        )
                        print(
                            f"  {mode} {method} {row['question_id']} pred={pred} gt={row['answer_letter']} "
                            f"ok={is_correct} {time.time() - t0:.1f}s hashes={hashes}",
                            flush=True,
                        )
                    except torch.cuda.OutOfMemoryError as exc:
                        rec["error"] = repr(exc)
                        rec["status"] = "oom"
                        append_jsonl(pred_path, rec)
                        persist_file(pred_path, persist_pred)
                        raise SystemExit(2) from exc
                    except Exception as exc:
                        rec["error"] = repr(exc)
                        rec["status"] = "runtime_error"
                        append_jsonl(pred_path, rec)
                        persist_file(pred_path, persist_pred)
                        raise
                    append_jsonl(pred_path, rec)
                    persist_file(pred_path, persist_pred)
                    completed.add(key)
        finally:
            if staged is not None and staged.exists():
                staged.unlink()
        if row_i % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"progress {row_i}/{len(rows)}", flush=True)

    persist_file(man_path, persist_dir / "frame_manifest.jsonl")
    if pred_path.exists():
        if alias.resolve() != pred_path.resolve():
            shutil.copy2(pred_path, alias)
        persist_file(pred_path, persist_pred)
        persist_file(alias, persist_dir / "predictions.jsonl")
        records = read_jsonl(pred_path)
        metrics = score_predictions(records)
        metrics["fingerprint"] = {mode: fingerprint_of(payloads[mode]) for mode in modes}
        metrics["code_commit"] = coverage["code_commit"]
        metrics["dataset_revision"] = dataset_revision
        metrics["model_revision"] = model_revision
        metrics["num_frames"] = k
        metrics["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(out_dir / "metrics.json", metrics)
        atomic_write_text(out_dir / "metrics.md", markdown_table(metrics))
        atomic_write_text(out_dir / "metrics.csv", csv_table(metrics))
        for name in ("metrics.json", "metrics.md", "metrics.csv"):
            persist_file(out_dir / name, persist_dir / name)
        print(markdown_table(metrics), flush=True)
    print("SMOKE OK" if smoke else "FULL OK", flush=True)
    _exit_process(0)


def build_parser() -> argparse.ArgumentParser:
    from reljoin.nextgqa import DATASET_REVISION, NUM_FRAMES

    p = argparse.ArgumentParser(description="NExT-GQA FrameWeave V6 eval (Qwen2.5-VL-7B-Instruct)")
    p.add_argument("command", choices=["prepare", "audit", "manifest", "smoke", "run", "report"])
    p.add_argument("--root", default="/content")
    p.add_argument("--data-root", default="/content/nextgqa")
    p.add_argument("--out-dir", default="/content/outputs_reljoin_v6_qwen/eval_nextgqa")
    p.add_argument("--persist-dir", default="")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--reljoin-ckpt", default=DEFAULT_CKPT)
    p.add_argument("--dataset-revision", default=DATASET_REVISION)
    p.add_argument("--model-revision", default="")
    p.add_argument("--split", default="val", choices=["val", "test", "both"])
    p.add_argument("--mode", default="oracle", help="oracle | uniform | both")
    p.add_argument("--methods", default="vanilla,hybrid")
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    p.add_argument("--max-pixels", type=int, default=MAX_PIXELS_DEFAULT)
    p.add_argument("--min-pixels", type=int, default=MIN_PIXELS_DEFAULT)
    p.add_argument("--stage-video-dir", default="")
    p.add_argument("--skip-model", action="store_true")
    p.add_argument("--predictions", default="")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.split == "both" and args.command in {"manifest", "smoke", "run"}:
        raise SystemExit("Use --split val or --split test. Run val first, then test.")
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "audit":
        cmd_audit(args)
    elif args.command == "manifest":
        cmd_manifest(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "smoke":
        run_eval(args, smoke=True)
    else:
        run_eval(args, smoke=False)


if __name__ == "__main__":
    main()
