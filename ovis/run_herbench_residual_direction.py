#!/usr/bin/env python3
"""HERBench lite_v2 residual-direction eval for FrameWeave Hybrid v6.

Captures ΔX once per question from a fixed Hybrid.pt checkpoint, after the
gate, RMS calibration, and mask. Later conditions only replace the addition
at the frozen LLM input: X, X+ΔX, X+2ΔX, X−ΔX, X+R, and X+D^parallel.
Scaling does not run RMS calibration again.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

os.environ.pop("PYTHONPATH", None)
os.environ.pop("PYTHONHOME", None)

import pandas as pd
import torch

from reljoin.residual_direction import (
    CONDITIONS,
    assert_injection_matches_spec,
    build_injection,
    jobs_for_sample,
)
from run_herbench_oracle_reljoin import (
    best_effort_copy,
    load_reljoin_module,
    row_task_type,
    stage_one_video,
)

FIELD_ORDER = [
    "run_fingerprint",
    "sample_id",
    "question_id",
    "video_id",
    "video_path",
    "task_type",
    "condition",
    "lambda_scale",
    "random_seed",
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
    "norm_ratio_vs_delta",
    "cosine_vs_delta",
    "cosine_vs_x",
    "max_abs_norm_error",
    "n_nonzero_delta",
    "runtime_seconds",
    "status",
    "error_message",
    "error",
    "eval_mode",
]


def csv_fields(path: Path) -> list[str]:
    path = Path(path)
    if path.is_file() and path.stat().st_size > 0:
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), None)
        if header:
            return list(header)
    return list(FIELD_ORDER)


def append_record(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = csv_fields(path)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: record.get(key, "") for key in fields})
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass


def save_record(local_csv: Path, persist_csv: Path, record: dict) -> None:
    append_record(local_csv, record)
    try:
        if local_csv.resolve() == persist_csv.resolve():
            return
    except OSError:
        if os.path.abspath(local_csv) == os.path.abspath(persist_csv):
            return
    append_record(persist_csv, record)


def job_key(question_id: str, condition: str, random_seed: int) -> tuple[str, str, int]:
    return (str(question_id), str(condition), int(random_seed))


def _as_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _json_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def capture_delta(
    model,
    joining,
    input_ids,
    pixel_values,
    grid_thws,
    question_hidden,
    timestamps,
    frame_mask,
) -> dict:
    """One full forward. ΔX is last_delta_x after gate, RMS, and mask."""
    from reljoin.ovis_hooks import (
        configure_reljoin,
        get_hook_state,
        reset_hook_diagnostics,
        set_reljoin_context,
    )

    reset_hook_diagnostics()
    configure_reljoin(
        joining,
        enabled=True,
        bypass=False,
        eval_mode="full",
        inject_delta=None,
    )
    set_reljoin_context(
        question_hidden=question_hidden,
        timestamps=timestamps,
        frame_mask=frame_mask,
    )
    with torch.inference_mode():
        model.merge_multimodal(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
        )
    hook = get_hook_state()
    if joining.last_delta_x is None or hook.last_visual_embeds is None or hook.last_token_mask is None:
        raise RuntimeError("RelJoin capture failed: missing ΔX, X, or token_mask")
    return {
        "delta": joining.last_delta_x.detach().clone(),
        "x": hook.last_visual_embeds.detach().clone(),
        "token_mask": hook.last_token_mask.detach().clone(),
        "tokens_per_frame": list(hook.last_tokens_per_frame),
    }


def _condition_frame(df: pd.DataFrame, condition: str, random_seed: int | None = None) -> pd.DataFrame:
    sub = df[df["condition"].astype(str) == condition]
    if random_seed is not None and "random_seed" in sub.columns:
        seeds = pd.to_numeric(sub["random_seed"], errors="coerce")
        sub = sub[seeds == int(random_seed)]
    return sub


def _by_task(sub: pd.DataFrame) -> dict:
    if sub.empty or "task_type" not in sub.columns:
        return {}
    out = {}
    for task, task_rows in sub.groupby("task_type"):
        n = len(task_rows)
        out[str(task)] = {
            "n": int(n),
            "strict_accuracy": float(task_rows["is_correct"].mean()) if n else 0.0,
        }
    return out


def run(args: argparse.Namespace) -> None:
    from reljoin.config import (
        HERBENCH_ORACLE_FRAMES,
        MAX_PIXELS,
        METHOD_VERSION,
        MIN_PIXELS,
        MODEL_ID,
        ExperimentConfig,
    )
    from reljoin.diagnostics import accuracy_reports, parse_choice_strict
    from reljoin.herbench_oracle import (
        choose_oracle_times,
        extract_evidence_spans,
        load_module_from_file,
        load_oracle_pil_frames,
        load_rows,
        sha256_file,
    )
    from reljoin.ovis_generate import format_mc_prompt, generate_ovis, preprocess_ovis_inputs
    from reljoin.ovis_hooks import (
        begin_prepare_cache,
        encode_question_hidden,
        end_prepare_cache,
        freeze_backbone,
        install_reljoin_hook,
        ovis_hidden_size,
    )
    from reljoin.stats import compare_methods, save_statistical_tests
    from reljoin.train import load_ovis

    herbench_config = str(getattr(args, "herbench_config", "lite_v2") or "lite_v2").strip()
    if herbench_config != "lite_v2":
        raise ValueError(
            f"Residual-direction eval is HERBench lite_v2 only, got {herbench_config!r}."
        )
    n_seeds = int(args.n_random_seeds)
    jobs = jobs_for_sample(n_seeds)

    root = Path(args.root)
    herbench_root = Path(args.herbench_root)
    videos_root = Path(args.videos_root)
    out_dir = Path(args.out_dir)
    persist_dir = Path(args.persist_dir) if args.persist_dir else out_dir
    if args.smoke:
        out_dir = out_dir / "smoke"
        persist_dir = persist_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    persist_dir.mkdir(parents=True, exist_ok=True)

    stage_dir_arg = str(getattr(args, "stage_video_dir", "") or "")
    stage_dir = Path(stage_dir_arg) if stage_dir_arg else None
    if stage_dir is not None:
        stage_dir.mkdir(parents=True, exist_ok=True)

    base_src = Path(args.herbench_base_vlm) if args.herbench_base_vlm else Path()
    acc_src = Path(args.herbench_accuracy) if args.herbench_accuracy else Path()
    if base_src.is_file() and args.herbench_base_sha256 and sha256_file(base_src) != args.herbench_base_sha256:
        raise RuntimeError("HERBench base_vlm.py SHA256 mismatch")
    if acc_src.is_file() and args.herbench_acc_sha256 and sha256_file(acc_src) != args.herbench_acc_sha256:
        raise RuntimeError("HERBench calculate_accuracy.py SHA256 mismatch")
    base_mod = load_module_from_file("herbench_base_pinned_direction", base_src) if base_src.is_file() else None
    acc_mod = load_module_from_file("herbench_accuracy_pinned_direction", acc_src) if acc_src.is_file() else None

    print(f"Loading HERBench lite_v2 ({args.herbench_revision}) ...", flush=True)
    rows = load_rows(herbench_root, videos_root, args.herbench_revision, config="lite_v2")
    for row in rows:
        if not row.get("task_type"):
            row["task_type"] = row_task_type(row)
    missing = [r["question_id"] for r in rows if not r["resolved_video_path"]]
    resolved_rows = [r for r in rows if r["resolved_video_path"]]
    print(
        f"Loaded {len(rows)} questions, {len(resolved_rows)} with local mp4, {len(missing)} missing video",
        flush=True,
    )
    if missing and not args.smoke:
        raise RuntimeError(
            f"{len(missing)} questions missing videos; first={missing[:5]}. --videos-root={videos_root}."
        )
    if args.smoke and not resolved_rows:
        raise RuntimeError(f"Smoke found no local mp4 under {videos_root}.")

    ckpt = Path(args.reljoin_ckpt)
    if ckpt.is_dir():
        ckpt = ckpt / "Hybrid.pt"
    if ckpt.name != "Hybrid.pt":
        raise RuntimeError(
            f"This experiment uses the fixed Hybrid checkpoint, got {ckpt.name}."
        )
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing Hybrid checkpoint: {ckpt}")

    cfg = ExperimentConfig(run_mode="eval_herbench", root=root)
    if args.model_dir:
        cfg.model_dir = Path(args.model_dir)
    print(f"Loading {MODEL_ID} — several minutes of silence is expected.", flush=True)
    model, tokenizer = load_ovis(cfg)
    print("Ovis2.5-9B loaded.", flush=True)
    model.eval()
    freeze_backbone(model)
    install_reljoin_hook(model)
    dtype = model.dtype
    joining = load_reljoin_module(ckpt, ovis_hidden_size(model), model.device, variant="Hybrid")

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
    max_new_tokens = int(args.max_new_tokens)

    fingerprint_payload = {
        "protocol": "herbench_lite_v2_oracle_frameweave_v6_ovis25_residual_direction",
        "herbench_config": "lite_v2",
        "dataset_revision": args.herbench_revision,
        "model_id": MODEL_ID,
        "model_revision": args.model_revision,
        "num_frames": num_frames,
        "max_pixels": max_pixels,
        "min_pixels": min_pixels,
        "max_new_tokens": max_new_tokens,
        "n_random_seeds": n_seeds,
        "conditions": list(CONDITIONS),
        "reljoin_ckpt": str(ckpt),
        "method_version": METHOD_VERSION,
        "formulation": "5.1.0",
        "backbone": "Ovis2.5-9B",
        "smoke": bool(args.smoke),
        "note": "lambda scales captured delta; RMS calibration is not rerun",
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    pairs_csv = out_dir / f"predictions_residual_{fingerprint}.csv"
    persist_csv = persist_dir / pairs_csv.name
    if persist_csv.exists() and (
        not pairs_csv.exists() or persist_csv.stat().st_mtime > pairs_csv.stat().st_mtime
    ):
        if best_effort_copy(persist_csv, pairs_csv):
            print(f"Restored resume CSV from persist: {persist_csv}", flush=True)

    completed: set[tuple[str, str, int]] = set()
    if pairs_csv.exists():
        prev = pd.read_csv(pairs_csv, on_bad_lines="error")
        for _, record in prev.iterrows():
            key = job_key(record["question_id"], record["condition"], int(float(record["random_seed"])))
            if str(record.get("status")) == "runtime_error":
                completed.discard(key)
            else:
                completed.add(key)
        print(f"Resuming {len(completed)} completed condition rows from {pairs_csv}", flush=True)

    pending = resolved_rows
    if args.smoke:
        smoke_rows = []
        seen_videos = set()
        for row in resolved_rows:
            if row["resolved_video_path"] not in seen_videos:
                smoke_rows.append(row)
                seen_videos.add(row["resolved_video_path"])
            if len(smoke_rows) == 2:
                break
        pending = smoke_rows or resolved_rows[:2]
        print(f"SMOKE mode: {len(pending)} questions × {len(jobs)} conditions", flush=True)

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in pending:
        groups[row["resolved_video_path"]].append(row)

    print(
        "Conditions:",
        ", ".join(f"{spec['condition']}(λ={spec['lambda_scale']}, seed={spec['random_seed']})" for spec in jobs),
        flush=True,
    )

    for video_number, (video_path, video_rows) in enumerate(groups.items(), start=1):
        video_name = Path(video_path).name
        left = 0
        for row in video_rows:
            left += sum(
                job_key(row["question_id"], spec["condition"], spec["random_seed"]) not in completed
                for spec in jobs
            )
        if left == 0:
            print(f"Video {video_number}/{len(groups)}: {video_name} skip", flush=True)
            continue
        print(f"Video {video_number}/{len(groups)}: {video_name} pending={left}", flush=True)
        staged: Path | None = None
        read_path = Path(video_path)
        try:
            if stage_dir is not None:
                read_path = stage_one_video(Path(video_path), stage_dir)
                staged = read_path
            for row in video_rows:
                todo = [
                    spec
                    for spec in jobs
                    if job_key(row["question_id"], spec["condition"], spec["random_seed"]) not in completed
                ]
                if not todo:
                    continue
                spans, mode_hint = extract_evidence_spans(row["metadata"])
                times, padded, mode = choose_oracle_times(spans, row["duration"], num_frames=num_frames)
                if mode_hint == "uniform_fallback":
                    mode = "uniform_fallback"
                    padded = False
                frames, indices, _meta = load_oracle_pil_frames(read_path, times, row["duration"])
                prompt = format_prompt(row)
                answer = row["answer"]
                option_order = list("ABCDE"[: len([c for c in row["choices"] if c])])
                question_hidden = encode_question_hidden(
                    model,
                    tokenizer,
                    row["question"],
                    row["choices"],
                    model.device,
                    dtype,
                )
                timestamps = torch.tensor([times], device=model.device, dtype=dtype)
                frame_mask = torch.ones(1, len(frames), device=model.device, dtype=torch.bool)
                begin_prepare_cache()
                try:
                    input_ids, pixel_values, grid_thws = preprocess_ovis_inputs(
                        model, frames, prompt, min_pixels, max_pixels
                    )
                    capture = capture_delta(
                        model,
                        joining,
                        input_ids,
                        pixel_values,
                        grid_thws,
                        question_hidden,
                        timestamps,
                        frame_mask,
                    )
                    for spec in todo:
                        key = job_key(row["question_id"], spec["condition"], spec["random_seed"])
                        injected = build_injection(
                            capture["delta"],
                            capture["x"],
                            capture["token_mask"],
                            spec,
                            str(row["question_id"]),
                        )
                        metrics = assert_injection_matches_spec(
                            capture["delta"],
                            capture["x"],
                            capture["token_mask"],
                            injected,
                            spec,
                        )
                        t0 = time.time()
                        record = {
                            "run_fingerprint": fingerprint,
                            "sample_id": row["question_id"],
                            "question_id": row["question_id"],
                            "video_id": row.get("video_id", ""),
                            "video_path": row.get("video_path", ""),
                            "task_type": row.get("task_type", ""),
                            "condition": spec["condition"],
                            "lambda_scale": float(spec["lambda_scale"]),
                            "random_seed": int(spec["random_seed"]),
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
                            "tokens_per_frame": json.dumps(capture["tokens_per_frame"]),
                            "norm_ratio_vs_delta": metrics["norm_ratio_vs_delta"],
                            "cosine_vs_delta": metrics["cosine_vs_delta"],
                            "cosine_vs_x": metrics["cosine_vs_x"],
                            "max_abs_norm_error": metrics["max_abs_norm_error"],
                            "n_nonzero_delta": metrics["n_nonzero_delta"],
                            "eval_mode": "inject",
                            "status": "runtime_error",
                            "error_message": "",
                            "error": "",
                        }
                        try:
                            raw = generate_ovis(
                                model,
                                frames,
                                prompt,
                                joining,
                                question_hidden,
                                timestamps,
                                frame_mask,
                                bypass=False,
                                max_pixels=max_pixels,
                                min_pixels=min_pixels,
                                max_new_tokens=max_new_tokens,
                                eval_mode="full",
                                inject_delta=injected,
                                prepared=(input_ids, pixel_values, grid_thws),
                            )
                            pred = parse_choice(raw)
                            is_valid = pred != "ERROR"
                            if not is_valid:
                                pred2, ok = parse_choice_strict(raw)
                                pred = pred2
                                is_valid = ok
                            is_correct = bool(is_valid and pred == answer)
                            record.update(
                                {
                                    "parsed_prediction": pred,
                                    "raw_output": raw,
                                    "is_valid": is_valid,
                                    "is_correct": is_correct,
                                    "runtime_seconds": time.time() - t0,
                                    "status": "ok",
                                }
                            )
                            print(
                                f"  {spec['condition']} λ={spec['lambda_scale']} "
                                f"seed={spec['random_seed']} {row['question_id']} "
                                f"pred={pred} gt={answer} ok={is_correct} "
                                f"ratio={metrics['norm_ratio_vs_delta']} "
                                f"{time.time() - t0:.1f}s",
                                flush=True,
                            )
                        except Exception as exc:
                            record["error_message"] = repr(exc)
                            record["error"] = repr(exc)
                            record["status"] = "runtime_error"
                            record["runtime_seconds"] = time.time() - t0
                            print(
                                f"  ERROR {row['question_id']} {spec['condition']}: {exc!r}",
                                flush=True,
                            )
                        save_record(pairs_csv, persist_csv, record)
                        if record["status"] == "ok":
                            completed.add(key)
                        del injected
                finally:
                    end_prepare_cache()
        finally:
            if staged is not None and staged.exists():
                staged.unlink()
                print(f"Removed staged {video_name}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not pairs_csv.exists():
        raise RuntimeError(f"No predictions written to {pairs_csv}")
    df = pd.read_csv(pairs_csv)
    before = len(df)
    df = df.drop_duplicates(subset=["question_id", "condition", "random_seed"], keep="last")
    if len(df) != before:
        print(f"Dropped {before - len(df)} duplicate CSV rows (kept last).", flush=True)
    df["is_correct"] = df["is_correct"].map(_as_bool)
    df["is_valid"] = df["is_valid"].map(_as_bool) if "is_valid" in df.columns else False
    df_ok = df[df["status"].astype(str) == "ok"].copy()

    summary: dict[str, dict] = {}
    random_seeds = (
        sorted(
            {
                int(float(s))
                for s in df_ok.loc[df_ok["condition"].astype(str) == "random", "random_seed"]
            }
        )
        if not df_ok.empty
        else []
    )
    summary_keys: list[tuple[str, str, int | None]] = [
        ("vanilla", "vanilla", None),
        ("learned", "learned", None),
        ("double", "double", None),
        ("reverse", "reverse", None),
        ("parallel", "parallel", None),
    ]
    if len(random_seeds) <= 1:
        summary_keys.append(("random", "random", random_seeds[0] if random_seeds else None))
    else:
        for seed in random_seeds:
            summary_keys.append((f"random_s{seed}", "random", seed))

    for label, condition, seed in summary_keys:
        sub = _condition_frame(df_ok, condition, seed if condition == "random" else None)
        rows_m = sub.to_dict(orient="records")
        summary[label] = accuracy_reports(rows_m)
        preds = {}
        for _, record in sub.iterrows():
            preds[str(record["question_id"])] = {
                "predicted_choice": str(record["parsed_prediction"]),
                "is_correct": bool(record["is_correct"]),
                "response": str(record["raw_output"]),
                "frames_used": int(record["num_frames"]),
                "gt_answer": str(record["ground_truth"]),
                "task_type": str(record["task_type"]),
            }
        summary[label]["official"] = (
            acc_mod.calculate_accuracy(preds) if acc_mod is not None and preds else {}
        )
        summary[label]["by_task_type"] = _by_task(sub)
        if "norm_ratio_vs_delta" in sub.columns and len(sub):
            summary[label]["mean_norm_ratio_vs_delta"] = _json_number(sub["norm_ratio_vs_delta"].mean())
            summary[label]["mean_cosine_vs_delta"] = _json_number(sub["cosine_vs_delta"].mean())
            summary[label]["mean_cosine_vs_x"] = _json_number(sub["cosine_vs_x"].mean())
        sub.to_csv(out_dir / f"herbench_{label}.csv", index=False)

    stats_payload = {}
    vanilla_rows = _condition_frame(df_ok, "vanilla").to_dict(orient="records")
    for label, condition, seed in summary_keys:
        if condition == "vanilla":
            continue
        other = _condition_frame(df_ok, condition, seed if condition == "random" else None)
        if other.empty or not vanilla_rows:
            continue
        stats_payload[f"vanilla_vs_{label}"] = compare_methods(
            vanilla_rows,
            other.to_dict(orient="records"),
            id_key="question_id",
            correct_key="is_correct",
        )
    save_statistical_tests(stats_payload, out_dir / "statistical_tests.json")

    payload = {
        "run_fingerprint": fingerprint,
        "manifest": fingerprint_payload,
        "herbench_config": "lite_v2",
        "n_questions": int(len(rows)),
        "n_resolved": int(len(resolved_rows)),
        "n_missing": int(len(missing)),
        "smoke": bool(args.smoke),
        "n_random_seeds": n_seeds,
        "checkpoint": str(ckpt),
        "summary": summary,
        "stats": stats_payload,
    }
    summary_path = out_dir / f"summary_residual_{fingerprint}.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if summary_path.resolve() != (persist_dir / summary_path.name).resolve():
        best_effort_copy(summary_path, persist_dir / summary_path.name)
        best_effort_copy(out_dir / "statistical_tests.json", persist_dir / "statistical_tests.json")
    print(f"Wrote {summary_path}", flush=True)
    for label, payload_i in summary.items():
        print(
            f"  {label}: acc={payload_i.get('strict_accuracy')} n={payload_i.get('n')} "
            f"norm_ratio={payload_i.get('mean_norm_ratio_vs_delta')}",
            flush=True,
        )
    print("SMOKE OK" if args.smoke else "FULL OK", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HERBench lite_v2 residual direction controls (FrameWeave v6 / Ovis2.5)"
    )
    parser.add_argument("--root", default="/content")
    parser.add_argument("--herbench-root", required=True)
    parser.add_argument("--videos-root", required=True)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--persist-dir", default="")
    parser.add_argument("--herbench-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--herbench-base-vlm", default="")
    parser.add_argument("--herbench-accuracy", default="")
    parser.add_argument("--herbench-base-sha256", default="")
    parser.add_argument("--herbench-acc-sha256", default="")
    parser.add_argument(
        "--reljoin-ckpt",
        default="/content/outputs_reljoin_v6_ovis/checkpoints/best_external_val/Hybrid.pt",
    )
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--min-pixels", type=int, default=448 * 448)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--n-random-seeds", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--herbench-config", default="lite_v2")
    parser.add_argument("--stage-video-dir", default="")
    return parser


if __name__ == "__main__":
    try:
        run(build_parser().parse_args())
    except Exception:
        import traceback

        traceback.print_exc()
        os._exit(1)
    os._exit(0)
