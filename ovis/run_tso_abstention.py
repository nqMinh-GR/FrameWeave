#!/usr/bin/env python3
"""Mask the TSO abstention letter at answer time. Inference only.

Vanilla bypasses the hook. FrameWeave adds the captured final ΔX.
Each model is generated once; masked and unmasked answers share that score vector.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

os.environ.pop("PYTHONPATH", None)
os.environ.pop("PYTHONHOME", None)

import torch

from reljoin.config import FORMULATION, METHOD_VERSION, MODEL_ID
from reljoin.diagnostics import parse_choice_strict
from reljoin.herbench_oracle import (
    EXPECTED_TSO_N,
    load_module_from_file,
    load_oracle_pil_frames,
    resolve_video,
    sha256_file,
)
from reljoin.ovis_generate import format_mc_prompt, generate_ovis, preprocess_ovis_inputs
from reljoin.ovis_hooks import (
    begin_prepare_cache,
    encode_question_hidden,
    end_prepare_cache,
    freeze_backbone,
    get_hook_state,
    install_reljoin_hook,
)
from reljoin.tso_abstention import (
    ANALYSIS_SEED,
    MODELS,
    N_BOOTSTRAP,
    SCORING_FIRST,
    SCORING_SEQUENCE,
    ProtocolError,
    append_csv,
    append_jsonl,
    argmax_letter,
    assert_prompt_keeps_choices,
    bool_cell,
    build_reports,
    build_tso_manifest,
    canonical_manifest_hash,
    encode_scores,
    error_pair_records,
    fingerprint16,
    generation_flags,
    greedy_matches_unmasked,
    input_fingerprint,
    load_jsonl,
    load_prediction_rows,
    load_source_table,
    prediction_pair_records,
    predictions_from_score_record,
    resume_plan,
    scoring_mode_for_letters,
    select_smoke_ids,
    smoke_id_payload,
    write_json,
    write_reports,
)
from reljoin.train import ExperimentConfig, load_ovis
from run_herbench_oracle_reljoin import load_reljoin_module, stage_one_video
from run_herbench_residual_direction import capture_delta


class ScoringFallback(RuntimeError):
    """First-token letter scores do not match the greedy generation."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TSO abstention mask on FrameWeave v6 / Ovis2.5 (inference only)"
    )
    parser.add_argument("--results-path", required=True)
    parser.add_argument(
        "--checkpoint",
        default="/content/outputs_reljoin_v6_ovis/checkpoints/best_external_val/Hybrid.pt",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", type=int, default=0, help="0 runs all 152; 8 runs the fixed smoke ids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--root", default="/content")
    parser.add_argument("--herbench-root", default="/content/HERBench_official")
    parser.add_argument("--videos-root", default="/content/HERBench_official/videos")
    parser.add_argument("--model-dir", default="/content/checkpoints/ovis2_5_9b")
    parser.add_argument("--herbench-revision", default="30b42e97180f2c7c0fdc3f2aed412ae04f0366ca")
    parser.add_argument("--model-revision", default="5a1f9b995c98ae5b85a28ec89d88d74295be0ea7")
    parser.add_argument("--herbench-base-vlm", default="/content/herbench_eval_pins/base_vlm.py")
    parser.add_argument(
        "--herbench-base-sha256",
        default="6104884c366906d5c223cc9727c3bad778dcdf98c881c4b17b556c20e940af97",
    )
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--min-pixels", type=int, default=448 * 448)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--scoring-mode",
        default="auto",
        choices=["auto", SCORING_FIRST, SCORING_SEQUENCE],
    )
    parser.add_argument("--stage-video-dir", default="")
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--analysis-seed", type=int, default=ANALYSIS_SEED)
    return parser


def _read_config(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _archive_outputs(output_dir: Path) -> None:
    for name in (
        "predictions.csv",
        "candidate_scores.jsonl",
        "summary.json",
        "statistical_tests.json",
        "results.md",
        "results.tex",
        "config.json",
    ):
        src = output_dir / name
        if not src.is_file():
            continue
        dest = output_dir / f"{src.stem}.first_token_abandoned{src.suffix}"
        if dest.exists():
            dest = output_dir / f"{src.stem}.first_token_abandoned.{int(time.time())}{src.suffix}"
        src.replace(dest)


def _encode_letters(tokenizer, letters: list[str]) -> dict[str, list[int]]:
    encoded = {}
    for letter in letters:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        encoded[letter] = [int(item) for item in ids]
        if not encoded[letter]:
            raise ProtocolError(f"tokenizer returned no tokens for {letter}")
    return encoded


def _next_token_logits(model, input_ids, pixel_values, grid_thws):
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": pixel_values,
        "grid_thws": grid_thws,
        "use_cache": False,
    }
    with torch.inference_mode():
        try:
            out = model(**kwargs)
        except TypeError:
            kwargs.pop("attention_mask", None)
            kwargs.pop("use_cache", None)
            out = model(**kwargs)
    logits = out.logits if hasattr(out, "logits") else out[0]
    return logits[0, -1].detach().float().cpu()


def _letter_logprobs(step, encoded: dict[str, list[int]], letters: list[str]) -> dict[str, float]:
    logp = torch.log_softmax(step.float(), dim=-1)
    return {letter: float(logp[int(encoded[letter][0])].item()) for letter in letters}


def _continuation_logprob(model, input_ids, pixel_values, grid_thws, token_ids: list[int]) -> float:
    if not token_ids:
        raise ProtocolError("empty letter token sequence")
    continuation = torch.tensor([token_ids], device=input_ids.device, dtype=input_ids.dtype)
    full = torch.cat([input_ids, continuation], dim=1)
    state = get_hook_state()
    saved_flag = state.cache_prepare
    saved_cache = state.cached_result
    state.cache_prepare = False
    state.cached_result = None
    try:
        with torch.inference_mode():
            try:
                out = model(
                    input_ids=full,
                    attention_mask=torch.ones_like(full),
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    use_cache=False,
                )
            except TypeError:
                out = model(input_ids=full, pixel_values=pixel_values, grid_thws=grid_thws)
    finally:
        state.cache_prepare = saved_flag
        state.cached_result = saved_cache
    logits = out.logits if hasattr(out, "logits") else out[0]
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    prompt_len = int(input_ids.shape[-1])
    total = 0.0
    for offset, token_id in enumerate(token_ids):
        total += float(logp[prompt_len - 1 + offset, int(token_id)].item())
    return total


def _parse_letter(text: str, letters: list[str], base_mod) -> tuple[str, bool]:
    if base_mod is not None:
        pred = base_mod.BaseVLM.extract_answer_choice(None, str(text))
        pred = "" if pred is None else str(pred).strip()
        if pred.upper() != "ERROR" and pred:
            letter = pred.upper()[:1]
            if letter in letters:
                return letter, True
    return parse_choice_strict(text, n_choices=len(letters))


def _write_manifest(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> None:
    if int(args.smoke) < 0:
        raise ProtocolError("--smoke must be >= 0")
    if int(args.bootstrap) < 1:
        raise ProtocolError("--bootstrap must be >= 1")
    if int(args.num_frames) != 8:
        raise ProtocolError("this protocol uses the stored 8 frame timestamps")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = output_dir / "predictions.csv"
    score_path = output_dir / "candidate_scores.jsonl"
    config_path = output_dir / "config.json"

    frame, member = load_source_table(Path(args.results_path))
    manifest_result = build_tso_manifest(frame, source_member=member, expected_n=EXPECTED_TSO_N)
    manifest = manifest_result.samples
    source_hash = canonical_manifest_hash(manifest)
    _write_manifest(output_dir / "tso_manifest.jsonl", manifest)
    print(
        f"TSO manifest {len(manifest)} questions, "
        f"abstention letters {dict(Counter(sample['abstention_letter'] for sample in manifest))}",
        flush=True,
    )
    if manifest_result.learned_missing_ids or manifest_result.learned_extra_ids:
        print(
            f"Learned cross-check missing {len(manifest_result.learned_missing_ids)} "
            f"extra {len(manifest_result.learned_extra_ids)}",
            flush=True,
        )

    if int(args.smoke) == 0 or int(args.smoke) >= len(manifest):
        selected = list(manifest)
    else:
        smoke_ids = select_smoke_ids(manifest, int(args.smoke))
        by_id = {sample["sample_id"]: sample for sample in manifest}
        selected = [by_id[sample_id] for sample_id in smoke_ids]
        write_json(output_dir / "smoke_ids.json", smoke_id_payload(manifest, smoke_ids))
        print(f"Smoke ids: {smoke_ids}", flush=True)

    ckpt = Path(args.checkpoint)
    if ckpt.is_dir():
        ckpt = ckpt / "Hybrid.pt"
    if ckpt.name != "Hybrid.pt":
        raise RuntimeError(f"This experiment uses Hybrid.pt, got {ckpt.name}.")
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing Hybrid checkpoint: {ckpt}")
    ckpt_hash = sha256_file(ckpt)

    base_path = Path(args.herbench_base_vlm) if args.herbench_base_vlm else Path()
    base_mod = None
    prompt_id = "format_mc_prompt"
    if base_path.is_file():
        digest = sha256_file(base_path)
        expected = str(args.herbench_base_sha256 or "")
        if expected and digest != expected:
            raise RuntimeError(f"HERBench base_vlm.py SHA256 mismatch: {digest} != {expected}")
        base_mod = load_module_from_file("herbench_base_pinned_tso", base_path)
        prompt_id = digest
        print(f"Prompt formatter {base_path} {digest[:12]}", flush=True)
    else:
        print(f"Using built-in MC prompt; base_vlm.py not at {base_path}", flush=True)

    def format_prompt(sample: dict) -> str:
        if base_mod is not None:
            return base_mod.BaseVLM._format_prompt(None, sample["question"], sample["choices"])
        return format_mc_prompt(sample["question"], sample["choices"])

    letters = list(manifest[0]["letters"])
    if any(sample["letters"] != letters for sample in manifest):
        raise ProtocolError("TSO samples do not share one A–E label set")

    existing = _read_config(config_path)
    if pred_csv.exists() and pred_csv.stat().st_size > 0 and not args.resume:
        raise ProtocolError(
            f"{pred_csv} already exists. Pass --resume or choose an empty output directory."
        )

    def fingerprint_for(scoring_mode: str, dtype_name: str, torch_version: str) -> tuple[str, dict]:
        payload = {
            "protocol": "tso_abstention_mask_frameweave_v6_ovis25",
            "source_manifest_sha256": source_hash,
            "source_member": member,
            "checkpoint_sha256": ckpt_hash,
            "checkpoint": str(ckpt),
            "model_id": MODEL_ID,
            "model_revision": args.model_revision,
            "dataset_revision": args.herbench_revision,
            "method_version": METHOD_VERSION,
            "formulation": FORMULATION,
            "min_pixels": int(args.min_pixels),
            "max_pixels": int(args.max_pixels),
            "num_frames": int(args.num_frames),
            "max_new_tokens": int(args.max_new_tokens),
            "do_sample": False,
            "enable_thinking": False,
            "scoring_mode": scoring_mode,
            "dtype": dtype_name,
            "torch": torch_version,
            "python": platform.python_version(),
            "prompt_id": prompt_id,
            "prompt_note": "original choices retained; no masked-branch reminder",
            "vanilla": "hook_bypass",
            "frameweave": "inject_captured_delta",
        }
        return fingerprint16(payload), payload

    def expected_inputs(scoring_mode: str) -> dict[tuple[str, str], str]:
        expected = {}
        for sample in selected:
            prompt = format_prompt(sample)
            assert_prompt_keeps_choices(prompt, sample["choices"], sample["letters"])
            digest = input_fingerprint(
                sample,
                prompt,
                scoring_mode,
                int(args.min_pixels),
                int(args.max_pixels),
            )
            for model_name in MODELS:
                expected[(sample["sample_id"], model_name)] = digest
        return expected

    if (
        args.resume
        and existing
        and existing.get("source_manifest_sha256") == source_hash
        and existing.get("checkpoint_sha256") == ckpt_hash
        and existing.get("prompt_id") == prompt_id
    ):
        sticky_mode = str(existing.get("scoring_mode") or "")
        sticky_fp, _ = fingerprint_for(
            sticky_mode,
            str(existing.get("dtype") or ""),
            str(existing.get("torch") or ""),
        )
        if sticky_fp == existing.get("run_fingerprint"):
            plan = resume_plan(
                load_prediction_rows(pred_csv),
                load_jsonl(score_path),
                sticky_fp,
                expected_inputs(sticky_mode),
            )
            needed = {(sample["sample_id"], model_name) for sample in selected for model_name in MODELS}
            if not plan["foreign"] and needed and needed <= set(plan["done"]):
                print("Resume found a complete score cache. Writing the report.", flush=True)
                _finish(args, output_dir, manifest, selected, sticky_fp, sticky_mode)
                return

    print(f"Loading {MODEL_ID}. Several minutes of silence is expected.", flush=True)
    cfg = ExperimentConfig(run_mode="eval_herbench", root=Path(args.root))
    if args.model_dir:
        cfg.model_dir = Path(args.model_dir)
    model, tokenizer = load_ovis(cfg)
    print("Ovis2.5-9B loaded.", flush=True)
    model.eval()
    freeze_backbone(model)
    install_reljoin_hook(model)
    dtype = model.dtype
    text_tokenizer = getattr(model, "text_tokenizer", None) or tokenizer
    from reljoin.ovis_hooks import ovis_hidden_size

    joining = load_reljoin_module(ckpt, ovis_hidden_size(model), model.device, variant="Hybrid")
    encoded = _encode_letters(text_tokenizer, letters)
    natural_mode = scoring_mode_for_letters(encoded)
    if args.scoring_mode == SCORING_FIRST and natural_mode == SCORING_SEQUENCE:
        raise ProtocolError("letters are not one distinct token; first_token_letter is unavailable")
    if args.scoring_mode == "auto":
        scoring_mode = natural_mode
        if (
            existing
            and existing.get("scoring_mode") == SCORING_SEQUENCE
            and existing.get("source_manifest_sha256") == source_hash
            and existing.get("checkpoint_sha256") == ckpt_hash
            and existing.get("prompt_id") == prompt_id
            and natural_mode == SCORING_FIRST
        ):
            scoring_mode = SCORING_SEQUENCE
            print("Resuming the sequence_logprob table from the previous switch.", flush=True)
    else:
        scoring_mode = str(args.scoring_mode)

    def activate(scoring_mode: str) -> tuple[str, dict]:
        fingerprint, payload = fingerprint_for(scoring_mode, str(dtype), torch.__version__)
        payload["run_fingerprint"] = fingerprint
        payload["letter_token_ids"] = encoded
        payload["learned_missing"] = len(manifest_result.learned_missing_ids)
        payload["learned_extra"] = len(manifest_result.learned_extra_ids)
        if existing and existing.get("run_fingerprint") != fingerprint:
            raise ProtocolError(
                "output directory belongs to a different checkpoint, prompt, or scoring mode. "
                "Use a new --output-dir."
            )
        write_json(config_path, payload)
        return fingerprint, payload

    fingerprint, _payload = activate(scoring_mode)
    print(f"Scoring mode {scoring_mode} fingerprint {fingerprint}", flush=True)

    def score_pass(scoring_mode: str, fingerprint: str) -> None:
        expected = expected_inputs(scoring_mode)
        plan = resume_plan(
            load_prediction_rows(pred_csv),
            load_jsonl(score_path),
            fingerprint,
            expected,
        )
        if plan["foreign"]:
            raise ProtocolError(
                "prediction cache fingerprint does not match this run. Use a new --output-dir."
            )
        done = set(plan["done"])
        for record in plan["rebuild"]:
            for row in predictions_from_score_record(record):
                append_csv(pred_csv, row)
            done.add((_text_id(record), str(record["model_variant"])))
            print(f"Rebuilt {record['sample_id']} {record['model_variant']} from the score cache", flush=True)

        herbench_root = Path(args.herbench_root)
        videos_root = Path(args.videos_root)
        stage_dir = Path(args.stage_video_dir) if args.stage_video_dir else None
        if stage_dir is not None:
            stage_dir.mkdir(parents=True, exist_ok=True)
        grouped: dict[str, list[dict]] = defaultdict(list)
        for sample in selected:
            resolved = resolve_video(sample["video_path"], herbench_root, videos_root)
            copied = dict(sample)
            copied["resolved_video_path"] = str(resolved) if resolved else ""
            grouped[copied["resolved_video_path"]].append(copied)

        for video_number, (video_path, video_rows) in enumerate(grouped.items(), start=1):
            pending = [
                sample
                for sample in video_rows
                if any((sample["sample_id"], model_name) not in done for model_name in MODELS)
            ]
            if not pending:
                print(f"Video {video_number}/{len(grouped)} skip", flush=True)
                continue
            print(
                f"Video {video_number}/{len(grouped)}: {Path(video_path).name if video_path else 'missing'} "
                f"pending={len(pending)}",
                flush=True,
            )
            staged = None
            read_path = Path(video_path) if video_path else None
            try:
                if read_path is not None and stage_dir is not None:
                    read_path = stage_one_video(read_path, stage_dir)
                    staged = read_path
                for sample in pending:
                    if not sample["resolved_video_path"]:
                        _write_model_errors(
                            pred_csv,
                            sample,
                            fingerprint,
                            scoring_mode,
                            ckpt_hash,
                            format_prompt(sample),
                            int(args.min_pixels),
                            int(args.max_pixels),
                            "video file was not found",
                            done,
                        )
                        continue
                    _score_sample(
                        model=model,
                        tokenizer=text_tokenizer,
                        joining=joining,
                        dtype=dtype,
                        sample=sample,
                        read_path=read_path,
                        format_prompt=format_prompt,
                        base_mod=base_mod,
                        encoded=encoded,
                        scoring_mode=scoring_mode,
                        fingerprint=fingerprint,
                        ckpt_hash=ckpt_hash,
                        min_pixels=int(args.min_pixels),
                        max_pixels=int(args.max_pixels),
                        max_new_tokens=int(args.max_new_tokens),
                        pred_csv=pred_csv,
                        score_path=score_path,
                        done=done,
                    )
            finally:
                if staged is not None and staged.exists():
                    staged.unlink()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    try:
        score_pass(scoring_mode, fingerprint)
    except ScoringFallback as exc:
        if scoring_mode != SCORING_FIRST or args.scoring_mode == SCORING_FIRST:
            print(f"First-token scoring stopped: {exc}", flush=True)
            _finish(args, output_dir, manifest, selected, fingerprint, scoring_mode)
            raise
        print(f"Switching the whole run to sequence_logprob: {exc}", flush=True)
        _archive_outputs(output_dir)
        existing = None
        scoring_mode = SCORING_SEQUENCE
        fingerprint, _payload = activate(scoring_mode)
        print(f"Scoring mode {scoring_mode} fingerprint {fingerprint}", flush=True)
        score_pass(scoring_mode, fingerprint)
    _finish(args, output_dir, manifest, selected, fingerprint, scoring_mode)


def _text_id(record: dict) -> str:
    return str(record.get("sample_id") or "")


def _sample_context(
    sample: dict,
    *,
    fingerprint: str,
    scoring_mode: str,
    ckpt_hash: str,
    prompt: str,
    min_pixels: int,
    max_pixels: int,
    model_variant: str,
) -> dict:
    return {
        "run_fingerprint": fingerprint,
        "input_fingerprint": input_fingerprint(sample, prompt, scoring_mode, min_pixels, max_pixels),
        "sample_id": sample["sample_id"],
        "video_id": sample["video_id"],
        "video_path": sample["video_path"],
        "task_type": sample["task_type"],
        "model_variant": model_variant,
        "abstention_letter": sample["abstention_letter"],
        "original_labels": json.dumps(sample["letters"]),
        "ground_truth": sample["ground_truth"],
        "scoring_mode": scoring_mode,
        "raw_native_output": "",
        "greedy_matches_unmasked": "false",
        "first_new_token_id": "",
        "checkpoint_sha256": ckpt_hash,
        "question": sample["question"],
        "choices_json": json.dumps(sample["choices"], ensure_ascii=False),
        "option_order": json.dumps(sample["option_order"]),
        "frame_timestamps": json.dumps(sample["frame_timestamps"]),
        "oracle_mode": sample["oracle_mode"],
        "oracle_padded": bool_cell(bool(sample["oracle_padded"])),
        "oracle_frame_times": json.dumps(sample["oracle_frame_times"]),
        "oracle_frame_indices": json.dumps(sample["oracle_frame_indices"]),
        "loaded_frame_indices": "",
        "tokens_per_frame": json.dumps(sample.get("tokens_per_frame") or []),
        "num_frames": sample["num_frames"],
        "runtime_seconds": "",
    }


def _write_model_errors(
    pred_csv: Path,
    sample: dict,
    fingerprint: str,
    scoring_mode: str,
    ckpt_hash: str,
    prompt: str,
    min_pixels: int,
    max_pixels: int,
    message: str,
    done: set,
) -> None:
    for model_name in MODELS:
        if (sample["sample_id"], model_name) in done:
            continue
        context = _sample_context(
            sample,
            fingerprint=fingerprint,
            scoring_mode=scoring_mode,
            ckpt_hash=ckpt_hash,
            prompt=prompt,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            model_variant=model_name,
        )
        for row in error_pair_records(context, message):
            append_csv(pred_csv, row)
        print(f"  ERROR {sample['sample_id']} {model_name}: {message}", flush=True)


def _score_sample(
    *,
    model,
    tokenizer,
    joining,
    dtype,
    sample: dict,
    read_path: Path,
    format_prompt,
    base_mod,
    encoded: dict[str, list[int]],
    scoring_mode: str,
    fingerprint: str,
    ckpt_hash: str,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
    pred_csv: Path,
    score_path: Path,
    done: set,
) -> None:
    prompt = format_prompt(sample)
    assert_prompt_keeps_choices(prompt, sample["choices"], sample["letters"])
    times = [float(item) for item in sample["frame_timestamps"]]
    frames, indices, _meta = load_oracle_pil_frames(read_path, times, 0.0)
    if len(frames) != len(times):
        raise ProtocolError(f"{sample['sample_id']}: loaded {len(frames)} frames from {len(times)} timestamps")
    question_hidden = encode_question_hidden(
        model,
        tokenizer,
        sample["question"],
        sample["choices"],
        model.device,
        dtype,
    )
    timestamps = torch.tensor([times], device=model.device, dtype=dtype)
    frame_mask = torch.ones(1, len(frames), device=model.device, dtype=torch.bool)
    begin_prepare_cache()
    try:
        prepared = preprocess_ovis_inputs(model, frames, prompt, min_pixels, max_pixels)
        capture = capture_delta(
            model,
            joining,
            prepared[0],
            prepared[1],
            prepared[2],
            question_hidden,
            timestamps,
            frame_mask,
        )
        for model_name in MODELS:
            if (sample["sample_id"], model_name) in done:
                continue
            context = _sample_context(
                sample,
                fingerprint=fingerprint,
                scoring_mode=scoring_mode,
                ckpt_hash=ckpt_hash,
                prompt=prompt,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                model_variant=model_name,
            )
            context["loaded_frame_indices"] = json.dumps(indices)
            context["tokens_per_frame"] = json.dumps(capture["tokens_per_frame"])
            started = time.time()
            try:
                records, score_record = _score_model(
                    model=model,
                    joining=joining,
                    frames=frames,
                    prompt=prompt,
                    question_hidden=question_hidden,
                    timestamps=timestamps,
                    frame_mask=frame_mask,
                    prepared=prepared,
                    delta=capture["delta"],
                    sample=sample,
                    model_name=model_name,
                    encoded=encoded,
                    scoring_mode=scoring_mode,
                    base_mod=base_mod,
                    max_pixels=max_pixels,
                    min_pixels=min_pixels,
                    max_new_tokens=max_new_tokens,
                    context=context,
                )
            except (ScoringFallback, ProtocolError):
                raise
            except Exception as exc:
                elapsed = f"{time.time() - started:.3f}"
                for row in error_pair_records(context, repr(exc)):
                    row["runtime_seconds"] = elapsed
                    append_csv(pred_csv, row)
                print(f"  ERROR {sample['sample_id']} {model_name}: {exc!r}", flush=True)
                continue
            elapsed = f"{time.time() - started:.3f}"
            for row in records:
                row["runtime_seconds"] = elapsed
                append_csv(pred_csv, row)
            append_jsonl(score_path, score_record)
            done.add((sample["sample_id"], model_name))
            print(
                f"  {model_name} {sample['sample_id']} "
                f"unmasked={records[0]['predicted_letter']} masked={records[1]['predicted_letter']} "
                f"gt={sample['ground_truth']} abs={sample['abstention_letter']} {elapsed}s",
                flush=True,
            )
    finally:
        end_prepare_cache()
        del frames


def _score_model(
    *,
    model,
    joining,
    frames,
    prompt,
    question_hidden,
    timestamps,
    frame_mask,
    prepared,
    delta,
    sample: dict,
    model_name: str,
    encoded: dict[str, list[int]],
    scoring_mode: str,
    base_mod,
    max_pixels: int,
    min_pixels: int,
    max_new_tokens: int,
    context: dict,
) -> tuple[list[dict], dict]:
    flags = generation_flags(model_name)
    inject = delta if flags["inject"] else None
    letters = list(sample["letters"])
    common = dict(
        bypass=flags["bypass"],
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        max_new_tokens=max_new_tokens,
        eval_mode="full",
        inject_delta=inject,
        prepared=prepared,
    )
    if scoring_mode == SCORING_FIRST:
        try:
            meta = generate_ovis(
                model,
                frames,
                prompt,
                joining,
                question_hidden,
                timestamps,
                frame_mask,
                return_meta=True,
                return_scores=True,
                **common,
            )
        except TypeError as exc:
            raise ScoringFallback(f"generate did not return scores: {exc}") from exc
        step = meta.get("first_step_logits")
        if step is None:
            raise ScoringFallback(f"{sample['sample_id']} {model_name}: missing first-step scores")
        scores = {letter: float(step[int(encoded[letter][0])].item()) for letter in letters}
        raw = str(meta["text"])
        first_id = meta.get("first_new_token_id")
        pred_unmasked = argmax_letter(scores, letters)
        parsed, parsed_ok = _parse_letter(raw, letters, base_mod)
        matches = greedy_matches_unmasked(
            first_token_id=first_id,
            letter_token_ids=encoded,
            pred_unmasked=pred_unmasked,
            parsed_letter=parsed,
            parsed_ok=parsed_ok,
        )
        if not matches:
            raise ScoringFallback(
                f"{sample['sample_id']} {model_name}: greedy token {first_id} / {parsed!r} "
                f"disagrees with unmasked argmax {pred_unmasked}"
            )
    else:
        meta = generate_ovis(
            model,
            frames,
            prompt,
            joining,
            question_hidden,
            timestamps,
            frame_mask,
            return_meta=True,
            return_scores=False,
            **common,
        )
        raw = str(meta["text"])
        first_id = meta.get("first_new_token_id")
        single = all(len(encoded[letter]) == 1 for letter in letters)
        if single:
            step = _next_token_logits(model, prepared[0], prepared[1], prepared[2])
            scores = _letter_logprobs(step, encoded, letters)
        else:
            scores = {
                letter: _continuation_logprob(
                    model, prepared[0], prepared[1], prepared[2], encoded[letter]
                )
                for letter in letters
            }
        pred_unmasked = argmax_letter(scores, letters)
        parsed, parsed_ok = _parse_letter(raw, letters, base_mod)
        matches = greedy_matches_unmasked(
            first_token_id=first_id,
            letter_token_ids=encoded,
            pred_unmasked=pred_unmasked,
            parsed_letter=parsed,
            parsed_ok=parsed_ok,
        )
    context["raw_native_output"] = raw
    context["first_new_token_id"] = "" if first_id is None else str(first_id)
    context["greedy_matches_unmasked"] = bool_cell(matches)
    records = prediction_pair_records(
        scores=scores,
        letters=letters,
        abstention_letter=sample["abstention_letter"],
        ground_truth=sample["ground_truth"],
        context=context,
    )
    score_record = {
        "run_fingerprint": context["run_fingerprint"],
        "input_fingerprint": context["input_fingerprint"],
        "sample_id": sample["sample_id"],
        "model_variant": model_name,
        "scoring_mode": scoring_mode,
        "scores": encode_scores(scores),
        "letters": letters,
        "abstention_letter": sample["abstention_letter"],
        "ground_truth": sample["ground_truth"],
        "context": context,
    }
    return records, score_record


def _finish(args, output_dir: Path, manifest: list[dict], selected: list[dict], fingerprint: str, scoring_mode: str) -> None:
    rows = load_prediction_rows(output_dir / "predictions.csv")
    summary, stats = build_reports(
        selected,
        rows,
        n_manifest=len(manifest),
        run_fingerprint=fingerprint,
        scoring_mode=scoring_mode,
        n_resamples=int(args.bootstrap),
        seed=int(args.analysis_seed),
        full_manifest=manifest,
    )
    write_reports(output_dir, summary, stats)
    if summary["incomplete"]:
        print(f"INCOMPLETE: {summary['incomplete_reasons']}", flush=True)
        os._exit(2)
    gains = summary["gains"]
    print(
        f"G_masked {gains['G_masked_pp']:.2f} pp, "
        f"difference of gains {gains['difference_pp']:.2f} pp, "
        f"videos {summary['n_videos']}",
        flush=True,
    )
    print("SMOKE OK" if summary["population"] == "subset" else "FULL OK", flush=True)


if __name__ == "__main__":
    try:
        run(build_parser().parse_args())
    except Exception:
        import traceback

        traceback.print_exc()
        os._exit(1)
    os._exit(0)
