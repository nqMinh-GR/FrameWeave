#!/usr/bin/env python3
"""HERBench decoder-bias diagnostics for Vanilla, DTR, FrameWeave Hybrid v6.

Eval-only. Same oracle 8-frame / prompt protocol as run_herbench_oracle_dtr.py.
FrameWeave uses last-token h_q and in-place X̃ (no join tokens).
Default comparison: vanilla vs DTR vs hybrid.

Three experiments:
  1. A–E logits at the answer position
  2. Cyclic option permutation (ABCDE → BCDEA → …)
  3. Option-content likelihood (not letter NLL)
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import shutil
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from run_herbench_oracle_reljoin import (  # noqa: E402
    METHOD_TO_VARIANT,
    dtr_runner,
    generate_reljoin,
    load_reljoin_module,
)

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
    "generated_letter",
    "raw_output",
    "is_valid",
    "letter_correct",
    "logit_argmax_letter",
    "logit_correct",
    "p_A",
    "p_B",
    "p_C",
    "p_D",
    "p_E",
    "z_A",
    "z_B",
    "z_C",
    "z_D",
    "z_E",
    "entropy",
    "margin_zA_minus_zy",
    "gt_rank",
    "gt_top1",
    "gt_top2",
    "content_s_A",
    "content_s_B",
    "content_s_C",
    "content_s_D",
    "content_s_E",
    "content_pred",
    "content_correct",
    "perm_pred_letters_json",
    "perm_contents_json",
    "content_consistency",
    "first_position_rate",
    "num_frames",
    "frame_timestamps",
    "oracle_mode",
    "oracle_padded",
    "oracle_frame_times",
    "oracle_frame_indices",
    "frame_paths_or_ids",
    "runtime_seconds",
    "status",
    "error_message",
    "error",
]


def append_record(path: Path, record: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_ORDER, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in FIELD_ORDER})


def smoke_rows(rows: list[dict], n: int) -> list[dict]:
    seen = set()
    out: list[dict] = []
    for r in rows:
        vp = r.get("resolved_video_path") or ""
        if vp in seen:
            continue
        seen.add(vp)
        out.append(r)
        if len(out) >= n:
            return out
    ids = {r["question_id"] for r in out}
    for r in rows:
        if r["question_id"] in ids:
            continue
        out.append(r)
        ids.add(r["question_id"])
        if len(out) >= n:
            break
    return out[:n]


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    if n <= 0 or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[str(r.get("task_type") or "unknown")].append(r)
    keys = sorted(by_task)
    total = max(len(rows), 1)
    alloc = {k: min(len(by_task[k]), max(1, int(math.floor(n * len(by_task[k]) / total)))) for k in keys}
    while sum(alloc.values()) > n:
        k_max = max(keys, key=lambda k: alloc[k])
        if alloc[k_max] <= 1:
            break
        alloc[k_max] -= 1
    queues = {}
    for k in keys:
        pool = list(by_task[k])
        rng.shuffle(pool)
        queues[k] = pool
    out: list[dict] = []
    for k in keys:
        take = min(alloc[k], len(queues[k]))
        out.extend(queues[k][:take])
        queues[k] = queues[k][take:]
    rest: list[dict] = []
    for k in keys:
        rest.extend(queues[k])
    rng.shuffle(rest)
    need = n - len(out)
    if need > 0:
        out.extend(rest[:need])
    return out[:n]


def parse_generated_letter(raw: str, parse_choice, parse_choice_strict) -> tuple[str, bool]:
    pred = parse_choice(raw)
    is_valid = pred != "ERROR"
    if not is_valid:
        pred2, ok = parse_choice_strict(raw)
        return pred2, ok
    return pred, True


def run(args: argparse.Namespace) -> None:
    import time

    from reljoin.videollava_hooks import (
        configure_reljoin,
        encode_question_hidden,
        freeze_backbone,
        install_reljoin_hook,
        set_reljoin_context,
    )
    from reljoin.diagnostics import (
        LETTERS,
        content_pred_letter,
        cyclic_rotations,
        letter_logit_stats,
        letter_to_content,
        letter_token_ids,
        next_token_logits,
        nonempty_choices,
        option_content_loglikelihoods,
        parse_choice_strict,
        perm_payload_from_run,
        summarize_decoder_diagnostics,
    )

    expected_prefix = str(Path(args.venv_site).resolve()) if args.venv_site else ""
    dtr_runner.patch_torchvision_functional_tensor()
    versions = dtr_runner.assert_venv_packages(expected_prefix)
    print("Package check OK:", json.dumps(versions, indent=2))

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

    if dtr_runner.sha256_file(dtr_repo / "run_inference_dtr.py") != args.dtr_sha256:
        raise RuntimeError("DTR source SHA256 mismatch")
    base_src = Path(args.herbench_base_vlm)
    acc_src = Path(args.herbench_accuracy)
    if dtr_runner.sha256_file(base_src) != args.herbench_base_sha256:
        raise RuntimeError("HERBench base_vlm.py SHA256 mismatch")
    if dtr_runner.sha256_file(acc_src) != args.herbench_acc_sha256:
        raise RuntimeError("HERBench calculate_accuracy.py SHA256 mismatch")

    sys.path.insert(0, str(dtr_repo.resolve()))
    import run_inference_dtr as dtr
    from transformers.models.llama.modeling_llama import LlamaAttention

    base_mod = dtr_runner.load_module_from_file("herbench_base_pinned", base_src)

    rows = dtr_runner.load_rows(herbench_root, videos_root, args.herbench_revision)
    missing = [r["question_id"] for r in rows if not r["resolved_video_path"]]
    if missing:
        n_mp4 = len(list(Path(videos_root).rglob("*.mp4"))) if Path(videos_root).exists() else 0
        sample_vp = next((str(r.get("video_path") or "") for r in rows if r.get("video_path")), "")
        raise RuntimeError(
            f"{len(missing)} questions missing videos; first={missing[:5]}. "
            f"Found {n_mp4} mp4 under --videos-root={videos_root}. "
            f"Example annotation video_path={sample_vp!r}."
        )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods:
        methods = ["vanilla", "dtr", "hybrid"]
    methods = ["hybrid" if m == "reljoin" else m for m in methods]
    methods = list(dict.fromkeys(methods))
    unknown = [m for m in methods if m not in {"vanilla", "dtr"} and m not in METHOD_TO_VARIANT]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Known: vanilla, dtr, {list(METHOD_TO_VARIANT)}")
    if "hybrid" not in methods:
        methods = methods + ["hybrid"]
    print("Decoder-diag methods:", methods)

    model, tokenizer, video_processor = dtr.load_model_with_global_rebalance(
        model_path=str(model_dir),
        device=0,
        start_layer=18,
        end_layer=31,
        alpha=0.5,
        beta=0.4,
    )
    model.eval()
    freeze_backbone(model)
    try:
        import decord  # noqa: F401
    except ImportError:
        video_processor.config.vision_config.video_decode_backend = "opencv"

    vanilla_fn = LlamaAttention.forward
    dtr_fn = dtr.llama_new_forward_with_global_rebalance
    dtr_runner.set_attention_condition(model, dtr, vanilla_fn, dtr_fn, use_dtr=False)

    num_video_frames = int(model.get_video_tower().config.num_frames)
    if num_video_frames != 8:
        raise RuntimeError(f"Expected 8 native frames, found {num_video_frames}")

    install_reljoin_hook(model)
    dtype = torch.float16
    llm_dim = model.config.hidden_size
    token_ids = letter_token_ids(tokenizer, n_choices=5)
    print("Letter token ids:", token_ids)

    joining_modules = {}
    ckpt_root = Path(args.reljoin_ckpt)
    for m in methods:
        if m in {"vanilla", "dtr"}:
            continue
        vname = METHOD_TO_VARIANT[m]
        cand = ckpt_root / f"{vname}.pt" if ckpt_root.is_dir() else ckpt_root
        if not Path(cand).is_file():
            raise FileNotFoundError(
                f"Missing checkpoint for method={m} variant={vname}: {cand}"
            )
        joining_modules[m] = load_reljoin_module(cand, llm_dim, model.device, vname)

    def format_prompt(question: str, choices: list[str]) -> str:
        return base_mod.BaseVLM._format_prompt(None, question, choices)

    def parse_choice(text: str) -> str:
        return base_mod.BaseVLM.extract_answer_choice(None, str(text))

    if args.smoke:
        subset_tag = f"smoke{args.smoke_n}"
        pending = smoke_rows(rows, args.smoke_n)
        print(f"SMOKE mode: {len(pending)} questions × {len(methods)} methods")
    elif args.max_samples and args.max_samples > 0:
        subset_tag = f"n{args.max_samples}"
        pending = stratified_sample(rows, args.max_samples, args.seed)
        print(
            f"DIAG subset: {len(pending)} / {len(rows)} questions × {len(methods)} methods "
            f"(seed={args.seed})"
        )
    else:
        subset_tag = "full"
        pending = rows
        print(f"FULL mode: {len(pending)} questions × {len(methods)} methods")

    fingerprint_payload = {
        "protocol": "herbench_lite_v2_oracle8_decoder_diag_frameweave_v6_videollava",
        "dataset_revision": args.herbench_revision,
        "dtr_commit": args.dtr_commit,
        "model_revision": args.model_revision,
        "num_frames": num_video_frames,
        "methods": methods,
        "reljoin_ckpt": args.reljoin_ckpt,
        "subset": subset_tag,
        "letter_logits": bool(args.letter_logits),
        "cyclic_perm": bool(args.cyclic_perm),
        "content_likelihood": bool(args.content_likelihood),
        "perm_use_generate": bool(args.perm_use_generate),
        "seed": int(args.seed),
        "method": "FrameWeave-Hybrid",
        "method_version": "6.0.0",
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    tag = f"decoderdiag_{fingerprint}"
    pairs_csv = out_dir / f"diagnostics_{tag}.csv"
    persist_csv = persist_dir / pairs_csv.name

    if persist_csv.exists() and (
        not pairs_csv.exists() or persist_csv.stat().st_mtime > pairs_csv.stat().st_mtime
    ):
        shutil.copy2(persist_csv, pairs_csv)

    completed: set[tuple[str, str]] = set()
    if pairs_csv.exists():
        prev = pd.read_csv(pairs_csv, on_bad_lines="error")
        for _, r in prev.iterrows():
            if str(r.get("status")) != "runtime_error":
                completed.add((str(r["question_id"]), str(r["method"])))
        print(f"Resume: {len(completed)} completed (question, method) pairs")

    clip_cache = dtr_runner.OracleClipCache(cache_dir)
    oracle_seed_base = 20260811
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in pending:
        groups[r["resolved_video_path"]].append(r)

    def encode_h(question: str, choices: list[str], method: str):
        return encode_question_hidden(
            model,
            tokenizer,
            question,
            choices,
            model.device,
            dtype,
            use_raw_mean=False,
        )

    def set_method_context(method: str, question_hidden=None, timestamps=None, frame_mask=None) -> None:
        if method in {"vanilla", "dtr"}:
            configure_reljoin(None, enabled=False, bypass=True)
            dtr_runner.set_attention_condition(
                model, dtr, vanilla_fn, dtr_fn, use_dtr=(method == "dtr")
            )
            return
        dtr_runner.set_attention_condition(model, dtr, vanilla_fn, dtr_fn, use_dtr=False)
        configure_reljoin(joining_modules[method], enabled=True, bypass=False)
        set_reljoin_context(
            question_hidden=question_hidden,
            timestamps=timestamps,
            frame_mask=frame_mask,
        )

    def generate_text(method, video_tensor, prompt, question_hidden, timestamps, frame_mask):
        if method == "vanilla":
            configure_reljoin(None, enabled=False, bypass=True)
            return dtr_runner.generate_once(
                model, tokenizer, video_tensor, prompt, num_video_frames,
                dtr, vanilla_fn, dtr_fn, use_dtr=False,
            )
        if method == "dtr":
            configure_reljoin(None, enabled=False, bypass=True)
            return dtr_runner.generate_once(
                model, tokenizer, video_tensor, prompt, num_video_frames,
                dtr, vanilla_fn, dtr_fn, use_dtr=True,
            )
        return generate_reljoin(
            model, tokenizer, video_tensor, prompt, num_video_frames,
            joining_modules[method], question_hidden, timestamps, frame_mask,
        )

    n_pending_pairs = sum(1 for r in pending for m in methods if (r["question_id"], m) not in completed)
    print(f"Pairs to run: {n_pending_pairs}")

    for video_number, (video_path, video_rows) in enumerate(groups.items(), start=1):
        print(f"Video {video_number}/{len(groups)}: {Path(video_path).name}")
        for row in video_rows:
            spans, mode_hint = dtr_runner.extract_evidence_spans(row["metadata"])
            times, padded, mode = dtr_runner.choose_oracle_times(
                spans, row["duration"], num_frames=num_video_frames
            )
            if mode_hint == "uniform_fallback":
                mode = "uniform_fallback"
                padded = False
            clip_path, indices, _meta = clip_cache.get_or_build(
                Path(video_path),
                str(row.get("video_id") or Path(video_path).stem),
                times,
                row["duration"],
            )
            seed = oracle_seed_base + int(
                hashlib.sha256(row["question_id"].encode()).hexdigest()[:8], 16
            )
            video_tensor = dtr_runner.prepare_video_tensor(
                video_processor, clip_path, model, seed=seed
            )
            choices = list(row["choices"])
            active = nonempty_choices(choices)
            n_ch = max(len(active), 1)
            question = row["question"]
            prompt = format_prompt(question, choices)
            answer = str(row["answer"] or "").strip().upper()
            option_order = list(LETTERS[:n_ch])
            timestamps = torch.tensor([times], device=model.device, dtype=dtype)
            frame_mask = torch.ones(1, num_video_frames, device=model.device, dtype=torch.bool)

            for method in methods:
                key = (row["question_id"], method)
                if key in completed:
                    continue
                t0 = time.time()
                record = {
                    "run_fingerprint": fingerprint,
                    "sample_id": row["question_id"],
                    "question_id": row["question_id"],
                    "video_id": row.get("video_id", ""),
                    "video_path": row.get("video_path", ""),
                    "task_type": row.get("task_type", ""),
                    "method": method,
                    "question": question,
                    "option_order": json.dumps(option_order),
                    "choices_json": json.dumps(choices, ensure_ascii=False),
                    "ground_truth": answer,
                    "answer": answer,
                    "num_frames": num_video_frames,
                    "frame_timestamps": json.dumps(times),
                    "oracle_mode": mode,
                    "oracle_padded": bool(padded),
                    "oracle_frame_times": json.dumps(times),
                    "oracle_frame_indices": json.dumps(indices),
                    "frame_paths_or_ids": str(clip_path),
                    "status": "runtime_error",
                    "error_message": "",
                    "error": "",
                }
                try:
                    h_q = encode_h(question, choices, method)

                    def set_ctx(_h=h_q):
                        set_method_context(
                            method,
                            question_hidden=_h,
                            timestamps=timestamps,
                            frame_mask=frame_mask,
                        )

                    raw = generate_text(
                        method, video_tensor, prompt, h_q, timestamps, frame_mask
                    )
                    gen_letter, gen_valid = parse_generated_letter(
                        raw, parse_choice, parse_choice_strict
                    )
                    record["raw_output"] = raw
                    record["generated_letter"] = gen_letter
                    record["is_valid"] = bool(gen_valid)
                    record["letter_correct"] = bool(gen_valid and gen_letter == answer)

                    if args.letter_logits:
                        set_ctx()
                        z = next_token_logits(
                            model, tokenizer, video_tensor, prompt, num_video_frames, set_ctx
                        )
                        stats = letter_logit_stats(z, token_ids, answer, n_choices=n_ch)
                        record.update(stats)
                        record["logit_argmax_letter"] = stats["argmax_letter"]
                        record["logit_correct"] = bool(stats["argmax_letter"] == answer)
                    else:
                        record["logit_argmax_letter"] = ""
                        record["logit_correct"] = False

                    if args.cyclic_perm:
                        rot_letters: list[str] = []
                        rot_contents: list[str] = []
                        rotations = cyclic_rotations(active)
                        for _k, rotated in rotations:
                            rot_prompt = format_prompt(question, rotated)
                            rot_h = encode_h(question, rotated, method)

                            def set_rot_ctx(_h=rot_h):
                                set_method_context(
                                    method,
                                    question_hidden=_h,
                                    timestamps=timestamps,
                                    frame_mask=frame_mask,
                                )

                            if args.perm_use_generate:
                                raw_k = generate_text(
                                    method,
                                    video_tensor,
                                    rot_prompt,
                                    rot_h,
                                    timestamps,
                                    frame_mask,
                                )
                                pred_k, ok_k = parse_generated_letter(
                                    raw_k, parse_choice, parse_choice_strict
                                )
                                if not ok_k:
                                    pred_k = ""
                            else:
                                z_k = next_token_logits(
                                    model,
                                    tokenizer,
                                    video_tensor,
                                    rot_prompt,
                                    num_video_frames,
                                    set_rot_ctx,
                                )
                                pred_k = letter_logit_stats(
                                    z_k, token_ids, answer, n_choices=len(rotated)
                                )["argmax_letter"]
                            rot_letters.append(pred_k)
                            rot_contents.append(letter_to_content(pred_k, rotated))
                        record.update(
                            perm_payload_from_run(rot_letters, rot_contents, n_rot=len(rotations))
                        )

                    if args.content_likelihood:
                        set_ctx()
                        scores = option_content_loglikelihoods(
                            model,
                            tokenizer,
                            video_tensor,
                            prompt,
                            choices,
                            num_video_frames,
                            set_ctx,
                        )
                        for L in LETTERS:
                            record[f"content_s_{L}"] = scores.get(L, "")
                        pred_c = content_pred_letter(scores)
                        record["content_pred"] = pred_c
                        record["content_correct"] = bool(pred_c and pred_c == answer)

                    record["runtime_seconds"] = time.time() - t0
                    record["status"] = "ok"
                except Exception as exc:
                    record["error_message"] = repr(exc)
                    record["error"] = traceback.format_exc()
                    record["status"] = "runtime_error"
                    record["runtime_seconds"] = time.time() - t0
                    print(f"ERROR {row['question_id']} {method}: {exc!r}")
                    traceback.print_exc()
                finally:
                    dtr_runner.set_attention_condition(
                        model, dtr, vanilla_fn, dtr_fn, use_dtr=False
                    )
                    configure_reljoin(None, enabled=False, bypass=True)

                append_record(pairs_csv, record)
                completed.add(key)

            del video_tensor
            gc.collect()
            torch.cuda.empty_cache()

        shutil.copy2(pairs_csv, persist_csv)

    df = pd.read_csv(pairs_csv)
    summary = summarize_decoder_diagnostics(df, methods=methods)
    payload = {
        "run_fingerprint": fingerprint,
        "manifest": fingerprint_payload,
        "summary": summary,
        "n_rows": int(len(df)),
        "n_ok": int((df["status"] == "ok").sum()) if "status" in df.columns else int(len(df)),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "pairs_csv": str(pairs_csv),
        "letter_token_ids": token_ids,
    }
    summary_path = out_dir / f"summary_{tag}.json"
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    shutil.copy2(summary_path, persist_dir / summary_path.name)
    shutil.copy2(pairs_csv, persist_csv)

    print(json.dumps(summary, indent=2, default=str))
    if args.smoke:
        print("SMOKE OK")
        cols = [
            c
            for c in [
                "question_id",
                "method",
                "generated_letter",
                "logit_argmax_letter",
                "content_pred",
                "ground_truth",
                "p_A",
                "gt_rank",
            ]
            if c in df.columns
        ]
        print(df[cols].to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(_THIS))
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
    p.add_argument(
        "--methods",
        default="vanilla,dtr,hybrid",
        help="Default Vanilla vs DTR vs FrameWeave Hybrid.",
    )
    p.add_argument(
        "--reljoin-ckpt",
        default=str(_THIS / "outputs_reljoin_v6_videollava" / "checkpoints" / "best_external_val"),
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-n", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=0, help="0 = all rows (unless --smoke)")
    p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--letter-logits", dest="letter_logits", action="store_true", default=True)
    p.add_argument("--skip-letter-logits", dest="letter_logits", action="store_false")
    p.add_argument("--cyclic-perm", dest="cyclic_perm", action="store_true", default=True)
    p.add_argument("--skip-cyclic-perm", dest="cyclic_perm", action="store_false")
    p.add_argument(
        "--content-likelihood", dest="content_likelihood", action="store_true", default=True
    )
    p.add_argument("--skip-content-likelihood", dest="content_likelihood", action="store_false")
    p.add_argument("--perm-use-generate", action="store_true", default=False)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
