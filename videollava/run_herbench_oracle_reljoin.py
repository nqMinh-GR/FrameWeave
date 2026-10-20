#!/usr/bin/env python3
"""HERBench oracle evaluation for FrameWeave Hybrid v6 on Video-LLaVA-7B.

Default config is lite_v2 (methods vanilla,dtr,hybrid,dtr_hybrid). Pass
`--herbench-config full --skip-unresolved --methods vanilla,hybrid` for the
Drive full pack (27,566 / 27,631; uid_vid_00109 absent from the official tar).
Full eval does not run method `dtr` or `dtr_hybrid`.
`dtr_hybrid` is Hybrid.pt plus DTR attention (layers 18–31) on the same clip.

Oracle 8-frame clips follow run_herbench_oracle_dtr.py. FrameWeave writes X̃ in
place; Vanilla is hook bypass.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
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
    """Copy without blocking eval if Google Drive / NFS stalls."""
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
        _PERSIST_DISABLED = True
        print(
            f"WARN: persist copy timed out after {timeout}s ({src} -> {dst}); "
            "disabling further Drive/NFS persist this run. Local CSV is authoritative.",
            flush=True,
        )
        return False
    if err:
        print(f"WARN: persist copy failed: {err[0]!r}", flush=True)
        return False
    return True

# Colab injects PYTHONPATH → system torch. Drop it before importing packages so
# /content/dtr_env/bin/python actually uses the venv site-packages.
os.environ.pop("PYTHONPATH", None)
os.environ.pop("PYTHONHOME", None)
# Interpreter already copied PYTHONPATH into sys.path at startup — strip Colab sites.
_sys_path = []
for _p in sys.path:
    _pl = str(_p).replace("\\", "/")
    _colab_site = (
        "/dist-packages" in _pl
        or _pl == "/env/python"
        or "/usr/local/lib/python" in _pl
    )
    if _colab_site and "dtr_env" not in _pl:
        continue
    _sys_path.append(_p)
sys.path[:] = _sys_path

import pandas as pd
import torch

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))


def _load_dtr_runner():
    path = _THIS / "run_herbench_oracle_dtr.py"
    spec = importlib.util.spec_from_file_location("run_herbench_oracle_dtr", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


dtr_runner = _load_dtr_runner()

from reljoin.config import (
    METHOD_VERSION,
    METHOD_TO_VARIANT,
    VARIANT_TO_METHOD,
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
    "frame_paths_or_ids",
    "ground_truth_logprob",
    "join_gain",
    "loo_degradation",
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


def discover_methods_from_ckpt_dir(ckpt_dir: Path) -> tuple[list[str], dict[str, Path]]:
    """vanilla + residual variants whose .pt files exist under ckpt_dir."""
    ckpt_dir = Path(ckpt_dir)
    found: dict[str, Path] = {}
    methods = ["vanilla"]
    if not ckpt_dir.is_dir():
        return methods, found
    for variant, method in VARIANT_TO_METHOD.items():
        path = ckpt_dir / f"{variant}.pt"
        if path.is_file():
            found[method] = path
            methods.append(method)
    return methods, found


def find_extract_root(root: Path) -> Path:
    """`/content/outputs_reljoin_v6_videollava` (or `{root}/outputs_reljoin_v6_videollava`)."""
    extract = Path(root) / "outputs_reljoin_v6_videollava"
    if extract.is_dir():
        return extract
    raise FileNotFoundError(
        f"Missing {extract}. Put folder outputs_reljoin_v6_videollava at "
        f"{Path(root) / 'outputs_reljoin_v6_videollava'}."
    )


def find_reljoin_pt(extract: Path) -> Path:
    """Find Hybrid.pt under outputs_reljoin_v6_videollava (any layout)."""
    extract = Path(extract)
    preferred = [
        extract / "checkpoints" / "best_external_val" / "Hybrid.pt",
        extract / "checkpoints" / "last" / "Hybrid.pt",
        extract / "checkpoints" / "Hybrid.pt",
        extract / "best_external_val" / "Hybrid.pt",
        extract / "last" / "Hybrid.pt",
        extract / "Hybrid.pt",
    ]
    for path in preferred:
        if path.is_file():
            return path
    hits = sorted(p for p in extract.rglob("Hybrid.pt") if p.is_file()) if extract.is_dir() else []
    if hits:
        print("Found Hybrid.pt at", hits[0])
        return hits[0]
    listing = []
    if extract.is_dir():
        listing = [p.name + ("/" if p.is_dir() else "") for p in sorted(extract.iterdir())]
        pts = [str(p.relative_to(extract)) for p in extract.rglob("*.pt")]
    else:
        pts = []
    raise FileNotFoundError(
        f"No Hybrid.pt under {extract}.\n"
        f"Top-level: {listing or 'MISSING DIR'}\n"
        f"*.pt: {pts or 'none'}"
    )


def stage_ckpt_dir(extract_dir: Path) -> Path:
    """Directory that contains Hybrid.pt."""
    reljoin_pt = find_reljoin_pt(extract_dir)
    ckpt_dir = reljoin_pt.parent
    last = Path(extract_dir) / "checkpoints" / "last"
    if ckpt_dir.resolve() != last.resolve():
        for variant in VARIANT_TO_METHOD:
            name = f"{variant}.pt"
            if not (ckpt_dir / name).is_file() and (last / name).is_file():
                shutil.copy2(last / name, ckpt_dir / name)
                print(f"missing {name} → copy from last/")
    print("Using FrameWeave ckpt dir", ckpt_dir)
    return ckpt_dir


def load_reljoin_module(ckpt_path: Path, llm_dim: int, device, variant: str):
    from reljoin.reljoin_modules import build_joining_module

    module = build_joining_module(llm_dim, variant=variant).to(device=device, dtype=torch.float32)
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("reljoin", ckpt)
        if not isinstance(state, dict):
            raise RuntimeError(f"Unexpected RelJoin ckpt payload in {ckpt_path}: {type(state)}")
        try:
            module.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"RelJoin state_dict mismatch for variant={variant} from {ckpt_path}: {exc}"
            ) from exc
        print(f"Loaded FrameWeave ckpt {ckpt_path} for variant={variant}", flush=True)
    else:
        print(f"WARNING: no ckpt for {variant}; using random init (eval-only smoke)", flush=True)
    module.eval()
    return module


def generate_reljoin(
    model,
    tokenizer,
    video_tensor,
    prompt: str,
    num_video_frames: int,
    joining,
    question_hidden,
    timestamps,
    frame_mask,
    max_new_tokens: int = 32,
    eval_mode: str | None = None,
    bypass: bool = False,
) -> str:
    from reljoin.videollava_hooks import generate_videollava

    return generate_videollava(
        model,
        tokenizer,
        video_tensor,
        prompt,
        num_video_frames,
        joining=joining,
        question_hidden=question_hidden,
        timestamps=timestamps,
        frame_mask=frame_mask,
        max_new_tokens=max_new_tokens,
        eval_mode=eval_mode,
        bypass=bypass,
    )


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


def run(args: argparse.Namespace) -> None:
    import time

    from reljoin.videollava_hooks import (
        encode_question_hidden,
        freeze_backbone,
        install_reljoin_hook,
    )
    from reljoin.diagnostics import accuracy_reports, parse_choice_strict
    from reljoin.stats import compare_methods, save_statistical_tests

    expected_prefix = str(Path(args.venv_site).resolve()) if args.venv_site else ""
    dtr_runner.patch_torchvision_functional_tensor()
    versions = dtr_runner.assert_venv_packages(expected_prefix)
    print("Package check OK:", json.dumps(versions, indent=2), flush=True)

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
    stage_dir_arg = str(getattr(args, "stage_video_dir", "") or "")
    stage_dir = Path(stage_dir_arg) if stage_dir_arg else None
    if stage_dir is not None:
        stage_dir.mkdir(parents=True, exist_ok=True)
        for old in list(stage_dir.iterdir()):
            if old.is_file() or old.is_symlink():
                old.unlink()
                print(f"Removed leftover stage file {old}", flush=True)
        print(f"Staging one video at a time in {stage_dir}", flush=True)

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

    from reljoin.herbench_oracle import (
        EXPECTED_FULL_MISSING,
        EXPECTED_FULL_QUESTIONS,
        EXPECTED_TSO_N,
        PACK_GAP_STEM,
        choose_oracle_times,
        extract_evidence_spans,
        is_tso_task,
        load_rows,
    )

    base_mod = dtr_runner.load_module_from_file("herbench_base_pinned", base_src)
    acc_mod = dtr_runner.load_module_from_file("herbench_accuracy_pinned", acc_src)

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
        sample_vp = next(
            (str(r.get("video_path") or "") for r in rows if r.get("video_path")),
            "",
        )
        raise RuntimeError(
            f"{len(missing)} questions missing videos; first={missing[:5]}. "
            f"--videos-root={videos_root}. Example annotation video_path={sample_vp!r}."
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
    if args.smoke and missing:
        print(
            f"SMOKE: {len(missing)} {herbench_config} questions have no local mp4; "
            f"using {len(resolved_rows)} resolved rows."
        )

    ckpt_root = Path(args.reljoin_ckpt)
    discovered, ckpt_by_method = discover_methods_from_ckpt_dir(ckpt_root)
    if args.methods.strip():
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        methods = discovered
        if herbench_config != "full" and "dtr" not in methods:
            methods.append("dtr")
        if "hybrid" not in methods and "reljoin" not in methods:
            methods.append("hybrid")
        if herbench_config != "full" and "dtr_hybrid" not in methods:
            methods.append("dtr_hybrid")
    methods = ["hybrid" if m == "reljoin" else m for m in methods]
    methods = list(dict.fromkeys(methods))
    if "vanilla" not in methods:
        methods = ["vanilla"] + [m for m in methods if m != "vanilla"]
    blocked_dtr = [m for m in methods if m in {"dtr", "dtr_hybrid"}]
    if herbench_config == "full" and blocked_dtr:
        raise ValueError(
            "DTR is not part of HERBench full eval "
            f"(refusing {blocked_dtr}). Use --methods vanilla,hybrid."
        )
    known_methods = {"vanilla", "dtr", *METHOD_TO_VARIANT.keys(), *VARIANT_TO_METHOD.values()}
    unknown = [m for m in methods if m not in known_methods]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}. Known: {sorted(known_methods)}")
    print("Eval methods:", methods, flush=True)
    weave_methods = [m for m in methods if m not in {"vanilla", "dtr"}]
    for m in weave_methods:
        vname = METHOD_TO_VARIANT.get(m, m)
        cand = ckpt_by_method.get(m) or (ckpt_root / f"{vname}.pt" if ckpt_root.is_dir() else ckpt_root)
        if not Path(cand).is_file():
            scanned = sorted(p.name for p in ckpt_root.glob("*.pt")) if ckpt_root.exists() else []
            raise FileNotFoundError(
                f"{vname}.pt is required under {ckpt_root} for method={m}. "
                f"Found: {scanned or 'NO DIR'}."
            )
    print(
        "Loading Video-LLaVA-7B — several minutes of silence is expected; "
        "worker is still running.",
        flush=True,
    )
    model, tokenizer, video_processor = dtr.load_model_with_global_rebalance(
        model_path=str(model_dir),
        device=0,
        start_layer=18,
        end_layer=31,
        alpha=0.5,
        beta=0.4,
    )
    print("Video-LLaVA-7B loaded.", flush=True)
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
    joining_modules = {}
    for m in methods:
        if m in {"vanilla", "dtr"}:
            continue
        vname = METHOD_TO_VARIANT[m]
        cand = ckpt_by_method.get(m)
        if cand is None:
            cand = ckpt_root / f"{vname}.pt" if ckpt_root.is_dir() else ckpt_root
        if not Path(cand).is_file():
            raise FileNotFoundError(
                f"Missing checkpoint for method={m} variant={vname}: {cand}"
            )
        joining_modules[m] = load_reljoin_module(cand, llm_dim, model.device, vname)

    def format_prompt(row: dict) -> str:
        return base_mod.BaseVLM._format_prompt(None, row["question"], row["choices"])

    def parse_choice(text: str) -> str:
        return base_mod.BaseVLM.extract_answer_choice(None, str(text))

    fingerprint_payload = {
        "protocol": (
            f"herbench_{'full' if herbench_config == 'full' else 'lite_v2'}"
            f"_oracle8_frameweave_v6_videollava"
        ),
        "herbench_config": herbench_config,
        "skip_unresolved": skip_unresolved,
        "dataset_revision": args.herbench_revision,
        "dtr_commit": args.dtr_commit,
        "model_revision": args.model_revision,
        "num_frames": num_video_frames,
        "methods": methods,
        "reljoin_ckpt": args.reljoin_ckpt,
        "method": "FrameWeave-Hybrid",
        "method_version": METHOD_VERSION,
        "backbone": "Video-LLaVA-7B",
        "formulation": "5.1.0",
    }
    if tso_only:
        fingerprint_payload["protocol"] = f"{fingerprint_payload['protocol']}_tso152"
        fingerprint_payload["tso_only"] = True
    # Fingerprint excludes --relation-off so vanilla/reljoin resume the same CSV.
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
        print(
            f"SMOKE mode: {len(pending)} questions × {len(methods)} methods",
            flush=True,
        )

    clip_cache = dtr_runner.OracleClipCache(cache_dir)
    oracle_seed_base = 20260811
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

    all_records = []
    need_dtr = any(m in {"dtr", "dtr_hybrid"} for m in methods)
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
                spans, mode_hint = extract_evidence_spans(row["metadata"])
                times, padded, mode = choose_oracle_times(
                    spans, row["duration"], num_frames=num_video_frames
                )
                if mode_hint == "uniform_fallback":
                    mode = "uniform_fallback"
                    padded = False
                clip_path, indices, meta = clip_cache.get_or_build(
                    read_path,
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
                    use_raw_mean=False,
                )
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
                        "task_type": row.get("task_type", "") or row_task_type(row),
                        "method": method,
                        "question": row["question"],
                        "option_order": json.dumps(option_order),
                        "choices_json": json.dumps(row["choices"], ensure_ascii=False),
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
                        "eval_mode": "",
                    }
                    try:
                        if method == "vanilla":
                            record["eval_mode"] = "bypass"
                            if need_dtr:
                                from reljoin.videollava_hooks import configure_reljoin

                                configure_reljoin(None, enabled=False, bypass=True)
                                raw = dtr_runner.generate_once(
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
                            else:
                                raw = generate_reljoin(
                                    model,
                                    tokenizer,
                                    video_tensor,
                                    prompt,
                                    num_video_frames,
                                    None,
                                    q_last,
                                    timestamps,
                                    frame_mask,
                                    bypass=True,
                                )
                        elif method == "dtr":
                            from reljoin.videollava_hooks import configure_reljoin

                            configure_reljoin(None, enabled=False, bypass=True)
                            record["eval_mode"] = "dtr"
                            raw = dtr_runner.generate_once(
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
                            dtr_runner.set_attention_condition(
                                model, dtr, vanilla_fn, dtr_fn, use_dtr=False
                            )
                        elif method == "dtr_hybrid":
                            record["eval_mode"] = "dtr+hybrid"
                            dtr_runner.set_attention_condition(
                                model, dtr, vanilla_fn, dtr_fn, use_dtr=True
                            )
                            try:
                                raw = generate_reljoin(
                                    model,
                                    tokenizer,
                                    video_tensor,
                                    prompt,
                                    num_video_frames,
                                    joining_modules[method],
                                    q_last,
                                    timestamps,
                                    frame_mask,
                                )
                            finally:
                                dtr_runner.set_attention_condition(
                                    model, dtr, vanilla_fn, dtr_fn, use_dtr=False
                                )
                        else:
                            if need_dtr:
                                dtr_runner.set_attention_condition(
                                    model, dtr, vanilla_fn, dtr_fn, use_dtr=False
                                )
                            qh = q_last
                            record["eval_mode"] = "full"
                            raw = generate_reljoin(
                                model,
                                tokenizer,
                                video_tensor,
                                prompt,
                                num_video_frames,
                                joining_modules[method],
                                qh,
                                timestamps,
                                frame_mask,
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
                            f"  {method} {row['question_id']} pred={pred} gt={answer} "
                            f"ok={is_correct} {time.time() - t0:.1f}s",
                            flush=True,
                        )
                    except Exception as exc:
                        record["error_message"] = repr(exc)
                        record["error"] = repr(exc)
                        record["status"] = "runtime_error"
                        append_record(pairs_csv, record)
                        best_effort_copy(pairs_csv, persist_csv)
                        raise
                    append_record(pairs_csv, record)
                    all_records.append(record)
                    completed.add(key)

            del video_tensor
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            if staged is not None and Path(staged).exists():
                Path(staged).unlink()
                print(f"Removed stage file {staged}", flush=True)

        best_effort_copy(pairs_csv, persist_csv)

    df = pd.read_csv(pairs_csv)
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
        official = acc_mod.calculate_accuracy(preds) if preds else {}
        summary[method]["official"] = official
        sub.to_csv(out_dir / f"herbench_{method}.csv", index=False)

    stats_payload = {}
    if "vanilla" in present:
        base_rows = df_ok[df_ok["method"].astype(str) == "vanilla"].to_dict(orient="records")
        for method in methods:
            if method == "vanilla" or method not in present:
                continue
            other = df_ok[df_ok["method"].astype(str) == method].to_dict(orient="records")
            if other:
                stats_payload[f"vanilla_vs_{method}"] = compare_methods(
                    base_rows, other, id_key="question_id", correct_key="is_correct"
                )
    if "dtr" in present and "hybrid" in present:
        dtr_rows = df_ok[df_ok["method"].astype(str) == "dtr"].to_dict(orient="records")
        hyb_rows = df_ok[df_ok["method"].astype(str) == "hybrid"].to_dict(orient="records")
        stats_payload["dtr_vs_hybrid"] = compare_methods(
            dtr_rows, hyb_rows, id_key="question_id", correct_key="is_correct"
        )
    if "dtr_hybrid" in present:
        combo_rows = df_ok[df_ok["method"].astype(str) == "dtr_hybrid"].to_dict(
            orient="records"
        )
        if "dtr" in present:
            dtr_rows = df_ok[df_ok["method"].astype(str) == "dtr"].to_dict(orient="records")
            stats_payload["dtr_vs_dtr_hybrid"] = compare_methods(
                dtr_rows, combo_rows, id_key="question_id", correct_key="is_correct"
            )
        if "hybrid" in present:
            hyb_rows = df_ok[df_ok["method"].astype(str) == "hybrid"].to_dict(
                orient="records"
            )
            stats_payload["hybrid_vs_dtr_hybrid"] = compare_methods(
                hyb_rows, combo_rows, id_key="question_id", correct_key="is_correct"
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
        "interventions": {},
        "summary": summary,
        "stats": stats_payload,
        "n_ok": int(len(df_ok)),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "pairs_csv": str(pairs_csv),
    }
    summary_path = out_dir / f"summary_{tag}.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    from reljoin.diagnostics import answer_position_bias

    bias = {}
    for method in present:
        sub = df_ok[df_ok["method"].astype(str) == method]
        bias[method] = answer_position_bias(sub.to_dict(orient="records"))
    bias_path = out_dir / "answer_position_bias.json"
    bias_path.write_text(json.dumps(bias, indent=2), encoding="utf-8")
    best_effort_copy(pairs_csv, persist_csv)
    best_effort_copy(summary_path, persist_dir / summary_path.name)
    best_effort_copy(out_dir / "statistical_tests.json", persist_dir / "statistical_tests.json")
    best_effort_copy(bias_path, persist_dir / bias_path.name)

    print(json.dumps(summary, indent=2), flush=True)
    if tso_only:
        print(f"TSO-only n={len(rows)}", flush=True)
    if args.smoke:
        print("SMOKE OK", flush=True)
        cols = [c for c in ["question_id", "method", "parsed_prediction", "ground_truth"] if c in df_ok.columns]
        if cols:
            print(df_ok[cols].to_string(index=False), flush=True)
    else:
        print("FULL OK", flush=True)


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
    p.add_argument(
        "--methods",
        default="vanilla,dtr,hybrid,dtr_hybrid",
        help="Comma list. lite default vanilla,dtr,hybrid,dtr_hybrid. full: vanilla,hybrid.",
    )
    p.add_argument(
        "--reljoin-ckpt",
        default="/content/outputs_reljoin_v6_videollava/checkpoints/best_external_val",
    )
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
        # Skip CUDA destructor hang so the notebook can start the next job.
        os._exit(1)
    os._exit(0)
