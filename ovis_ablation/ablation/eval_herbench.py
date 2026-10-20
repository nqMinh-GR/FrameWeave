"""HERBench lite_v2 evaluation on one shared frame manifest."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from .common import append_jsonl, ensure_v6_on_path, read_jsonl, record_blocker, sha256_json, write_json
from .config import VARIANTS, AblationConfig
from .modules import build_adapter
from .paths import ResolvedPaths
from .train import _load_module, _v6_config, environment_snapshot, variant_dir

ensure_v6_on_path()

from reljoin.herbench_oracle import (  # noqa: E402
    choose_oracle_times,
    extract_evidence_spans,
    load_oracle_pil_frames,
    load_rows,
    times_to_frame_indices,
)
from reljoin.ovis_generate import generate_ovis, load_herbench_prompt_parser  # noqa: E402
from reljoin.ovis_hooks import (  # noqa: E402
    configure_reljoin,
    encode_question_hidden,
    freeze_backbone,
    install_reljoin_hook,
    ovis_hidden_size,
)
from reljoin.train import load_ovis  # noqa: E402


def _checkpoint_id(path: Path | None) -> str:
    if path is None:
        return "vanilla"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:16]


def _question_key(question: str, choices: list[str], template: str, revision: str) -> str:
    return sha256_json(
        {
            "question": question,
            "choices": list(choices),
            "template": template,
            "model_revision": revision,
        }
    )


def _preprocess_hash(record: dict, cfg: AblationConfig) -> str:
    return sha256_json(
        {
            "indices": record["frame_indices"],
            "timestamps": record["timestamps"],
            "min_pixels": cfg.min_pixels,
            "max_pixels": cfg.max_pixels,
            "model_revision": cfg.model_revision,
            "enable_thinking": cfg.enable_thinking,
            "oracle_mode": record["oracle_mode"],
        }
    )


def _video_geometry(path: str) -> dict:
    import cv2

    cached = _video_geometry.cache.get(path)  # type: ignore[attr-defined]
    if cached is not None:
        return cached
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or 1
    finally:
        capture.release()
    payload = {"fps": fps, "total": total, "duration": total / fps}
    _video_geometry.cache[path] = payload  # type: ignore[attr-defined]
    return payload


_video_geometry.cache = {}  # type: ignore[attr-defined]


def _plan_row(row: dict, frames: int) -> tuple[list[float], bool, str]:
    spans, hint = extract_evidence_spans(row.get("metadata") or {})
    times, padded, mode = choose_oracle_times(spans, float(row.get("duration") or 0.0), num_frames=frames)
    if hint == "uniform_fallback":
        mode = "uniform_fallback"
        padded = False
    return times, padded, mode


def build_eval_manifest(cfg: AblationConfig, resolved: ResolvedPaths) -> dict:
    reasons = []
    if not resolved.herbench_parquet:
        reasons.append("Local HERBench parquet was not found. This suite does not download lite_v2.")
    if not resolved.herbench_root or not resolved.herbench_videos:
        reasons.append("HERBench root or videos directory was not found.")
    if reasons:
        path = record_blocker(cfg.output_dir(), "eval", reasons)
        return {"status": "blocked", "reasons": reasons, "blocker": str(path), "rows": []}
    rows = load_rows(
        Path(resolved.herbench_root),
        Path(resolved.herbench_videos),
        cfg.herbench_revision,
        config=cfg.herbench_config,
    )
    if len(rows) != int(cfg.herbench_expected_questions):
        reasons.append(
            f"HERBench {cfg.herbench_config} has {len(rows)} questions, expected {cfg.herbench_expected_questions}."
        )
    manifest = []
    missing = []
    for row in rows:
        video = row.get("resolved_video_path") or ""
        planned = {
            "question_id": row["question_id"],
            "video_id": str(row.get("video_id") or ""),
            "task": str(row.get("task_type") or (row.get("metadata") or {}).get("task_type") or ""),
            "question": row["question"],
            "choices": list(row["choices"]),
            "gold_letter": str(row.get("answer") or ""),
            "video_path": video,
            "annotation_video": str(row.get("video_path") or ""),
        }
        if not video or not Path(video).is_file():
            planned.update(
                {
                    "frames_ready": False,
                    "frame_error": "missing_video",
                    "frame_indices": None,
                    "timestamps": None,
                    "oracle_mode": "",
                    "oracle_padded": False,
                }
            )
            missing.append(planned["question_id"])
        else:
            times, padded, mode = _plan_row(row, cfg.herbench_frames)
            geometry = _video_geometry(video)
            indices = times_to_frame_indices(times, geometry["fps"], geometry["total"])
            planned.update(
                {
                    "frames_ready": True,
                    "frame_error": "",
                    "frame_indices": [int(index) for index in indices],
                    "timestamps": [float(value) for value in times],
                    "oracle_mode": mode,
                    "oracle_padded": bool(padded),
                    "fps": geometry["fps"],
                    "duration": geometry["duration"] if float(row.get("duration") or 0.0) <= 0 else float(row["duration"]),
                }
            )
        planned["preprocess_hash"] = _preprocess_hash(planned, cfg) if planned["frames_ready"] else ""
        manifest.append(planned)
    videos = {row["video_id"] for row in manifest if row["video_id"]}
    report = {
        "status": "incomplete" if missing or reasons else "ready",
        "n_questions": len(manifest),
        "n_videos": len(videos),
        "n_missing": len(missing),
        "missing_question_ids": missing,
        "reasons": reasons,
        "rows": manifest,
    }
    shared = cfg.output_dir() / "shared"
    write_json(shared / "eval_manifest_meta.json", {key: value for key, value in report.items() if key != "rows"})
    manifest_path = shared / "eval_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if missing or reasons:
        record_blocker(
            cfg.output_dir(),
            "eval",
            reasons
            or [f"{len(missing)} questions have no local video. Evaluation is incomplete. See eval_manifest.jsonl."],
        )
    return report


def _completed(path: Path) -> set[tuple[str, str]]:
    done = set()
    for row in read_jsonl(path):
        if row.get("status") == "ok":
            done.add((str(row["question_id"]), str(row["variant"])))
    return done


def eval_prediction_dir(cfg: AblationConfig, limit: int | None) -> tuple[Path, bool]:
    """Smoke writes under eval_smoke. Full lite_v2 stays under eval."""
    smoke = limit is not None and int(limit) > 0
    name = "eval_smoke" if smoke else "eval"
    return cfg.output_dir() / name / f"seed_{cfg.seed}", smoke


def evaluate(
    cfg: AblationConfig,
    resolved: ResolvedPaths,
    variants: list[str],
    *,
    limit: int | None = None,
) -> dict:
    out_dir, smoke = eval_prediction_dir(cfg, limit)
    stage_name = "eval_smoke" if smoke else "eval"
    manifest_path = cfg.output_dir() / "shared" / "eval_manifest.jsonl"
    if not manifest_path.is_file():
        report = build_eval_manifest(cfg, resolved)
        if report["status"] == "blocked":
            print("EVAL BLOCKED:\n- " + "\n- ".join(report["reasons"]))
            return report
    rows = read_jsonl(manifest_path)
    if len(rows) != int(cfg.herbench_expected_questions):
        message = f"Eval manifest has {len(rows)} rows, expected {cfg.herbench_expected_questions}."
        record_blocker(cfg.output_dir(), stage_name, [message])
        print("EVAL BLOCKED:", message)
        return {"status": "blocked", "reasons": [message]}
    chosen = rows[: int(limit)] if smoke else rows
    ready = [row for row in chosen if row.get("frames_ready")]
    if not ready:
        message = "No HERBench question has a local video. Refusing to load Ovis."
        record_blocker(cfg.output_dir(), stage_name, [message])
        print("EVAL BLOCKED:", message)
        return {"status": "blocked", "reasons": [message]}
    if not torch.cuda.is_available():
        message = "CUDA is not available. Refusing a CPU fallback for HERBench generation."
        record_blocker(cfg.output_dir(), stage_name, [message])
        print("EVAL BLOCKED:", message)
        return {"status": "blocked", "reasons": [message]}
    names = [name for name in variants if name in VARIANTS]
    modules = {}
    checkpoints = {}
    v6_cfg = _v6_config(cfg, resolved)
    model, _tokenizer = load_ovis(v6_cfg)
    freeze_backbone(model)
    install_reljoin_hook(model)
    model.eval()
    llm_dim = ovis_hidden_size(model)
    for name in names:
        if name == "vanilla":
            modules[name] = None
            checkpoints[name] = "vanilla"
            continue
        ckpt = variant_dir(cfg, name) / "best" / "adapter.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing best checkpoint for {name}: {ckpt}")
        module, _payload = _load_module(ckpt, model.device)
        module.eval()
        modules[name] = module
        checkpoints[name] = _checkpoint_id(ckpt)
    parser_path = resolved.herbench_base_vlm or ""
    format_prompt, parse_choice, parser_info = load_herbench_prompt_parser(
        parser_path, cfg.herbench_base_sha256
    )
    template = "herbench_base_vlm" if parser_info.get("used_base_vlm") else "format_mc_prompt"
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = out_dir / "predictions.jsonl"
    if smoke and prediction_path.parents[1].name != "eval_smoke":
        raise RuntimeError(f"Smoke predictions must stay in eval_smoke, got {prediction_path}")
    if not smoke and prediction_path.parents[1].name != "eval":
        raise RuntimeError(f"Full predictions must stay in eval, got {prediction_path}")
    done = _completed(prediction_path)
    question_cache: dict[str, torch.Tensor] = {}
    vanilla_cache_path = out_dir / "vanilla_cache.jsonl"
    vanilla_cache = {
        row["cache_key"]: row for row in read_jsonl(vanilla_cache_path) if row.get("cache_key")
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in chosen:
        grouped[row.get("video_path") or ""].append(row)
    dtype = next(model.parameters()).dtype
    for video_path, group in grouped.items():
        for row in group:
            pending = [name for name in names if (row["question_id"], name) not in done]
            if not pending:
                continue
            if not row.get("frames_ready"):
                for name in pending:
                    _write_prediction(
                        prediction_path,
                        row,
                        name,
                        cfg.seed,
                        checkpoints.get(name, ""),
                        predicted="",
                        raw="",
                        valid=False,
                        status="missing_video",
                        error=row.get("frame_error") or "missing_video",
                    )
                continue
            frames, indices, _meta = load_oracle_pil_frames(
                Path(video_path), row["timestamps"], float(row.get("duration") or 0.0)
            )
            if [int(index) for index in indices] != [int(index) for index in row["frame_indices"]]:
                raise RuntimeError(
                    f"Frame indices for {row['question_id']} drifted from the shared manifest."
                )
            qkey = _question_key(row["question"], row["choices"], template, cfg.model_revision)
            if qkey not in question_cache:
                question_cache[qkey] = encode_question_hidden(
                    model,
                    model.text_tokenizer,
                    row["question"],
                    [choice for choice in row["choices"] if str(choice).strip()],
                    model.device,
                    dtype,
                )
            prompt = format_prompt(row["question"], [choice for choice in row["choices"] if str(choice).strip()])
            timestamps = torch.tensor([row["timestamps"]], device=model.device, dtype=dtype)
            frame_mask = torch.ones(1, len(frames), device=model.device, dtype=torch.bool)
            preprocess = row["preprocess_hash"]
            for name in pending:
                module = modules[name]
                bypass = name == "vanilla"
                cache_key = sha256_json({"question": qkey, "preprocess": preprocess, "variant": name})
                if bypass and cache_key in vanilla_cache:
                    raw = vanilla_cache[cache_key]["raw_response"]
                else:
                    try:
                        raw = generate_ovis(
                            model,
                            frames,
                            prompt,
                            module,
                            question_cache[qkey],
                            timestamps,
                            frame_mask,
                            bypass=bypass,
                            max_pixels=cfg.max_pixels,
                            min_pixels=cfg.min_pixels,
                        )
                    except Exception as exc:  # noqa: BLE001 - one failed sample must stay in the file
                        _write_prediction(
                            prediction_path,
                            row,
                            name,
                            cfg.seed,
                            checkpoints[name],
                            predicted="",
                            raw="",
                            valid=False,
                            status="runtime_error",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        continue
                    if bypass:
                        append_jsonl(
                            vanilla_cache_path,
                            {"cache_key": cache_key, "raw_response": raw, "preprocess_hash": preprocess},
                        )
                        vanilla_cache[cache_key] = {"raw_response": raw}
                n_choices = len([choice for choice in row["choices"] if str(choice).strip()]) or 5
                predicted, valid = parse_choice(raw, n_choices)
                gold = str(row["gold_letter"]).strip().upper()[:1]
                _write_prediction(
                    prediction_path,
                    row,
                    name,
                    cfg.seed,
                    checkpoints[name],
                    predicted=predicted,
                    raw=raw,
                    valid=valid,
                    status="ok",
                    error="",
                    correct=bool(valid and predicted == gold),
                )
            del frames
            configure_reljoin(None, enabled=False, bypass=True)
    predictions = read_jsonl(prediction_path)
    write_json(out_dir / "environment.json", environment_snapshot(cfg))
    write_json(
        out_dir / "metrics.json",
        {
            "paper_result": not smoke,
            "smoke": smoke,
            "limit": int(limit) if smoke else None,
            "complete": (not smoke)
            and len(predictions) >= len(rows) * len(names)
            and all(row.get("status") == "ok" for row in predictions),
            "n_predictions": len(predictions),
            "n_manifest": len(rows),
            "n_selected": len(chosen),
            "variants": names,
        },
    )
    del model
    return {"status": "ok", "predictions": str(prediction_path)}


def _write_prediction(path, row, variant, seed, checkpoint_id, *, predicted, raw, valid, status, error, correct=False) -> None:
    append_jsonl(
        path,
        {
            "question_id": row["question_id"],
            "video_id": row.get("video_id", ""),
            "task": row.get("task", ""),
            "gold_letter": row.get("gold_letter", ""),
            "predicted_letter": predicted,
            "raw_response": raw,
            "correct": bool(correct),
            "valid": bool(valid),
            "variant": variant,
            "seed": seed,
            "checkpoint_id": checkpoint_id,
            "status": status,
            "error": error,
        },
    )
