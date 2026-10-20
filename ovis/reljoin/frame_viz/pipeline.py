"""Collect, render, and optionally intervene. Rendering does not load Ovis."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .audit import sha256_file, source_revision, write_audit
from .collector import FrameVizSession
from .geometry import resolve_runtime_geometry
from .queries import select_queries
from .render import contact_sheet, render_example
from .survey import (
    SURVEY_RULE_VERSION,
    contact_sort_key,
    roles_disagree,
    viz_role,
    write_paper_groups,
)

PIPELINE_VERSION = "1"
UPDATED_ATOL = 1e-4
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def safe_example_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return (text or "example")[:80]


def fingerprint(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def assert_safe_output(path: Path) -> None:
    text = str(path)
    for banned in ("predictions_reljoin", "predictions_residual", "eval_herbench"):
        if banned in text:
            raise RuntimeError(f"Refusing to write visualizations into an evaluation output: {path}")


def _json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value))


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _updated_cpu():
    import torch
    from reljoin.ovis_hooks import get_hook_state

    embeds = get_hook_state().last_updated_embeds
    if embeds is None:
        return None
    return embeds.detach().float().cpu()


def _max_abs(left, right) -> float | None:
    import torch

    if left is None or right is None or tuple(left.shape) != tuple(right.shape):
        return None
    return float((left - right).abs().max()) if isinstance(left, torch.Tensor) else float(np.max(np.abs(left - right)))


def _letter_scores(logits, tokenizer, choices: list[str]) -> dict | None:
    from reljoin.losses import letter_token_ids

    n_choices = sum(1 for choice in choices if str(choice).strip())
    if logits is None or n_choices <= 0:
        return None
    if hasattr(logits, "detach"):
        logits = logits.detach().float().cpu()
    ids = letter_token_ids(tokenizer, n_choices)
    scores = {}
    for index in range(n_choices):
        scores[LETTERS[index]] = float(logits[int(ids[index])])
    return scores


def _prediction_record(text: str, scores: dict | None, parse_choice) -> dict:
    parsed = parse_choice(text)
    scored = ""
    if scores:
        scored = max(scores, key=scores.get)
    return {
        "raw": text,
        "parsed": parsed,
        "scored_letter": scored,
        "letter_scores": scores or {},
        "parsed_equals_scored": (not scored) or parsed == scored,
    }


def _save_npz(path: Path, bundle: dict, geometry: dict, dump_full_tokens: bool) -> None:
    spatial = np.asarray(geometry["spatial_token_mask"], dtype=bool)
    payload = {
        "alpha": np.asarray(bundle["alpha"], dtype=np.float32),
        "beta": np.asarray(bundle["beta"], dtype=np.float32),
        "pi_selected": np.asarray(bundle["pi_selected"], dtype=np.float32),
        "gamma_selected": np.asarray(bundle["gamma_selected"], dtype=np.float32),
        "input_rms": np.asarray(bundle["input_rms"], dtype=np.float32),
        "delta_rms": np.asarray(bundle["delta_rms"], dtype=np.float32),
        "module_delta_rms": np.asarray(bundle["module_delta_rms"], dtype=np.float32),
        "eta": np.asarray(bundle["eta"], dtype=np.float32),
        "cosine_x_delta": np.asarray(bundle["cosine_x_delta"], dtype=np.float32),
        "token_mask": np.asarray(bundle["token_mask"], dtype=bool),
        "frame_mask": np.asarray(bundle["frame_mask"], dtype=bool),
        "spatial_token_mask": spatial,
        "query_indices": np.asarray(bundle["query_indices"], dtype=np.int64),
    }
    if bundle.get("gamma") is not None:
        payload["gamma"] = np.asarray(bundle["gamma"], dtype=np.float32)
    if dump_full_tokens and bundle.get("original_tokens") is not None:
        payload["x"] = np.asarray(bundle["original_tokens"], dtype=np.float32)
        payload["effective_delta"] = np.asarray(bundle["effective_delta"], dtype=np.float32)
    np.savez(path, **payload)


def _run_frameweave(model, joining, prepared, question_hidden, timestamps, frame_mask, pixels, blocked_sources=None):
    from reljoin.ovis_generate import generate_ovis

    min_pixels, max_pixels = pixels
    return generate_ovis(
        model,
        None,
        "",
        joining,
        question_hidden,
        timestamps,
        frame_mask,
        bypass=False,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        prepared=prepared,
        return_scores=True,
        blocked_sources=blocked_sources,
    )


def _compare_passes(reference: dict, other: dict, label: str) -> dict:
    text_equal = reference["text"] == other["text"]
    token_gap = _max_abs(reference["updated"], other["updated"])
    ok = text_equal and token_gap is not None and token_gap <= UPDATED_ATOL
    if not ok:
        raise RuntimeError(
            f"debug {label} diverged from the collector-off forward: "
            f"text_equal={text_equal} max_abs_updated={token_gap} atol={UPDATED_ATOL}"
        )
    return {"label": label, "text_equal": text_equal, "max_abs_updated": token_gap, "atol": UPDATED_ATOL}


def collect_examples(args) -> None:
    import torch

    from reljoin.config import HERBENCH_ORACLE_FRAMES, MAX_PIXELS, MIN_PIXELS, MODEL_ID, ExperimentConfig
    from reljoin.diagnostics import parse_choice_strict
    from reljoin.herbench_oracle import (
        choose_oracle_times,
        extract_evidence_spans,
        load_module_from_file,
        load_oracle_pil_frames,
        load_rows,
        sha256_file as sha256_path,
    )
    from reljoin.ovis_generate import format_mc_prompt, generate_ovis, preprocess_ovis_inputs
    from reljoin.ovis_hooks import (
        begin_prepare_cache,
        encode_question_hidden,
        end_prepare_cache,
        freeze_backbone,
        get_hook_state,
        install_reljoin_hook,
        ovis_hidden_size,
        visual_atom_spans,
    )
    from reljoin.train import load_ovis
    from run_herbench_oracle_reljoin import load_reljoin_module, row_task_type

    output_dir = Path(args.output_dir)
    assert_safe_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.adapter_checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "Hybrid.pt"
    if checkpoint.name != "Hybrid.pt":
        raise RuntimeError(f"Visualization uses the Hybrid checkpoint, got {checkpoint.name}")
    herbench_root = Path(args.herbench_root)
    videos_root = Path(args.videos_root)
    rows = load_rows(herbench_root, videos_root, args.herbench_revision, config="lite_v2")
    for row in rows:
        if not row.get("task_type"):
            row["task_type"] = row_task_type(row)
    chosen = _select_rows(rows, args)
    if not chosen:
        raise RuntimeError(f"No HERBench rows with a local video under {videos_root}")

    base_path = Path(args.herbench_base_vlm) if args.herbench_base_vlm else Path()
    if base_path.is_file() and args.herbench_base_sha256 and sha256_path(base_path) != args.herbench_base_sha256:
        raise RuntimeError("HERBench base_vlm.py SHA256 mismatch")
    base_mod = load_module_from_file("herbench_base_pinned_viz", base_path) if base_path.is_file() else None

    def format_prompt(row: dict) -> str:
        if base_mod is not None:
            return base_mod.BaseVLM._format_prompt(None, row["question"], row["choices"])
        return format_mc_prompt(row["question"], row["choices"])

    def parse_choice(text: str) -> str:
        if base_mod is not None:
            return base_mod.BaseVLM.extract_answer_choice(None, str(text))
        parsed, ok = parse_choice_strict(text)
        return parsed if ok else "ERROR"

    root = Path(args.root)
    cfg = ExperimentConfig(run_mode="eval_herbench", root=root)
    if args.model_dir:
        cfg.model_dir = Path(args.model_dir)
    model, tokenizer = load_ovis(cfg)
    model.eval()
    freeze_backbone(model)
    install_reljoin_hook(model)
    joining = load_reljoin_module(checkpoint, ovis_hidden_size(model), model.device, variant="Hybrid")
    joining.eval()
    num_frames = int(args.num_frames or HERBENCH_ORACLE_FRAMES)
    min_pixels = int(args.min_pixels or MIN_PIXELS)
    max_pixels = int(args.max_pixels or MAX_PIXELS)
    manifest = _load_selection(args.selection_manifest)
    query_method = str(manifest.get("query_method") or args.query_method or "top_eta")
    ckpt_sha = sha256_file(checkpoint)
    revision = source_revision(root)
    run_fp = fingerprint(
        {
            "pipeline": PIPELINE_VERSION,
            "checkpoint": ckpt_sha,
            "model_revision": args.model_revision,
            "herbench_revision": args.herbench_revision,
            "num_frames": num_frames,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "query_method": query_method,
            "max_queries": int(args.max_queries),
            "model_id": MODEL_ID,
            "survey_rule": str(manifest.get("survey_rule") or SURVEY_RULE_VERSION),
        }
    )
    _json_dump(
        output_dir / "run_manifest.json",
        {
            "command": "collect",
            "fingerprint": run_fp,
            "pipeline_version": PIPELINE_VERSION,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": ckpt_sha,
            "source_revision": revision,
            "model_id": MODEL_ID,
            "model_revision": args.model_revision,
            "herbench_revision": args.herbench_revision,
            "query_method": query_method,
            "max_queries": int(args.max_queries),
            "dump_full_tokens": bool(args.dump_full_tokens),
            "intervention": False,
            "device_dtype": "inherited from ExperimentConfig / loaded Ovis",
            "updated_token_atol": UPDATED_ATOL,
            "survey_rule": str(manifest.get("survey_rule") or SURVEY_RULE_VERSION),
            "examples": [row["question_id"] for row in chosen],
        },
    )
    survey_by_id = {
        str(item.get("question_id")): item for item in (manifest.get("selected") or [])
    }
    parity_rows = []
    for row in chosen:
        example_id = safe_example_id(row["question_id"])
        example_dir = output_dir / example_id
        example_fp = fingerprint(
            {
                "run": run_fp,
                "question_id": row["question_id"],
                "explicit": (manifest.get("queries") or {}).get(row["question_id"]),
                "rois": (manifest.get("rois") or {}).get(row["question_id"]),
            }
        )
        meta_path = example_dir / "metadata.json"
        if meta_path.is_file() and (example_dir / "tensors.npz").is_file():
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
            if previous.get("fingerprint") == example_fp:
                print(f"resume {example_id}", flush=True)
                continue
        example_dir.mkdir(parents=True, exist_ok=True)
        spans, mode_hint = extract_evidence_spans(row["metadata"])
        times, _padded, mode = choose_oracle_times(spans, row["duration"], num_frames=num_frames)
        if mode_hint == "uniform_fallback":
            mode = "uniform_fallback"
        frames, indices, frame_meta = load_oracle_pil_frames(
            Path(row["resolved_video_path"]), times, row["duration"]
        )
        prompt = format_prompt(row)
        # Gold is stored for the figure header only. It is not passed into encoding.
        question_hidden = encode_question_hidden(
            model, tokenizer, row["question"], row["choices"], model.device, model.dtype
        )
        timestamps = torch.tensor([times], device=model.device, dtype=model.dtype)
        frame_mask = torch.ones(1, len(frames), device=model.device, dtype=torch.bool)
        begin_prepare_cache()
        try:
            from reljoin.ovis_hooks import configure_reljoin

            configure_reljoin(joining, enabled=True, bypass=True)
            prepared = preprocess_ovis_inputs(model, frames, prompt, min_pixels, max_pixels)
            input_ids, _pixels, grid_thws = prepared
            atom_spans = visual_atom_spans(input_ids)
            tokens_per_frame = [end - start for start, end in atom_spans]
            geometry = resolve_runtime_geometry(
                model, frames, grid_thws, tokens_per_frame,
                min_pixels=min_pixels, max_pixels=max_pixels,
            )
            frame_dir = example_dir / "frames"
            frame_dir.mkdir(exist_ok=True)
            for index, image in enumerate(frames):
                image.save(frame_dir / f"frame_{index:02d}.png")
            vanilla = generate_ovis(
                model, None, "", joining, question_hidden, timestamps, frame_mask,
                bypass=True, max_pixels=max_pixels, min_pixels=min_pixels,
                prepared=prepared, return_scores=True,
            )
            off = _capture_pass(
                model, joining, prepared, question_hidden, timestamps, frame_mask,
                (min_pixels, max_pixels), session=None,
            )
            get_hook_state()
            if list(get_hook_state().last_tokens_per_frame) != tokens_per_frame:
                raise RuntimeError("visual-atom spans changed between preprocess and the forward")
            explicit = (manifest.get("queries") or {}).get(row["question_id"]) or []
            rois = (manifest.get("rois") or {}).get(row["question_id"]) or []
            comparisons = []
            scalar_pass, scalar_bundle = _capture_bundle(
                model, joining, prepared, question_hidden, timestamps, frame_mask,
                (min_pixels, max_pixels), query_indices=[], collect_pi=False,
                keep_updated=bool(args.dump_full_tokens),
            )
            comparisons.append(_compare_passes(off, scalar_pass, "scalar"))
            eta_np = np.asarray(scalar_bundle["eta"][0])
            spatial_np = np.asarray(geometry["spatial_token_mask"], dtype=bool)
            if tuple(spatial_np.shape) != tuple(eta_np.shape):
                raise RuntimeError(
                    f"geometry spatial mask {spatial_np.shape} != eta {eta_np.shape}"
                )
            query_doc = select_queries(
                query_method,
                eta=eta_np,
                token_mask=np.asarray(scalar_bundle["token_mask"][0]),
                spatial_mask=spatial_np,
                geometry=geometry,
                max_queries=int(args.max_queries),
                explicit=explicit,
                rois=rois,
                seed=0,
            )
            queries = [(item["frame_index"], item["token_index"]) for item in query_doc["queries"]]
            if queries:
                pi_pass, bundle = _capture_bundle(
                    model, joining, prepared, question_hidden, timestamps, frame_mask,
                    (min_pixels, max_pixels), query_indices=queries, collect_pi=True,
                    keep_updated=bool(args.dump_full_tokens),
                )
                comparisons.append(_compare_passes(off, pi_pass, "pi"))
                eta_gap = float(np.max(np.abs(np.asarray(bundle["eta"]) - np.asarray(scalar_bundle["eta"]))))
                if eta_gap > UPDATED_ATOL:
                    raise RuntimeError(f"eta changed between the scalar and pi passes: {eta_gap}")
                used_pass = pi_pass
            else:
                bundle = scalar_bundle
                used_pass = scalar_pass
            off["updated"] = None
            scalar_pass["updated"] = None
            if queries:
                pi_pass["updated"] = None
            survey_info = dict(survey_by_id.get(str(row["question_id"])) or {})
            survey_info["n_evidence_spans"] = len(spans)
            _persist_example(
                example_dir, row, prompt, times, indices, frame_meta, mode,
                geometry, bundle, query_doc, vanilla, used_pass, off, comparisons,
                tokenizer, parse_choice, args, checkpoint, ckpt_sha, revision, example_fp,
                min_pixels, max_pixels, num_frames, grid_thws, input_ids, survey_info,
            )
            parity_rows.append({"example_id": row["question_id"], "comparisons": comparisons, "status": "pass"})
            print(f"collected {example_id}", flush=True)
        finally:
            end_prepare_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_examples_index(output_dir)
    status = "pass" if parity_rows else "resumed_without_new_forward"
    _merge_validation(
        output_dir,
        {
            "gpu_collect": {"status": status, "examples": parity_rows, "atol": UPDATED_ATOL},
            "gpu_smoke": {
                "status": status,
                "examples": [row["example_id"] for row in parity_rows],
                "atol": UPDATED_ATOL,
                "note": "pass only after a real Hybrid forward on prepared HERBench frames",
            },
        },
    )


def _capture_pass(model, joining, prepared, question_hidden, timestamps, frame_mask, pixels, session):
    if session is None:
        payload = _run_frameweave(
            model, joining, prepared, question_hidden, timestamps, frame_mask, pixels
        )
        return {"text": payload["text"], "payload": payload, "updated": _updated_cpu()}
    with session:
        payload = _run_frameweave(
            model, joining, prepared, question_hidden, timestamps, frame_mask, pixels
        )
        updated = _updated_cpu()
        bundle = session.export()
    return {"text": payload["text"], "payload": payload, "updated": updated, "bundle": bundle}


def _capture_bundle(
    model, joining, prepared, question_hidden, timestamps, frame_mask, pixels,
    *, query_indices, collect_pi, keep_updated,
):
    session = FrameVizSession(
        query_indices=query_indices, collect_pi=collect_pi, keep_updated=keep_updated,
    )
    captured = _capture_pass(
        model, joining, prepared, question_hidden, timestamps, frame_mask, pixels, session
    )
    return captured, captured["bundle"]


def _persist_example(
    example_dir, row, prompt, times, indices, frame_meta, mode, geometry, bundle, query_doc,
    vanilla, used_pass, off, comparisons, tokenizer, parse_choice, args, checkpoint, ckpt_sha,
    revision, example_fp, min_pixels, max_pixels, num_frames, grid_thws, input_ids,
    survey_info=None,
) -> None:
    from reljoin.ovis_generate import sha256_ids

    vanilla_scores = _letter_scores(vanilla.get("first_step_logits"), tokenizer, row["choices"])
    fw_scores = _letter_scores(used_pass["payload"].get("first_step_logits"), tokenizer, row["choices"])
    predictions = {
        "vanilla": _prediction_record(vanilla["text"], vanilla_scores, parse_choice),
        "frameweave": _prediction_record(used_pass["text"], fw_scores, parse_choice),
        "collector_off_raw": off["text"],
        "parity": comparisons,
        "note": "parsed is the official generation parser; scored_letter is argmax of first-step letter logits",
    }
    shapes = {
        key: list(np.asarray(bundle[key]).shape)
        for key in ("alpha", "beta", "pi_selected", "gamma_selected", "eta", "token_mask", "frame_mask")
    }
    metadata = {
        "example_id": row["question_id"],
        "safe_example_id": example_dir.name,
        "video_id": Path(row["resolved_video_path"]).stem,
        "video_path": row["resolved_video_path"],
        "task": row.get("task_type") or "",
        "question": row["question"],
        "choices": row["choices"],
        "gold": row["answer"],
        "fingerprint": example_fp,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": ckpt_sha,
        "source_revision": revision,
        "model_revision": args.model_revision,
        "frame_indices": indices,
        "timestamps": [float(t) for t in times],
        "frame_selection_mode": mode,
        "fps": frame_meta.get("fps"),
        "mapping_status": geometry.get("mapping_status"),
        "indexing": {
            "query_indices": "[S, 2] as (receiver_frame_index, token_index) into the packed [K, Pmax] tokens",
            "beta": "row receiver frame i, column frame read j",
            "gamma_selected": "[B, S, K] from last_gamma[b, i, p, j]",
            "pi_selected": "[B, S, K, Pmax], softmax over the last axis inside LocalRelation",
            "eta": "RMS(effective delivered delta) / (RMS(x) + 1e-8); not the pre-cast module delta",
        },
        "shapes": shapes,
        "dtypes": "float32 CPU copies; masks bool",
        "dtype_gap_max_abs": bundle.get("dtype_gap_max_abs"),
        "eta_source": bundle.get("eta_source"),
        "module_stats": bundle.get("module_stats"),
        "normalization": geometry.get("composition_rules"),
        "dump_full_tokens": bool(args.dump_full_tokens),
        "prompt_sha256": _sha_text(prompt),
        "n_evidence_spans": (survey_info or {}).get("n_evidence_spans", ""),
        "survey_quota": (survey_info or {}).get("quota", ""),
        "survey_reason": (survey_info or {}).get("reason", ""),
        "prior_role": (survey_info or {}).get("prior_role", ""),
        "task_family": (survey_info or {}).get("family", ""),
    }
    _json_dump(example_dir / "metadata.json", metadata)
    _json_dump(example_dir / "token_geometry.json", geometry)
    _json_dump(example_dir / "queries.json", query_doc)
    _json_dump(example_dir / "predictions.json", predictions)
    grids = grid_thws.detach().cpu().tolist() if hasattr(grid_thws, "detach") else grid_thws
    _json_dump(
        example_dir / "preprocessing.json",
        {
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "num_frames": num_frames,
            "timestamps": [float(t) for t in times],
            "frame_indices": indices,
            "grid_thws": grids,
            "prompt_sha256": _sha_text(prompt),
            "input_ids_sha256": sha256_ids(input_ids),
            "tokenizer_spec": geometry.get("tokenizer_spec") or {},
            "mapping_notes": geometry.get("notes") or [],
        },
    )
    _save_npz(example_dir / "tensors.npz", bundle, geometry, bool(args.dump_full_tokens))


def render_output(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    for example_dir in sorted(path for path in output_dir.iterdir() if (path / "tensors.npz").is_file()):
        render_example(example_dir)
        print(f"rendered {example_dir.name}", flush=True)
    write_examples_index(output_dir)
    write_paper_groups(output_dir)
    contact_sheet(output_dir)
    _write_caption_rollup(output_dir)


def write_examples_index(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    rows = []
    for example_dir in sorted(path for path in output_dir.iterdir() if (path / "metadata.json").is_file()):
        meta = json.loads((example_dir / "metadata.json").read_text(encoding="utf-8"))
        predictions = json.loads((example_dir / "predictions.json").read_text(encoding="utf-8"))
        queries = json.loads((example_dir / "queries.json").read_text(encoding="utf-8"))
        gold = str(meta.get("gold") or "")
        vanilla = (predictions.get("vanilla") or {}).get("parsed", "")
        frameweave = (predictions.get("frameweave") or {}).get("parsed", "")
        role = viz_role(str(meta.get("task") or ""), gold, str(vanilla), str(frameweave), meta.get("choices") or [])
        prior = str(meta.get("prior_role") or "")
        frame_path = example_dir / "frames" / "frame_00.png"
        rows.append(
            {
                "example_id": meta.get("example_id", ""),
                "video_id": meta.get("video_id", ""),
                "task": meta.get("task", ""),
                "role": role,
                "gold": gold,
                "vanilla": vanilla,
                "frameweave": frameweave,
                "vanilla_correct": str(vanilla == gold),
                "frameweave_correct": str(frameweave == gold),
                "n_evidence_spans": meta.get("n_evidence_spans", ""),
                "survey_reason": meta.get("survey_reason", ""),
                "prior_role": prior,
                "role_disagrees_with_prior_eval": str(roles_disagree(prior, role)),
                "n_frames": len(meta.get("timestamps") or []),
                "query_method": queries.get("method", ""),
                "mapping_status": meta.get("mapping_status", ""),
                "figure_cross_frame": str(example_dir / "figure_cross_frame.png"),
                "figure_read_update": str(example_dir / "figure_read_update.png"),
                "figure_frame_matrix": str(example_dir / "figure_frame_matrix.png"),
                "frame_path": str(frame_path if frame_path.is_file() else ""),
            }
        )
    rows.sort(key=contact_sort_key)
    path = output_dir / "examples_index.csv"
    fields = [
        "example_id", "video_id", "task", "role", "gold", "vanilla", "frameweave",
        "vanilla_correct", "frameweave_correct", "n_evidence_spans", "survey_reason",
        "prior_role", "role_disagrees_with_prior_eval", "n_frames", "query_method",
        "mapping_status", "figure_cross_frame", "figure_read_update",
        "figure_frame_matrix", "frame_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_caption_rollup(output_dir: Path) -> None:
    parts = ["# English captions", "", "These describe the rendered examples. They are not a benchmark summary.", ""]
    for example_dir in sorted(output_dir.iterdir()):
        caption = example_dir / "captions_en.md"
        if caption.is_file():
            parts.append(caption.read_text(encoding="utf-8"))
    (output_dir / "captions_en.md").write_text("\n".join(parts), encoding="utf-8")


def _select_rows(rows: list[dict], args) -> list[dict]:
    manifest = _load_selection(args.selection_manifest)
    if args.selection_manifest and "example_ids" not in manifest:
        raise RuntimeError(f"Selection manifest has no example_ids: {args.selection_manifest}")
    ids = [item for item in str(args.example_ids or "").split(",") if item.strip()]
    ids.extend(str(item) for item in manifest.get("example_ids") or [])
    if args.selection_manifest and not ids:
        raise RuntimeError(f"Selection manifest selected no examples: {args.selection_manifest}")
    resolved = [row for row in rows if row.get("resolved_video_path")]
    if ids:
        by_id = {str(row["question_id"]): row for row in rows}
        missing = [item for item in ids if item not in by_id]
        if missing:
            raise RuntimeError(f"Unknown example ids: {missing[:8]}")
        chosen = [by_id[item] for item in ids]
        if getattr(args, "command", "") == "smoke":
            keep = max(1, int(getattr(args, "limit", 0) or 1))
            return chosen[:keep]
        return chosen
    limit = int(args.limit or 24)
    return resolved[:limit]


def _load_selection(path: str) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _merge_validation(output_dir: Path, update: dict) -> None:
    path = Path(output_dir) / "validation_report.json"
    current = {}
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(update)
    _json_dump(path, current)


def audit_command(args) -> None:
    output_dir = Path(args.output_dir)
    assert_safe_output(output_dir)
    checkpoint = Path(args.adapter_checkpoint) if args.adapter_checkpoint else None
    if checkpoint is not None and checkpoint.is_dir():
        checkpoint = checkpoint / "Hybrid.pt"
    blockers = []
    if checkpoint is None or not checkpoint.is_file():
        blockers.append(f"Hybrid checkpoint not found at {checkpoint}")
    if args.model_dir and not (Path(args.model_dir) / "config.json").is_file():
        blockers.append(f"Ovis config.json missing under {args.model_dir}")
    if args.videos_root and not Path(args.videos_root).exists():
        blockers.append(f"video root missing: {args.videos_root}")
    blockers.append("GPU smoke is not marked pass by audit. Run smoke on Colab with the checkpoint and lite_v2 videos.")
    write_audit(
        output_dir,
        repo_root=Path(args.root),
        checkpoint=checkpoint,
        dataset={
            "herbench_config": "lite_v2",
            "herbench_revision": args.herbench_revision,
        },
        blockers=blockers,
    )
    print(f"wrote {output_dir / 'implementation_audit.md'}", flush=True)


def intervene_examples(args) -> None:
    """Block one read-source frame and rerun the official forward. Default off."""
    import torch

    from reljoin.config import ExperimentConfig
    from reljoin.diagnostics import parse_choice_strict
    from reljoin.herbench_oracle import load_module_from_file, sha256_file as sha256_path
    from reljoin.ovis_generate import format_mc_prompt, preprocess_ovis_inputs
    from reljoin.ovis_hooks import (
        begin_prepare_cache,
        encode_question_hidden,
        end_prepare_cache,
        freeze_backbone,
        install_reljoin_hook,
        ovis_hidden_size,
    )
    from reljoin.train import load_ovis
    from run_herbench_oracle_reljoin import load_reljoin_module

    output_dir = Path(args.output_dir)
    examples = [path for path in sorted(output_dir.iterdir()) if (path / "tensors.npz").is_file()]
    if args.example_ids:
        wanted = {safe_example_id(item) for item in str(args.example_ids).split(",") if item.strip()}
        examples = [path for path in examples if path.name in wanted]
    if not examples:
        raise RuntimeError(f"No collected examples in {output_dir}")
    checkpoint = Path(args.adapter_checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "Hybrid.pt"
    root = Path(args.root)
    cfg = ExperimentConfig(run_mode="eval_herbench", root=root)
    if args.model_dir:
        cfg.model_dir = Path(args.model_dir)
    model, tokenizer = load_ovis(cfg)
    model.eval()
    freeze_backbone(model)
    install_reljoin_hook(model)
    joining = load_reljoin_module(checkpoint, ovis_hidden_size(model), model.device, variant="Hybrid")
    joining.eval()
    base_path = Path(args.herbench_base_vlm) if args.herbench_base_vlm else Path()
    if base_path.is_file() and args.herbench_base_sha256 and sha256_path(base_path) != args.herbench_base_sha256:
        raise RuntimeError("HERBench base_vlm.py SHA256 mismatch")
    base_mod = load_module_from_file("herbench_base_pinned_intervene", base_path) if base_path.is_file() else None

    def format_prompt(question, choices) -> str:
        if base_mod is not None:
            return base_mod.BaseVLM._format_prompt(None, question, choices)
        return format_mc_prompt(question, choices)

    def parse_choice(text: str) -> str:
        if base_mod is not None:
            return base_mod.BaseVLM.extract_answer_choice(None, str(text))
        parsed, ok = parse_choice_strict(text)
        return parsed if ok else "ERROR"

    from PIL import Image

    for example_dir in examples:
        meta = json.loads((example_dir / "metadata.json").read_text(encoding="utf-8"))
        queries = json.loads((example_dir / "queries.json").read_text(encoding="utf-8"))
        chosen = (queries.get("queries") or [None])[0]
        if not chosen:
            _json_dump(example_dir / "intervention.json", {"status": "skipped", "reason": "no query"})
            continue
        arrays = np.load(example_dir / "tensors.npz", allow_pickle=False)
        gamma = np.asarray(arrays["gamma_selected"][0, 0], dtype=np.float64)
        frame_mask = np.asarray(arrays["frame_mask"][0], dtype=bool)
        receiver = int(chosen["frame_index"])
        candidates = [j for j in range(len(gamma)) if frame_mask[j] and j != receiver]
        if not candidates:
            _json_dump(example_dir / "intervention.json", {"status": "skipped", "reason": "query has no neighbor"})
            continue
        high = max(candidates, key=lambda j: (float(gamma[j]), -j))
        low = min(candidates, key=lambda j: (float(gamma[j]), j))
        frames = [Image.open(path).convert("RGB") for path in sorted((example_dir / "frames").glob("frame_*.png"))]
        pre = json.loads((example_dir / "preprocessing.json").read_text(encoding="utf-8"))
        prompt = format_prompt(meta["question"], meta["choices"])
        if _sha_text(prompt) != meta.get("prompt_sha256"):
            raise RuntimeError(f"Prompt hash changed for {example_dir.name}; refusing intervention")
        question_hidden = encode_question_hidden(
            model, tokenizer, meta["question"], meta["choices"], model.device, model.dtype
        )
        timestamps = torch.tensor([meta["timestamps"]], device=model.device, dtype=model.dtype)
        mask = torch.ones(1, len(frames), device=model.device, dtype=torch.bool)
        begin_prepare_cache()
        try:
            from reljoin.ovis_hooks import configure_reljoin

            configure_reljoin(joining, enabled=True, bypass=True)
            prepared = preprocess_ovis_inputs(
                model, frames, prompt, int(pre["min_pixels"]), int(pre["max_pixels"])
            )
            conditions = {
                "original": None,
                "blocked_high": (high,),
                "blocked_low": (low,),
            }
            reports = {}
            for name, blocked in conditions.items():
                payload = _run_frameweave(
                    model, joining, prepared, question_hidden, timestamps, mask,
                    (int(pre["min_pixels"]), int(pre["max_pixels"])),
                    blocked_sources=blocked,
                )
                scores = _letter_scores(payload.get("first_step_logits"), tokenizer, meta["choices"])
                record = _prediction_record(payload["text"], scores, parse_choice)
                record["margin"] = _margin(scores, str(meta.get("gold") or ""))
                reports[name] = record
            original_margin = reports["original"]["margin"]
            for name in ("blocked_high", "blocked_low"):
                if original_margin is None or reports[name]["margin"] is None:
                    reports[name]["margin_drop"] = None
                else:
                    reports[name]["margin_drop"] = original_margin - reports[name]["margin"]
            _json_dump(
                example_dir / "intervention.json",
                {
                    "status": "ran",
                    "receiver_frame": receiver,
                    "query": chosen,
                    "high_sender": high,
                    "low_sender": low,
                    "high_gamma": float(gamma[high]),
                    "low_gamma": float(gamma[low]),
                    "same_sender": high == low,
                    "selection": "argmax/argmin of this query's gamma over valid senders, before looking at the intervention answer",
                    "limitation": (
                        "A few examples diagnose the computation path. They do not establish object identity "
                        "or a causal explanation of the whole model."
                    ),
                    "conditions": reports,
                },
            )
            print(f"intervened {example_dir.name}", flush=True)
        finally:
            end_prepare_cache()
            for image in frames:
                image.close()


def _margin(scores: dict | None, gold: str) -> float | None:
    if not scores or gold not in scores:
        return None
    others = [value for letter, value in scores.items() if letter != gold]
    if not others:
        return None
    return float(scores[gold] - max(others))
