"""One TSO question: real alpha, beta, and counterfactual gold-logit contributions.

Positive contribution is ``z_y(full) - z_y(intervened)``. The primary token and
edge ablations stay inside the adapter. Raw zero/mean fills are logged only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import (
    MIN_PIXELS,
    MAX_PIXELS,
    AlignmentError,
    ModelRequired,
    processor_metadata_for_image,
    run_localization_test,
    verified_token_boxes,
)
from .select import K_CANDIDATES, select_keyframes, token_mass, top_alpha_tokens

EXPECTED_SHA256 = "3a19caac96f32e7bc82d5c373bf77c6b3884489ee2fb35c6957c5b7d8858da03"


class RuntimeBlocked(RuntimeError):
    def __init__(self, blockers: list[str]):
        self.blockers = blockers
        super().__init__(" ".join(blockers))


def runtime_blockers(model_dir: str, videos_root: str, video_path: str, checkpoint: Path) -> list[str]:
    blockers = []
    try:
        import torch
    except ImportError:
        blockers.append("PyTorch is not installed, so Ovis2.5-9B cannot be loaded.")
        return blockers
    if not torch.cuda.is_available():
        blockers.append("CUDA is not available. Refusing a CPU fallback and refusing a drawn heatmap.")
    if checkpoint.is_file():
        from reljoin.herbench_oracle import sha256_file

        digest = sha256_file(checkpoint)
        if digest != EXPECTED_SHA256:
            blockers.append(
                f"Checkpoint SHA256 {digest} != published seed-42 Hybrid {EXPECTED_SHA256}."
            )
    else:
        blockers.append(f"Hybrid checkpoint is missing: {checkpoint}")
    model_path = Path(model_dir) if model_dir else None
    if model_path is None or not (model_path / "config.json").is_file():
        blockers.append(
            "Ovis2.5-9B weights were not found. Pass --model-dir with a local config.json. "
            "No spatial figure was drawn."
        )
    if video_path:
        resolved = _resolve_video(video_path, Path(videos_root) if videos_root else None)
        if resolved is None:
            blockers.append(
                f"HERBench video for this question was not found ({video_path}). "
                "Pass --videos-root. No frame image was invented."
            )
    return blockers


def load_runtime(args) -> dict:
    """Load Ovis and the Hybrid checkpoint once. Localization runs once for that process."""
    blockers = runtime_blockers(
        getattr(args, "model_dir", ""),
        getattr(args, "videos_root", ""),
        "",
        Path(args.checkpoint),
    )
    if blockers:
        raise RuntimeBlocked(blockers)

    from reljoin.config import ExperimentConfig
    from reljoin.ovis_hooks import freeze_backbone, install_reljoin_hook, ovis_hidden_size
    from reljoin.train import load_ovis
    from run_herbench_oracle_reljoin import load_reljoin_module

    cfg = ExperimentConfig(run_mode="eval_herbench", root=Path(args.root))
    cfg.model_dir = Path(args.model_dir)
    model, tokenizer = load_ovis(cfg)
    model.eval()
    freeze_backbone(model)
    localization = run_localization_test(model, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    install_reljoin_hook(model)
    joining = load_reljoin_module(
        Path(args.checkpoint), ovis_hidden_size(model), model.device, variant="Hybrid"
    )
    joining.eval()
    return {
        "model": model,
        "tokenizer": tokenizer,
        "joining": joining,
        "localization": localization,
    }


def extract_question(case: dict, args, runtime: dict | None = None) -> dict:
    """Run one question. Raises if that question's video or the alignment test fails."""
    from reljoin.herbench_oracle import load_oracle_pil_frames, sha256_file

    if runtime is None:
        runtime = load_runtime(args)
    video_blockers = runtime_blockers(
        getattr(args, "model_dir", ""),
        getattr(args, "videos_root", ""),
        case.get("video_path", ""),
        Path(args.checkpoint),
    )
    video_blockers = [item for item in video_blockers if item.startswith("HERBench video")]
    if video_blockers:
        raise RuntimeBlocked(video_blockers)

    import torch

    from reljoin.losses import letter_token_ids
    from reljoin.ovis_generate import format_mc_prompt, generate_ovis, preprocess_ovis_inputs
    from reljoin.ovis_hooks import (
        begin_prepare_cache,
        encode_question_hidden,
        end_prepare_cache,
        get_hook_state,
        visual_atom_spans,
    )

    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    joining = runtime["joining"]
    localization = runtime["localization"]

    video = _resolve_video(case["video_path"], Path(args.videos_root))
    stamps = [float(value) for value in case["frame_timestamps"]]
    frames, indices, frame_meta = load_oracle_pil_frames(
        video, stamps, float(case["duration"] or frame_meta_duration(case))
    )
    if len(frames) != len(stamps):
        raise RuntimeError(f"Expected {len(stamps)} frames, loaded {len(frames)}")
    prompt = format_mc_prompt(case["question"], case["choices"])
    question_hidden = encode_question_hidden(
        model, tokenizer, case["question"], case["choices"], model.device, model.dtype
    )
    timestamps = torch.tensor([stamps], device=model.device, dtype=torch.float32)
    frame_mask = torch.ones(1, len(stamps), device=model.device, dtype=torch.bool)
    example = bool(getattr(args, "example", False))
    live = bool(getattr(args, "live", False))
    n_choices = len([choice for choice in case["choices"] if str(choice).strip()])
    letter_ids = letter_token_ids(tokenizer, n_choices)

    begin_prepare_cache()
    try:
        prepared = preprocess_ovis_inputs(model, frames, prompt, MIN_PIXELS, MAX_PIXELS)
        if live:
            vanilla = _scores(
                generate_ovis(
                    model, None, "", joining, question_hidden, timestamps, frame_mask,
                    bypass=False, eval_mode="identity", max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS,
                    prepared=prepared, return_scores=True, max_new_tokens=1,
                ),
                letter_ids,
            )
            full_payload = _forward(
                model, joining, prepared, question_hidden, timestamps, frame_mask, letter_ids,
                check_beta=True,
            )
            _assert_beta(full_payload["beta"], full_payload["valid"])
            return _live_result(
                case, args, frames, indices, frame_meta, localization, vanilla, full_payload
            )
        if example:
            full_payload = _forward(
                model, joining, prepared, question_hidden, timestamps, frame_mask, letter_ids,
                check_beta=True,
            )
            _assert_beta(full_payload["beta"], full_payload["valid"])
            return _example_result(case, args, frames, indices, frame_meta, localization, full_payload)
        input_ids, _pixels, grid_thws = prepared
        spans = visual_atom_spans(input_ids)
        tokens_per_frame = [end - start for start, end in spans]
        boxes = _boxes_for_frames(model, frames, grid_thws, tokens_per_frame)
        vanilla = _scores(
            generate_ovis(
                model, None, "", joining, question_hidden, timestamps, frame_mask,
                bypass=False, eval_mode="identity", max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS,
                prepared=prepared, return_scores=True, max_new_tokens=1,
            ),
            letter_ids,
        )
        full_payload = _forward(
            model, joining, prepared, question_hidden, timestamps, frame_mask, letter_ids,
            check_beta=True,
        )
        full = full_payload["scores"]
        alpha = full_payload["alpha"]
        beta = full_payload["beta"]
        valid = full_payload["valid"]
        _assert_beta(beta, valid)
        gold_index = case_index(case)
        if gold_index >= len(full["z"]) or gold_index >= len(vanilla["z"]):
            raise RuntimeError(
                f"{case['question_id']} gold {case.get('gold')!r} is outside the scored letters "
                f"({len(full['z'])})."
            )
        _attach_margin(vanilla, gold_index)
        _attach_margin(full, gold_index)
        selection = select_keyframes(beta[0])
        if not selection["edges"]:
            selection = _fallback_edges(selection, beta[0])
        gamma = full_payload["gamma"]
        bundle = {
            "question_id": case["question_id"],
            "question": case.get("question") or "",
            "gold": case.get("gold"),
            "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
            "localization": localization,
            "prompt": prompt,
            "frame_indices": indices,
            "frame_meta": {key: frame_meta[key] for key in frame_meta if key != "frames"},
            "timestamps": [float(t) for t in case["frame_timestamps"]],
            "boxes": [item.tolist() for item in boxes],
            "tokens_per_frame": tokens_per_frame,
            "alpha": alpha[0].tolist(),
            "beta": beta[0].tolist(),
            "gamma": None if gamma is None else gamma[0].tolist(),
            "gate": full_payload["gate"][0].tolist(),
            "residual_ratio": full_payload["residual_ratio"][0].tolist(),
            "vanilla": _pack_scores(vanilla, case, "vanilla"),
            "frameweave": _pack_scores(full, case, "frameweave"),
            "overall_delta_z": full["z"][gold_index] - vanilla["z"][gold_index],
            "overall_delta_margin": full["margin"] - vanilla["margin"],
            "selection": _jsonify_selection(selection),
            "beta_direction": "arrow j -> i uses beta[i, j]; beta.sum(-1) == 1 on receivers with a neighbor",
            "contribution_definition": "delta_z = z_y(full) - z_y(intervened); positive supports gold",
        }
        if not selection["edges"]:
            bundle["render_allowed"] = False
            bundle["reject_reason"] = "Cross-frame beta has no off-diagonal mass to draw."
            bundle["interpretation"] = _interpretation(case, selection)
            return {"bundle": bundle, "frames": frames, "render": False}

        gold_index = case_index(case)
        trials = []
        for k in K_CANDIDATES:
            token_sets = [
                top_alpha_tokens(alpha[0, frame], valid[0, frame], k)
                for frame in selection["keyframes"]
            ]
            token_impacts = []
            for frame, tokens in zip(selection["keyframes"], token_sets):
                mask = _suppress_mask(valid.shape, frame, tokens)
                scores = _forward(
                    model, joining, prepared, question_hidden, timestamps, frame_mask, letter_ids,
                    suppress_tokens=mask, check_beta=True,
                )["scores"]
                token_impacts.append(_delta(full, scores, gold_index, frame, tokens, alpha[0, frame]))
            edge_impacts = []
            for edge in selection["edges"]:
                pair = [(edge["receiver"], edge["source"])]
                pre = _forward(
                    model, joining, prepared, question_hidden, timestamps, frame_mask, letter_ids,
                    blocked_edges=pair, edge_ablation="pre_softmax", check_beta=True,
                )["scores"]
                post = _forward(
                    model, joining, prepared, question_hidden, timestamps, frame_mask, letter_ids,
                    blocked_edges=pair, edge_ablation="post_softmax", check_beta=False,
                )["scores"]
                edge_impacts.append(
                    {
                        "receiver": edge["receiver"],
                        "source": edge["source"],
                        "beta": edge["beta"],
                        "pre_softmax": _delta(full, pre, gold_index),
                        "post_softmax": _delta(full, post, gold_index),
                    }
                )
            joint_edges = [(edge["receiver"], edge["source"]) for edge in selection["edges"]]
            joint = _forward(
                model, joining, prepared, question_hidden, timestamps, frame_mask, letter_ids,
                blocked_edges=joint_edges, edge_ablation="pre_softmax", check_beta=True,
            )["scores"]
            trials.append(
                {
                    "k": k,
                    "token_sets": token_sets,
                    "token_impacts": token_impacts,
                    "edge_impacts": edge_impacts,
                    "joint": _delta(full, joint, gold_index),
                    "joint_argmax": joint["letter"],
                }
            )
        chosen = _choose_trial(trials)
        display = _display_trial(trials, chosen)
        bundle["k_trials"] = trials
        bundle["chosen_k"] = None if display is None else display["k"]
        bundle["interpretation"] = _interpretation(case, selection)
        if display is None:
            bundle["render_allowed"] = False
            bundle["reject_reason"] = "No keyframe had pooling mass to highlight."
            return {"bundle": bundle, "frames": frames, "render": False}
        bundle["chosen_trial"] = display
        passed = chosen is not None and _trial_supports_gold(chosen)
        if passed:
            raw_impacts = []
            for impact in chosen["token_impacts"]:
                mask = _suppress_mask(valid.shape, impact["frame"], impact["tokens"])
                for fill in ("zero", "mean"):
                    scores = _forward(
                        model, joining, prepared, question_hidden, timestamps, frame_mask, letter_ids,
                        raw_mask_tokens=mask, raw_fill=fill, check_beta=True,
                    )["scores"]
                    raw_impacts.append(
                        {"frame": impact["frame"], "fill": fill, **_delta(full, scores, gold_index)}
                    )
            bundle["raw_visual_impacts"] = raw_impacts
            bundle["displayed_edges"] = [
                edge for edge in chosen["edge_impacts"]
                if edge["pre_softmax"]["delta_z"] > 0 and edge["pre_softmax"]["delta_margin"] > 0
            ]
            bundle["render_allowed"] = True
            bundle["reject_reason"] = ""
        else:
            bundle["render_allowed"] = False
            if not selection["sharp"]:
                bundle["reject_reason"] = (
                    "Cross-frame beta is not concentrated on a directed evidence edge."
                )
            elif chosen is None:
                bundle["reject_reason"] = (
                    "No k in {4, 8, 12} produced positive gold-logit and margin contributions "
                    "on every displayed keyframe and on the dominant directed edge."
                )
            else:
                bundle["reject_reason"] = (
                    "Dominant edge or a highlighted region did not support the gold logit."
                )
            bundle["displayed_edges"] = list(display["edge_impacts"])
        return {"bundle": bundle, "frames": frames, "render": True}
    finally:
        end_prepare_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def case_index(case: dict) -> int:
    return ord(str(case["gold"]).strip().upper()) - ord("A")


def _resolve_video(video_path: str, videos_root: Path | None) -> Path | None:
    from reljoin.herbench_oracle import resolve_video

    if not video_path:
        return None
    root = videos_root or Path(".")
    return resolve_video(video_path, root, root)


def frame_meta_duration(case: dict) -> float:
    stamps = case.get("frame_timestamps") or [1.0]
    return max(float(case.get("duration") or 0.0), float(stamps[-1]), 1.0)


def _boxes_for_frames(model, frames, grid_thws, tokens_per_frame) -> list[np.ndarray]:
    from reljoin.frame_viz.geometry import _grid_rows, introspect_visual_tokenizer

    spec = introspect_visual_tokenizer(model)
    grids = _grid_rows(grid_thws)
    boxes = []
    if len(grids) != len(frames) or len(tokens_per_frame) != len(frames):
        raise AlignmentError("grid_thws or visual-atom spans do not match the 8 frames")
    for frame, grid, count in zip(frames, grids, tokens_per_frame):
        width, height = frame.size
        meta = processor_metadata_for_image(
            (height, width),
            patch_size=int(spec["patch_size"]),
            hidden_stride=int(spec["hidden_stride"]),
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )
        expected = (grid[1] // int(spec["hidden_stride"])) * (grid[2] // int(spec["hidden_stride"]))
        if count != expected:
            raise AlignmentError(
                f"frame has {count} visual atoms, merger expects {expected} from grid {grid}"
            )
        frame_boxes = verified_token_boxes((height, width), meta, grid)
        if frame_boxes.shape[0] != count:
            raise AlignmentError("verified boxes do not match the merged token count")
        boxes.append(frame_boxes)
    return boxes


def _forward(model, joining, prepared, question_hidden, timestamps, frame_mask, letter_ids, **kwargs):
    from reljoin.ovis_generate import generate_ovis
    from reljoin.ovis_hooks import get_hook_state

    payload = generate_ovis(
        model, None, "", joining, question_hidden, timestamps, frame_mask,
        bypass=False, eval_mode="full", max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS,
        prepared=prepared, return_scores=True, max_new_tokens=1, **kwargs,
    )
    state = get_hook_state()
    module = state.module
    alpha = _cpu(module.last_alpha)
    beta = _cpu(module.last_beta)
    gamma = None if module.last_gamma is None else _cpu(module.last_gamma)
    gate = _cpu(module.last_gate)
    delta = _cpu(module.last_delta_x)
    tokens = _cpu(state.last_visual_embeds)
    valid = _cpu(state.last_token_mask).astype(bool)
    ratio = _residual_ratio(delta, tokens)
    return {
        "scores": _scores(payload, letter_ids),
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "gate": gate.reshape(gate.shape[:-1]) if gate.ndim == 4 and gate.shape[-1] == 1 else gate,
        "residual_ratio": ratio,
        "valid": valid,
    }


def _attach_margin(scores: dict, gold_index: int) -> dict:
    scores["margin"] = _margin(scores["z"], gold_index)
    return scores


def _scores(payload, letter_ids) -> dict:
    import torch

    step = payload.get("first_step_logits") if isinstance(payload, dict) else None
    if step is None:
        raise RuntimeError("The forward did not return first-step letter logits.")
    step = step.detach().float().cpu().reshape(-1)
    ids = letter_ids.detach().to(device=step.device, dtype=torch.long).reshape(-1)
    chosen = step.index_select(0, ids)
    index = int(torch.argmax(chosen).item())
    gold_slot = chosen
    probs = torch.softmax(gold_slot, dim=0)
    letters = [chr(ord("A") + i) for i in range(int(chosen.numel()))]
    return {
        "z": [float(value) for value in chosen.tolist()],
        "prob": [float(value) for value in probs.tolist()],
        "letter": letters[index],
        "margin": None,
    }


def _delta(full: dict, other: dict, gold_index: int, frame: int | None = None, tokens=None, alpha_row=None) -> dict:
    z_full = full["z"][gold_index]
    z_other = other["z"][gold_index]
    margin_full = _margin(full["z"], gold_index)
    margin_other = _margin(other["z"], gold_index)
    full["margin"] = margin_full
    other["margin"] = margin_other
    payload = {
        "delta_z": float(z_full - z_other),
        "delta_margin": float(margin_full - margin_other),
        "z_full": float(z_full),
        "z_intervened": float(z_other),
        "margin_full": float(margin_full),
        "margin_intervened": float(margin_other),
        "argmax": other["letter"],
    }
    if frame is not None:
        payload["frame"] = int(frame)
        payload["tokens"] = [int(token) for token in tokens]
        payload["alpha_mass"] = token_mass(alpha_row, list(tokens))
    return payload


def _margin(scores: list[float], gold_index: int) -> float:
    values = list(scores)
    gold = values[gold_index]
    others = [value for index, value in enumerate(values) if index != gold_index]
    return float(gold - max(others))


def _suppress_mask(shape, frame: int, tokens: list[int]):
    import torch

    mask = torch.zeros(shape[-2:], dtype=torch.bool)
    for token in tokens:
        mask[int(frame), int(token)] = True
    return mask


def _cpu(tensor) -> np.ndarray:
    if tensor is None:
        raise RuntimeError("A required adapter tensor was not stored.")
    return tensor.detach().float().cpu().numpy()


def _residual_ratio(delta: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    delta_rms = np.sqrt(np.mean(np.square(delta), axis=-1))
    token_rms = np.sqrt(np.mean(np.square(tokens), axis=-1))
    return delta_rms / (token_rms + 1e-8)


def _assert_beta(beta: np.ndarray, valid: np.ndarray) -> None:
    matrix = np.asarray(beta[0], dtype=np.float64)
    present = valid[0].any(axis=-1)
    for receiver in range(matrix.shape[0]):
        if not present[receiver]:
            continue
        if abs(float(matrix[receiver].sum()) - 1.0) > 1e-3:
            raise AssertionError(
                f"beta receiver row {receiver} sums to {matrix[receiver].sum()}, expected 1. "
                "Arrow j -> i must use beta[i, j]."
            )
        if abs(float(matrix[receiver, receiver])) > 1e-4:
            raise AssertionError("beta diagonal is not closed")


def _live_result(case, args, frames, indices, frame_meta, localization, vanilla_scores, payload) -> dict:
    """Score Vanilla and Hybrid live. Draw only a Vanilla-wrong, Hybrid-correct question."""
    from reljoin.evidence_flow.mine import _choice_text, live_qualifies
    from reljoin.herbench_oracle import sha256_file

    gold = str(case.get("gold") or "").strip().upper()[:1]
    vanilla_letter = str(vanilla_scores.get("letter") or "")
    hybrid_letter = str(payload["scores"].get("letter") or "")
    choices = list(case.get("choices") or [])
    qualified = live_qualifies(vanilla_letter, hybrid_letter, gold)
    bundle = {
        "live": True,
        "qualified": qualified,
        "example": True,
        "question_id": case["question_id"],
        "task_type": case.get("task_type") or "",
        "question": case.get("question") or "",
        "choices": choices,
        "gold": gold,
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "timestamps": [float(value) for value in case["frame_timestamps"]],
        "vanilla": {
            "letter": vanilla_letter,
            "predicted_letter": vanilla_letter,
            "text": _choice_text(choices, vanilla_letter),
        },
        "frameweave": {
            "letter": hybrid_letter,
            "predicted_letter": hybrid_letter,
            "text": _choice_text(choices, hybrid_letter),
        },
        "csv_vanilla_letter": case.get("vanilla_letter") or "",
        "csv_hybrid_letter": case.get("hybrid_letter") or "",
        "render_allowed": False,
        "reject_reason": "",
    }
    if not qualified:
        bundle["reject_reason"] = (
            f"Live Vanilla {vanilla_letter or '?'} / FrameWeave {hybrid_letter or '?'} "
            f"/ gold {gold or '?'}: figure is kept only when Vanilla is wrong and FrameWeave is correct."
        )
        return {"bundle": bundle, "frames": frames, "render": False}
    beta = payload["beta"][0]
    selection = select_keyframes(beta)
    keyframes, edges = _example_view(selection, beta)
    bundle.update({
        "localization": localization,
        "frame_indices": indices,
        "frame_meta": {key: frame_meta[key] for key in frame_meta if key != "frames"},
        "selection": {"keyframes": keyframes},
        "displayed_edges": edges,
        "render_allowed": bool(keyframes),
    })
    if not keyframes:
        bundle["reject_reason"] = "The live answers qualify, but no frames were selected."
        return {"bundle": bundle, "frames": frames, "render": False}
    return {"bundle": bundle, "frames": frames, "render": True}


def _example_result(case, args, frames, indices, frame_meta, localization, payload) -> dict:
    """One Hybrid forward: CSV letters, at most four frames, and two arrows."""
    from reljoin.herbench_oracle import sha256_file

    beta = payload["beta"][0]
    selection = select_keyframes(beta)
    keyframes, edges = _example_view(selection, beta)
    bundle = {
        "example": True,
        "question_id": case["question_id"],
        "task_type": case.get("task_type") or "",
        "question": case.get("question") or "",
        "choices": list(case.get("choices") or []),
        "gold": case.get("gold"),
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "localization": localization,
        "frame_indices": indices,
        "frame_meta": {key: frame_meta[key] for key in frame_meta if key != "frames"},
        "timestamps": [float(value) for value in case["frame_timestamps"]],
        "selection": {"keyframes": keyframes},
        "displayed_edges": edges,
        "vanilla": {
            "letter": case.get("vanilla_letter") or "",
            "text": case.get("vanilla_text") or "",
        },
        "frameweave": {
            "letter": case.get("hybrid_letter") or "",
            "text": case.get("hybrid_text") or "",
        },
        "render_allowed": True,
        "reject_reason": "" if edges else "No off-diagonal relation was strong enough to draw an arrow.",
    }
    return {"bundle": bundle, "frames": frames, "render": True}


def _example_view(selection: dict, beta: np.ndarray, *, max_frames: int = 4, max_edges: int = 2):
    if not selection.get("edges"):
        selection = _fallback_edges(selection, beta)
    edges = sorted(selection.get("edges") or [], key=lambda edge: (-float(edge["beta"]), edge["receiver"], edge["source"]))
    edges = edges[:max_edges]
    touched = []
    for edge in edges:
        for frame in (int(edge["source"]), int(edge["receiver"])):
            if frame not in touched:
                touched.append(frame)
    if len(touched) < 2:
        for frame in selection.get("keyframes") or []:
            if int(frame) not in touched:
                touched.append(int(frame))
            if len(touched) >= 2:
                break
    keyframes = sorted(touched)[:max_frames]
    visible = set(keyframes)
    edges = [
        {"receiver": int(edge["receiver"]), "source": int(edge["source"])}
        for edge in edges
        if int(edge["receiver"]) in visible and int(edge["source"]) in visible
    ]
    return keyframes, edges


def _fallback_edges(selection: dict, beta: np.ndarray) -> dict:
    """Keep one real off-diagonal edge so a diffuse question can still be drawn."""
    matrix = np.asarray(beta, dtype=np.float64)
    best = None
    frames = selection["keyframes"] or list(range(matrix.shape[0]))
    for receiver in frames:
        for source in frames:
            if receiver == source:
                continue
            weight = float(matrix[receiver, source])
            if best is None or weight > best["beta"]:
                best = {"receiver": int(receiver), "source": int(source), "beta": weight}
    if best is None or best["beta"] <= 0:
        return selection
    updated = dict(selection)
    updated["edges"] = [best]
    return updated


def _display_trial(trials: list[dict], chosen: dict | None) -> dict | None:
    if chosen is not None:
        return chosen
    for trial in trials:
        if trial["k"] == 8 and trial["token_impacts"]:
            return trial
    for trial in trials:
        if trial["token_impacts"]:
            return trial
    return None


def _trial_supports_gold(trial: dict) -> bool:
    if not trial["token_impacts"] or not trial["edge_impacts"]:
        return False
    if not all(item["delta_z"] > 0 and item["delta_margin"] > 0 for item in trial["token_impacts"]):
        return False
    if not all(item["alpha_mass"] >= 0.12 for item in trial["token_impacts"]):
        return False
    dominant = max(trial["edge_impacts"], key=lambda edge: edge["beta"])
    pre = dominant["pre_softmax"]
    return pre["delta_z"] > 0 and pre["delta_margin"] > 0


def _choose_trial(trials: list[dict]) -> dict | None:
    viable = []
    for trial in trials:
        if _trial_supports_gold(trial):
            viable.append(trial)
    if not viable:
        return None
    viable.sort(key=lambda trial: (
        -float(np.mean([item["delta_z"] for item in trial["token_impacts"]])),
        trial["k"],
    ))
    return viable[0]


def _pack_scores(scores: dict, case: dict, which: str) -> dict:
    index = case_index(case)
    letter = case["vanilla_letter"] if which == "vanilla" else case["gold"]
    text = case["vanilla_text"] if which == "vanilla" else case["gold_text"]
    return {
        "letter": letter,
        "text": text,
        "predicted_letter": scores["letter"],
        "z_gold": scores["z"][index],
        "prob_gold": scores["prob"][index],
        "margin_gold": _margin(scores["z"], index),
        "z": scores["z"],
        "prob": scores["prob"],
    }


def _jsonify_selection(selection: dict) -> dict:
    payload = dict(selection)
    payload["profiles"] = [
        {key: value for key, value in profile.items() if key != "ranked"}
        | {"top_sources": profile["ranked"][:3]}
        for profile in selection["profiles"]
    ]
    return payload


def _interpretation(case: dict, selection: dict) -> dict:
    stamps = [float(case["frame_timestamps"][frame]) for frame in selection["keyframes"]]
    order = " -> ".join(f"{stamp:.1f}s" for stamp in stamps)
    return {
        "human_assigned": False,
        "lines": [
            "Focus patches: verified top-alpha cells on the displayed frames",
            f"Observed order: {order}",
            "Relation direction: source frame to receiver frame",
        ],
        "note": (
            "These lines locate verified high-α regions and the directed β edges. "
            "They are not probed object or state slots."
        ),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
