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
    resolved = _resolve_video(video_path, Path(videos_root) if videos_root else None)
    if resolved is None:
        blockers.append(
            f"HERBench video for this question was not found ({video_path}). "
            "Pass --videos-root. No frame image was invented."
        )
    return blockers


def extract_question(case: dict, args) -> dict:
    """Run localization, then one question. Raises if the runtime or the alignment test fails."""
    from reljoin.herbench_oracle import load_oracle_pil_frames, sha256_file

    blockers = runtime_blockers(
        getattr(args, "model_dir", ""),
        getattr(args, "videos_root", ""),
        case.get("video_path", ""),
        Path(args.checkpoint),
    )
    if blockers:
        raise RuntimeBlocked(blockers)

    import torch

    from reljoin.config import ExperimentConfig
    from reljoin.losses import letter_token_ids
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

    video = _resolve_video(case["video_path"], Path(args.videos_root))
    frames, indices, frame_meta = load_oracle_pil_frames(
        video, [float(t) for t in case["frame_timestamps"]], float(case["duration"] or frame_meta_duration(case))
    )
    if len(frames) != 8:
        raise RuntimeError(f"Expected 8 frames, loaded {len(frames)}")
    prompt = format_mc_prompt(case["question"], case["choices"])
    question_hidden = encode_question_hidden(
        model, tokenizer, case["question"], case["choices"], model.device, model.dtype
    )
    timestamps = torch.tensor([case["frame_timestamps"]], device=model.device, dtype=torch.float32)
    frame_mask = torch.ones(1, 8, device=model.device, dtype=torch.bool)
    n_choices = len([choice for choice in case["choices"] if str(choice).strip()])
    letter_ids = letter_token_ids(tokenizer, n_choices)

    begin_prepare_cache()
    try:
        prepared = preprocess_ovis_inputs(model, frames, prompt, MIN_PIXELS, MAX_PIXELS)
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
        selection = select_keyframes(beta[0])
        bundle = {
            "question_id": case["question_id"],
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
            "gamma": full_payload["gamma"][0].tolist(),
            "gate": full_payload["gate"][0].tolist(),
            "residual_ratio": full_payload["residual_ratio"][0].tolist(),
            "vanilla": _pack_scores(vanilla, case, "vanilla"),
            "frameweave": _pack_scores(full, case, "frameweave"),
            "overall_delta_z": full["z"][case_index(case)] - vanilla["z"][case_index(case)],
            "overall_delta_margin": full["margin"] - vanilla["margin"],
            "selection": _jsonify_selection(selection),
            "beta_direction": "arrow j -> i uses beta[i, j]; beta.sum(-1) == 1 on receivers with a neighbor",
            "contribution_definition": "delta_z = z_y(full) - z_y(intervened); positive supports gold",
        }
        if not selection["sharp"] or not selection["edges"]:
            bundle["render_allowed"] = False
            bundle["reject_reason"] = "Cross-frame beta is not concentrated on a directed evidence edge."
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
        bundle["k_trials"] = trials
        bundle["chosen_k"] = None if chosen is None else chosen["k"]
        if chosen is None:
            bundle["render_allowed"] = False
            bundle["reject_reason"] = (
                "No k in {4, 8, 12} produced positive gold-logit and margin contributions "
                "on every displayed keyframe and on the dominant directed edge."
            )
            return {"bundle": bundle, "frames": frames, "render": False}

        raw_impacts = []
        for impact in chosen["token_impacts"]:
            mask = _suppress_mask(valid.shape, impact["frame"], impact["tokens"])
            for fill in ("zero", "mean"):
                scores = _forward(
                    model, joining, prepared, question_hidden, timestamps, frame_mask, letter_ids,
                    raw_mask_tokens=mask, raw_fill=fill, check_beta=True,
                )["scores"]
                raw_impacts.append({"frame": impact["frame"], "fill": fill, **_delta(full, scores, gold_index)})
        bundle["raw_visual_impacts"] = raw_impacts
        kept_edges = [
            edge for edge in chosen["edge_impacts"]
            if edge["pre_softmax"]["delta_z"] > 0 and edge["pre_softmax"]["delta_margin"] > 0
        ]
        dominant = max(chosen["edge_impacts"], key=lambda edge: edge["beta"])
        if not (
            dominant["pre_softmax"]["delta_z"] > 0
            and dominant["pre_softmax"]["delta_margin"] > 0
            and kept_edges
            and all(item["delta_z"] > 0 and item["delta_margin"] > 0 for item in chosen["token_impacts"])
        ):
            bundle["render_allowed"] = False
            bundle["reject_reason"] = "Dominant edge or a highlighted region did not support the gold logit."
            bundle["chosen_trial"] = chosen
            return {"bundle": bundle, "frames": frames, "render": False}
        bundle["render_allowed"] = True
        bundle["chosen_trial"] = chosen
        bundle["displayed_edges"] = kept_edges
        bundle["interpretation"] = _interpretation(case, selection)
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
    gamma = _cpu(module.last_gamma)
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


def _scores(payload, letter_ids) -> dict:
    import torch

    step = payload.get("first_step_logits") if isinstance(payload, dict) else None
    if step is None:
        raise RuntimeError("The forward did not return first-step letter logits.")
    ids = letter_ids.to(dtype=torch.long)
    chosen = step.detach().float().cpu().index_select(0, ids)
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


def _choose_trial(trials: list[dict]) -> dict | None:
    viable = []
    for trial in trials:
        if not trial["token_impacts"] or not trial["edge_impacts"]:
            continue
        if not all(item["delta_z"] > 0 and item["delta_margin"] > 0 for item in trial["token_impacts"]):
            continue
        if not all(item["alpha_mass"] >= 0.12 for item in trial["token_impacts"]):
            continue
        dominant = max(trial["edge_impacts"], key=lambda edge: edge["beta"])
        pre = dominant["pre_softmax"]
        if pre["delta_z"] > 0 and pre["delta_margin"] > 0:
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
