"""Shared Ovis2.5 generation + MC prompt protocol for HERBench oracle eval."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from .diagnostics import parse_choice_strict

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
EVAL_MAX_NEW_TOKENS = 32


def format_mc_prompt(question: str, choices: Sequence[str]) -> str:
    lines = [str(question).strip(), ""]
    for i, c in enumerate(choices):
        if str(c).strip():
            lines.append(f"{LETTERS[i]}. {c}")
    lines.append("")
    lines.append("Answer with the option letter only.")
    return "\n".join(lines)


def sha256_ids(input_ids: torch.Tensor) -> str:
    arr = input_ids.detach().to("cpu").contiguous()
    return hashlib.sha256(arr.numpy().tobytes()).hexdigest()[:16]


def decode_generated_ids(output_ids, input_ids, tokenizer) -> str:
    sequences = output_ids.sequences if hasattr(output_ids, "sequences") else output_ids
    seq = sequences[0]
    prompt_len = int(input_ids.shape[-1])
    new_ids = seq[prompt_len:] if int(seq.shape[-1]) > prompt_len else seq
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def preprocess_ovis_inputs(model, frames, prompt: str, min_pixels: int, max_pixels: int):
    content = [{"type": "image", "image": im} for im in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=False,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    device = model.device
    input_ids = input_ids.to(device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(device=device, dtype=model.dtype)
    if grid_thws is not None:
        grid_thws = grid_thws.to(device)
    return input_ids, pixel_values, grid_thws


def generate_ovis(
    model,
    frames,
    prompt: str,
    joining,
    question_hidden,
    timestamps,
    frame_mask,
    *,
    bypass: bool,
    max_pixels: int,
    min_pixels: int,
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
    eval_mode: str | None = None,
    return_meta: bool = False,
    inject_delta=None,
    prepared: tuple | None = None,
    return_scores: bool = False,
    blocked_sources=None,
    suppress_tokens=None,
    blocked_edges=None,
    edge_ablation: str = "pre_softmax",
    check_beta: bool = False,
    raw_mask_tokens=None,
    raw_fill: str = "zero",
) -> Any:
    """Greedy letter generation for HERBench oracle eval.

    Vanilla = hook bypass. RelJoin = eval_mode full.
    ``inject_delta`` writes X + that tensor and does not rerun the residual head.
    ``prepared`` reuses one ``(input_ids, pixel_values, grid_thws)`` so every
    condition adds onto the same captured visual tokens.
    ``return_scores`` adds the first-step logits used for letter scores.
    Callers that omit it still receive the decoded text.
    Does not take ground_truth.
    """
    from .ovis_hooks import configure_reljoin, set_reljoin_context

    configure_reljoin(
        joining,
        enabled=joining is not None or inject_delta is not None,
        bypass=bypass,
        eval_mode=eval_mode or "full",
        inject_delta=inject_delta,
        blocked_sources=blocked_sources,
        suppress_tokens=suppress_tokens,
        blocked_edges=blocked_edges,
        edge_ablation=edge_ablation,
        check_beta=check_beta,
        raw_mask_tokens=raw_mask_tokens,
        raw_fill=raw_fill,
    )
    set_reljoin_context(
        question_hidden=question_hidden,
        timestamps=timestamps,
        frame_mask=frame_mask,
    )
    if prepared is None:
        input_ids, pixel_values, grid_thws = preprocess_ovis_inputs(
            model, frames, prompt, min_pixels, max_pixels
        )
    else:
        input_ids, pixel_values, grid_thws = prepared
    tokenizer = model.text_tokenizer
    gen_kwargs = {
        "inputs": input_ids,
        "pixel_values": pixel_values,
        "grid_thws": grid_thws,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if return_scores:
        gen_kwargs["output_scores"] = True
        gen_kwargs["return_dict_in_generate"] = True
    with torch.inference_mode():
        try:
            output_ids = model.generate(**gen_kwargs, enable_thinking=False)
        except TypeError:
            output_ids = model.generate(**gen_kwargs)
    sequences = output_ids.sequences if hasattr(output_ids, "sequences") else output_ids
    seq = sequences[0]
    prompt_len = int(input_ids.shape[-1])
    new_ids = seq[prompt_len:] if int(seq.shape[-1]) > prompt_len else seq
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    first_new_token_id = int(new_ids[0].item()) if int(new_ids.numel()) else None
    if not return_meta and not return_scores:
        return text
    payload = {
        "text": text,
        "input_ids_hash": sha256_ids(input_ids),
        "raw_output_len": len(text),
        "n_new_ids": int(new_ids.shape[-1]),
        "prompt_len": prompt_len,
        "first_new_token_id": first_new_token_id,
    }
    if return_scores:
        first_step = None
        scores = getattr(output_ids, "scores", None) or ()
        if scores:
            step = scores[0]
            if hasattr(step, "dim") and int(step.dim()) == 2:
                step = step[0]
            if hasattr(step, "detach"):
                first_step = step.detach().float().cpu()
        payload["first_step_logits"] = first_step
    return payload


def parse_choice_eval(text: str, n_choices: int = 5) -> tuple[str, bool]:
    """Fallback parser identical to HERBench oracle eval when base_vlm is absent."""
    pred, ok = parse_choice_strict(text, n_choices=n_choices)
    return pred, ok


def load_herbench_prompt_parser(
    path: str | Path | None,
    expected_sha256: str = "",
) -> tuple[Callable, Callable, dict]:
    """Return (format_prompt, parse_choice, info)."""
    info = {"used_base_vlm": False, "path": "", "sha256": ""}
    if path:
        p = Path(path)
        if p.is_file():
            from .herbench_oracle import load_module_from_file, sha256_file

            digest = sha256_file(p)
            if expected_sha256 and digest != expected_sha256:
                raise RuntimeError(
                    f"HERBench base_vlm.py SHA256 mismatch: {digest} != {expected_sha256}"
                )
            mod = load_module_from_file("herbench_base_pinned_audit", p)

            def format_prompt(question: str, choices: Sequence[str]) -> str:
                return mod.BaseVLM._format_prompt(None, question, list(choices))

            def parse_choice(text: str, n_choices: int = 5) -> tuple[str, bool]:
                pred = mod.BaseVLM.extract_answer_choice(None, str(text))
                pred = "" if pred is None else str(pred).strip()
                if pred.upper() == "ERROR":
                    pred = ""
                if pred:
                    letter = pred.strip().upper()[:1]
                    letters = LETTERS[:n_choices]
                    if letter in letters:
                        return letter, True
                return parse_choice_strict(text, n_choices=n_choices)

            info.update({"used_base_vlm": True, "path": str(p), "sha256": digest})
            return format_prompt, parse_choice, info

    return format_mc_prompt, parse_choice_eval, info


def resolve_herbench_base_vlm(herbench_root: Path, explicit: str = "") -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    root = Path(herbench_root)
    candidates.extend(
        [
            root / "evaluation" / "model_wrappers" / "base_vlm.py",
            root / "model_wrappers" / "base_vlm.py",
            root / "base_vlm.py",
        ]
    )
    for p in candidates:
        if p.is_file():
            return p
    return None
