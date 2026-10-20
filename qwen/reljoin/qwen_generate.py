"""Qwen2.5-VL generation + MC prompt protocol for FrameWeave v6 HERBench eval."""
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


def qwen_processor(model):
    proc = getattr(model, "_reljoin_processor", None)
    if proc is None:
        raise RuntimeError("Qwen processor missing. load_qwen must attach model._reljoin_processor.")
    return proc


def sha256_ids(input_ids: torch.Tensor) -> str:
    arr = input_ids.detach().to("cpu").contiguous()
    return hashlib.sha256(arr.numpy().tobytes()).hexdigest()[:16]


def decode_generated_ids(output_ids, input_ids, tokenizer) -> str:
    sequences = output_ids.sequences if hasattr(output_ids, "sequences") else output_ids
    seq = sequences[0]
    prompt_len = int(input_ids.shape[-1])
    new_ids = seq[prompt_len:] if int(seq.shape[-1]) > prompt_len else seq
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def preprocess_qwen_inputs(model, frames, prompt: str, min_pixels: int, max_pixels: int):
    from qwen_vl_utils import process_vision_info

    processor = qwen_processor(model)
    content = []
    for im in frames:
        content.append(
            {
                "type": "image",
                "image": im,
                "min_pixels": int(min_pixels),
                "max_pixels": int(max_pixels),
            }
        )
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = model.device
    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def generate_qwen(
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
) -> Any:
    """Greedy letter generation for HERBench oracle eval.

    Vanilla = hook bypass. FrameWeave = eval_mode full. Does not take ground_truth.
    """
    from .qwen_hooks import configure_reljoin, set_reljoin_context

    configure_reljoin(
        joining,
        enabled=joining is not None,
        bypass=bypass,
        eval_mode=eval_mode or "full",
    )
    set_reljoin_context(
        question_hidden=question_hidden,
        timestamps=timestamps,
        frame_mask=frame_mask,
    )
    inputs = preprocess_qwen_inputs(model, frames, prompt, min_pixels, max_pixels)
    processor = qwen_processor(model)
    tokenizer = processor.tokenizer
    gen_kwargs = dict(inputs)
    gen_kwargs.update(
        {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
    )
    prompt_len = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)
    sequences = output_ids.sequences if hasattr(output_ids, "sequences") else output_ids
    seq = sequences[0]
    new_ids = seq[prompt_len:] if int(seq.shape[-1]) > prompt_len else seq
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    if not return_meta:
        return text
    return {
        "text": text,
        "input_ids_hash": sha256_ids(inputs["input_ids"]),
        "raw_output_len": len(text),
        "n_new_ids": int(new_ids.shape[-1]),
        "prompt_len": prompt_len,
    }


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
