"""Overfit / lite training loops for FrameWeave Hybrid v6.0 (V5.1) on PerceptionTest + Ovis2.5-9B."""
from __future__ import annotations

import csv
import gc
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .config import (
    ENABLE_THINKING,
    FORMULATION,
    HYBRID_FAMILY,
    IGNORE_INDEX,
    METHOD_VERSION,
    ExperimentConfig,
    K_TRAIN_CHOICES,
    MAX_PIXELS,
    MIN_PIXELS,
    MODEL_ID,
    MODEL_REVISION,
    VALID_STAGES,
    VALID_VARIANTS,
    architecture_variant,
    make_output_tree,
    normalize_stage,
)
from .data_perception import (
    QASample,
    assert_no_herbench_in_training,
    deterministic_permute,
    download_perception_lite,
    load_and_filter_samples,
    permute_choices,
    save_manifests,
    select_for_mode,
    stable_int_seed,
)
from .frame_io import FrameJob, iter_frame_items
from .losses import (
    batch_reljoin_v5_loss,
    candidate_letter_logits,
    letter_token_ids,
    reljoin_v5_loss,
)
from .ovis_hooks import (
    assert_only_reljoin_trainable,
    begin_prepare_cache,
    configure_frozen_backbone_for_training,
    configure_reljoin,
    enable_frozen_backbone_checkpointing,
    encode_question_hidden,
    end_prepare_cache,
    freeze_backbone,
    get_hook_state,
    install_reljoin_hook,
    ovis_hidden_size,
    patch_qwen_sdpa_attention,
    report_parameter_budget,
    require_flash_attn,
    set_reljoin_context,
)
from .reljoin_modules import build_joining_module, variant_flags


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_stage_trainability(joining, stage: str, variant: str | None = None) -> None:
    """Joint-from-scratch trains the selected variant fully.

    GlobalTimeFix / HybridLocalOff freeze local. Hybrid trains global + local.
    B1/B2 remain optional continuations and are not the V5.1 main method.
    """
    st = normalize_stage(stage)
    arch = str(variant or getattr(joining, "variant", "Hybrid"))
    local_on = stage_local_enabled(arch, st)
    if st == "joint":
        for p in joining.parameters():
            p.requires_grad = True
        if not local_on:
            for p in joining.iter_local_parameters():
                p.requires_grad = False
        return
    if st == "B1":
        for p in joining.parameters():
            p.requires_grad = False
        for p in joining.iter_local_parameters():
            p.requires_grad = True
        return
    if st == "B2":
        for p in joining.parameters():
            p.requires_grad = True
        return
    raise ValueError(f"Unknown stage {stage!r}. Expected {VALID_STAGES}")


def build_optimizer(joining, cfg: ExperimentConfig) -> AdamW:
    lr = float(cfg.learning_rate)
    wd = float(cfg.weight_decay)
    if str(cfg.stage).upper() == "B2":
        groups = []
        local_params = [p for p in joining.iter_local_parameters() if p.requires_grad]
        global_params = [p for p in joining.iter_global_parameters() if p.requires_grad]
        head_params = [p for p in joining.iter_head_parameters() if p.requires_grad]
        if local_params:
            groups.append({"params": local_params, "lr": lr})
        if global_params:
            groups.append({"params": global_params, "lr": 0.1 * lr})
        if head_params:
            groups.append({"params": head_params, "lr": 0.1 * lr})
        if not groups:
            raise RuntimeError("Stage B2 has no trainable FrameWeave parameters")
        return AdamW(groups, weight_decay=wd)
    params = [p for p in joining.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(f"Stage {cfg.stage} has no trainable FrameWeave parameters")
    return AdamW(params, lr=lr, weight_decay=wd)


def load_joining_checkpoint(joining, path: str | Path) -> dict:
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"init_ckpt not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unexpected checkpoint payload in {ckpt_path}: {type(ckpt)}")
    state = ckpt.get("reljoin", ckpt)
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected RelJoin state in {ckpt_path}: {type(state)}")
    joining.load_state_dict(state, strict=True)
    print(
        f"Loaded init_ckpt {ckpt_path} method_version={ckpt.get('method_version')} "
        f"stage={ckpt.get('stage')} variant={ckpt.get('variant')}",
        flush=True,
    )
    return ckpt


def stage_local_enabled(variant: str, stage: str) -> bool:
    """Local follows the variant. Stage no longer force-disables it."""
    del stage
    return bool(variant_flags(variant)["local_enabled"])


def require_init_ckpt(stage: str, init_ckpt: str) -> None:
    st = normalize_stage(stage)
    if st == "B1" and not str(init_ckpt or "").strip():
        raise ValueError("stage B1 requires --init-ckpt (optional continuation, not the V5.1 main method)")


def _answer_ids_tensor(tokenizer, answer_text, device) -> torch.Tensor:
    ids = tokenizer.encode(str(answer_text).strip(), add_special_tokens=False)
    if not ids:
        return torch.zeros(0, device=device, dtype=torch.long)
    return torch.tensor(ids, device=device, dtype=torch.long)


def _set_reljoin_batch_context(batch, tokenizer, frame_mask=None) -> None:
    set_reljoin_context(
        question_hidden=batch["question_hidden"],
        timestamps=batch["timestamps"],
        frame_mask=batch["frame_mask"] if frame_mask is None else frame_mask,
        answer_ids=_answer_ids_tensor(tokenizer, batch["answer_text"], batch["input_ids"].device),
        labels=batch["labels"],
    )


def _candidate_logits_from_outputs(outputs, tokenizer, n_choices: int, labels=None) -> torch.Tensor:
    """z_m = logit(letter_m) at the first answer position. Shape [B, M]."""
    labs = labels if labels is not None else get_hook_state().last_labels
    if labs is None:
        raise RuntimeError("RelJoin labels were not stored. Pass labels into set_reljoin_context.")
    logits = outputs.logits
    if logits.size(1) != labs.size(1):
        raise RuntimeError(f"logits length {logits.size(1)} != labels {labs.size(1)}")
    ids = letter_token_ids(tokenizer, n_choices).to(device=logits.device)
    return candidate_letter_logits(logits, labs, ids, ignore_index=IGNORE_INDEX)


def resolve_model_source(cfg: ExperimentConfig) -> tuple[str, str | None]:
    local = Path(cfg.model_dir)
    if (local / "config.json").is_file():
        return str(local), None
    return MODEL_ID, MODEL_REVISION or None


def load_ovis(cfg: ExperimentConfig):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required — refusing CPU fallback")
    require_flash_attn()
    from transformers import AutoModelForCausalLM

    source, revision = resolve_model_source(cfg)
    print(f"Loading Ovis2.5-9B from {source} revision={revision}", flush=True)
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if revision:
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    model.cuda()
    tokenizer = model.text_tokenizer
    if getattr(cfg, "use_sdpa", True):
        patch_qwen_sdpa_attention(model)
    freeze_backbone(model)
    return model, tokenizer


def count_scheduler_updates(
    n_train: int,
    gradient_accumulation_steps: int,
    max_epochs: int,
    warmup_ratio: float,
) -> tuple[int, int, int]:
    """Count optimizer steps from the real micro-batch=1 dataloader."""
    accum = max(1, int(gradient_accumulation_steps))
    updates_per_epoch = max(1, math.ceil(max(0, int(n_train)) / accum))
    total_updates = max(1, int(max_epochs)) * updates_per_epoch
    warmup_updates = int(round(float(warmup_ratio) * total_updates))
    return updates_per_epoch, total_updates, warmup_updates


def cosine_scheduler(optimizer, num_warmup, num_training):
    def lr_lambda(step):
        if step < num_warmup:
            return float(step) / float(max(1, num_warmup))
        progress = float(step - num_warmup) / float(max(1, num_training - num_warmup))
        progress = min(1.0, max(0.0, progress))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def teacher_forced_letter_correct(outputs, tokenizer, answer_text: str) -> bool:
    letter = str(answer_text).strip().upper()[:1]
    if not letter or getattr(outputs, "logits", None) is None:
        return False
    pred = predicted_letter_from_outputs(outputs, tokenizer, 26)
    return pred == letter


def predicted_letter_from_outputs(outputs, tokenizer, n_choices: int) -> str | None:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: max(0, int(n_choices))]
    if not letters or getattr(outputs, "logits", None) is None:
        return None
    logits = outputs.logits
    if logits.dim() != 3 or logits.size(1) < 2:
        return None
    tail = logits[0, -8:].argmax(dim=-1)
    text = tokenizer.decode(tail.tolist(), skip_special_tokens=True).upper()
    for ch in text:
        if ch in letters:
            return ch
    return None


def letter_to_content(choices: list[str], letter: str | None) -> str | None:
    if letter is None:
        return None
    idx = ord(letter.upper()[:1]) - ord("A")
    if 0 <= idx < len(choices):
        return str(choices[idx]).strip().lower()
    return None


def _percentile(values: list[float], q: float) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    xs = sorted(finite)
    if len(xs) == 1:
        return xs[0]
    idx = (len(xs) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return xs[lo]
    weight = idx - lo
    return xs[lo] * (1.0 - weight) + xs[hi] * weight


def _summarize_floats(values: list[float]) -> dict[str, float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {"mean": float("nan"), "max": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {
        "mean": sum(finite) / len(finite),
        "max": max(finite),
        "p50": _percentile(finite, 0.50),
        "p95": _percentile(finite, 0.95),
    }


def _base_visual_rms() -> float:
    vis = get_hook_state().last_visual_embeds
    if vis is None or vis.numel() == 0:
        return float("nan")
    return float(vis.float().pow(2).mean().sqrt().item())


def sample_K(rng: random.Random) -> int:
    return rng.choice(list(K_TRAIN_CHOICES))


def _train_jobs(samples: list[QASample], seed: int, epoch: int) -> list[FrameJob]:
    rng = random.Random(seed + epoch)
    jobs: list[FrameJob] = []
    for sample in samples:
        K = sample_K(rng)
        new_choices, new_idx, _perm = permute_choices(sample.choices, sample.answer_index, rng)
        jobs.append(FrameJob(sample=sample, K=K, choices=new_choices, answer_index=new_idx))
    return jobs


def _val_jobs(samples: list[QASample]) -> list[FrameJob]:
    jobs: list[FrameJob] = []
    for vs in samples:
        new_c, new_i, _perm = deterministic_permute(
            vs.choices, vs.answer_index, f"val::{vs.sample_id}"
        )
        K = min(8, sample_K(random.Random(stable_int_seed(vs.sample_id))))
        jobs.append(FrameJob(sample=vs, K=K, choices=new_c, answer_index=new_i))
    return jobs


def format_mc_prompt(question: str, choices: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = [question.strip(), ""]
    for i, c in enumerate(choices):
        lines.append(f"{letters[i]}. {c}")
    lines.append("")
    lines.append("Answer with the option letter only.")
    return "\n".join(lines)


def prepare_batch_tensors(
    model,
    item: dict,
    device,
    dtype,
    *,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
    enable_thinking: bool = ENABLE_THINKING,
):
    """Tokenize multi-image messages + question hidden on GPU."""
    frames = item["frames"]
    question = item["question"]
    choices = list(item["choices"])
    answer_index = int(item["answer_index"])
    prompt = format_mc_prompt(question, choices)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    answer_text = letters[answer_index]
    tokenizer = model.text_tokenizer

    content = [{"type": "image", "image": im} for im in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    ans_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    if not ans_ids:
        raise RuntimeError(f"Tokenizer produced empty ids for answer {answer_text!r}")
    ans = torch.tensor(ans_ids, dtype=input_ids.dtype).view(1, -1)
    prompt_len = int(input_ids.shape[1])
    full_ids = torch.cat([input_ids, ans], dim=1)
    labels = torch.full_like(full_ids, IGNORE_INDEX)
    labels[:, prompt_len:] = ans
    attention_mask = torch.ones_like(full_ids)

    full_ids = full_ids.to(device)
    labels = labels.to(device)
    attention_mask = attention_mask.to(device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(device=device, dtype=dtype, non_blocking=True)
    if grid_thws is not None:
        grid_thws = grid_thws.to(device=device, non_blocking=True)

    question_hidden = encode_question_hidden(
        model,
        tokenizer,
        question,
        choices,
        device,
        dtype,
    )
    timestamps = item["timestamps"]
    if timestamps.dim() == 1:
        timestamps = timestamps.unsqueeze(0)
    timestamps = timestamps.to(device, dtype=dtype, non_blocking=True)
    frame_mask = item["frame_mask"]
    if frame_mask.dim() == 1:
        frame_mask = frame_mask.unsqueeze(0)
    frame_mask_b = frame_mask.to(device=device, dtype=torch.bool, non_blocking=True)

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "grid_thws": grid_thws,
        "question_hidden": question_hidden,
        "timestamps": timestamps,
        "frame_mask": frame_mask_b,
        "answer_text": answer_text,
        "answer_index": answer_index,
        "K_real": int(item["K_real"]),
        "sample_id": str(item.get("sample_id") or ""),
        "choices": choices,
        "original_choices": list(item.get("original_choices") or choices),
        "original_answer_index": int(item.get("original_answer_index", answer_index)),
    }


def _same_path(a: Path, b: Path) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a) == str(b)


def _copy_persist_item(src: Path, target: Path) -> None:
    if _same_path(src, target):
        return
    try:
        if src.is_dir():
            shutil.copytree(src, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    except (OSError, shutil.Error) as exc:
        print(f"Persist copy skipped {src} → {target}: {exc}")


def _sync_persist(cfg: ExperimentConfig, paths: dict[str, Path]) -> None:
    dest = Path(cfg.persist_dir)
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Persist dir unavailable ({exc}); skip Drive sync")
        return
    if _same_path(dest, cfg.out_dir):
        print("Persist dir is out_dir; skip Drive sync")
        return
    for key in ("checkpoints_last", "checkpoints_best", "logs", "data"):
        src = paths[key]
        if not src.exists():
            continue
        target = dest / src.relative_to(cfg.out_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_persist_item(src, target)
    for name in ("config.json", "environment.txt", "train_summary.json"):
        src = cfg.out_dir / name
        if src.is_file():
            _copy_persist_item(src, dest / name)


def _residual_metrics(joining) -> dict[str, float]:
    stats = getattr(joining, "last_stats", None) or {}
    st = get_hook_state()
    residual_rms = float(stats.get("residual_rms", float("nan")))
    base_rms = float(stats.get("base_rms", _base_visual_rms()))
    if not math.isfinite(base_rms):
        base_rms = _base_visual_rms()
    if math.isfinite(residual_rms) and math.isfinite(base_rms) and base_rms > 0:
        ratio = residual_rms / base_rms
    else:
        ratio = float("nan")
    return {
        "residual_rms": residual_rms,
        "delta_rms": float(stats.get("delta_rms", float("nan"))),
        "max_g": float(stats.get("max_g", float("nan"))),
        "gate_mean": float(stats.get("gate_mean", float("nan"))),
        "gate_p50": float(stats.get("gate_p50", float("nan"))),
        "gate_p95": float(stats.get("gate_p95", float("nan"))),
        "calib_mean": float(stats.get("calib_mean", float("nan"))),
        "rho_max": float(stats.get("rho_max", float("nan"))),
        "residual_norm": float(st.last_residual_norm),
        "residual_over_base_rms": ratio,
        "global_entropy": float(stats.get("global_entropy", float("nan"))),
        "local_frame_entropy": float(stats.get("local_frame_entropy", float("nan"))),
        "token_match_entropy": float(stats.get("token_match_entropy", float("nan"))),
        "normalized_token_match_entropy": float(
            stats.get("normalized_token_match_entropy", float("nan"))
        ),
        "local_logit_abs_mean": float(stats.get("local_logit_abs_mean", float("nan"))),
        "temporal_over_content": float(stats.get("temporal_over_content", float("nan"))),
        "local_ratio": float(stats.get("local_ratio", float("nan"))),
        "time_key_ratio": float(stats.get("time_key_ratio", float("nan"))),
        "time_value_ratio": float(stats.get("time_value_ratio", float("nan"))),
        "absolute_position_ratio": float(stats.get("absolute_position_ratio", float("nan"))),
        "base_rms": base_rms,
    }


def _warn_residual_conditions(rmetrics: dict[str, float], joining, *, warn_budget: dict[str, int]) -> None:
    delta = getattr(joining, "last_delta_x", None)
    if delta is not None and not bool(torch.isfinite(delta).all()):
        raise RuntimeError("Non-finite residual")
    nte = rmetrics.get("normalized_token_match_entropy", float("nan"))
    if math.isfinite(nte) and nte < 0.10 and warn_budget.get("collapse", 0) > 0:
        print("WARN: Local token attention may be collapsing", flush=True)
        warn_budget["collapse"] -= 1
    local_ratio = rmetrics.get("local_ratio", float("nan"))
    if math.isfinite(local_ratio) and local_ratio < 1e-4 and warn_budget.get("inactive", 0) > 0:
        print("WARN: Local branch may be inactive", flush=True)
        warn_budget["inactive"] -= 1
    over = rmetrics.get("residual_over_base_rms", float("nan"))
    if math.isfinite(over) and over > 1.0 and warn_budget.get("scale", 0) > 0:
        print("WARN: Residual exceeds base-token scale", flush=True)
        warn_budget["scale"] -= 1


def _params_grad_l2(params) -> float:
    total = 0.0
    found = False
    for p in params:
        if p is None or p.grad is None:
            continue
        found = True
        total += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total) if found else float("nan")


GRAD_GROUP_KEYS = (
    "grad_question",
    "grad_frame_pooler",
    "grad_global_relation",
    "grad_local_qk",
    "grad_local_value",
    "grad_local_message",
    "grad_local_score",
    "grad_local_adapter",
    "grad_output_projection",
)


def _grad_group_norms(joining) -> dict[str, float]:
    return {
        "grad_question": _params_grad_l2(joining.question_encoder.parameters()),
        "grad_frame_pooler": _params_grad_l2(joining.frame_pooler.parameters()),
        "grad_global_relation": _params_grad_l2(
            list(joining.global_relation.parameters()) + list(joining.abs_position.parameters())
        ),
        "grad_local_qk": _params_grad_l2(joining.iter_local_qk_parameters()),
        "grad_local_value": _params_grad_l2(joining.iter_local_value_parameters()),
        "grad_local_message": _params_grad_l2(joining.iter_local_message_parameters()),
        "grad_local_score": _params_grad_l2(joining.iter_local_score_parameters()),
        "grad_local_adapter": _params_grad_l2(joining.head.local_adapter.parameters()),
        "grad_output_projection": _params_grad_l2(joining.head.W_o.parameters()),
    }


METRIC_FIELDS = [
    "variant",
    "run_name",
    "stage",
    "epoch",
    "ans_nll",
    "loss_v5",
    "loss_cand",
    "loss_corr",
    "n_vanilla_correct",
    "n_vanilla_wrong",
    "train_acc",
    "val_nll",
    "val_acc",
    "vanilla_val_nll",
    "vanilla_val_acc",
    "delta_val_nll",
    "val_loss_cand",
    "val_loss_corr",
    "val_n_vanilla_correct",
    "paired_wins",
    "paired_losses",
    "paired_ties",
    "perm_consistency",
    "grad_norm_mean",
    "grad_norm_max",
    "grad_norm_p95",
    "residual_rms_mean",
    "residual_rms_max",
    "residual_rms_p95",
    "gate_mean",
    "gate_p50",
    "gate_p95",
    "calib_mean",
    "rho_max",
    "residual_over_base_rms_mean",
    "global_entropy_mean",
    "local_frame_entropy_mean",
    "token_match_entropy_mean",
    "normalized_token_match_entropy_mean",
    "local_logit_abs_mean",
    "temporal_over_content_mean",
    "local_ratio_mean",
    "time_key_ratio_mean",
    "time_value_ratio_mean",
    "absolute_position_ratio_mean",
    "grad_question",
    "grad_frame_pooler",
    "grad_global_relation",
    "grad_local_qk",
    "grad_local_value",
    "grad_local_message",
    "grad_local_score",
    "grad_local_adapter",
    "grad_output_projection",
    "val_acc_k3",
    "val_acc_k4",
    "val_acc_k5",
    "val_acc_k6",
    "val_acc_k7",
    "val_acc_k8",
    "val_nll_k3",
    "val_nll_k4",
    "val_nll_k5",
    "val_nll_k6",
    "val_nll_k7",
    "val_nll_k8",
    "lr",
    "gpu_mem_gb",
    "skip_count",
    "updates_per_epoch",
    "global_step",
]


def _model_forward(model, batch):
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
        grid_thws=batch["grid_thws"],
        labels=batch["labels"],
        use_cache=False,
    )


def _forward_candidates(model, tokenizer, batch, joining, *, bypass: bool):
    """One multimodal forward → candidate letter logits [1, M]."""
    configure_reljoin(joining, enabled=True, bypass=bypass)
    _set_reljoin_batch_context(batch, tokenizer)
    outputs = _model_forward(model, batch)
    z = _candidate_logits_from_outputs(outputs, tokenizer, len(batch["choices"]), labels=batch["labels"])
    return outputs, z


def _target_index(batch, device) -> torch.Tensor:
    return torch.tensor([int(batch["answer_index"])], device=device, dtype=torch.long)


def _pred_letter_from_z(z: torch.Tensor, n_choices: int) -> str | None:
    if z.numel() == 0 or not bool(torch.isfinite(z).all().item()):
        return None
    idx = int(z[0].argmax().item())
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: max(0, int(n_choices))]
    if 0 <= idx < len(letters):
        return letters[idx]
    return None


def train_variant(cfg: ExperimentConfig, variant: str | None = None) -> Path:
    variant = architecture_variant(variant or cfg.variant)
    stage = normalize_stage(cfg.stage)
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Expected one of {VALID_VARIANTS}")
    if stage not in {"joint", "B1", "B2"}:
        raise ValueError(f"Unknown stage {stage!r}. Expected {VALID_STAGES}")
    if stage in {"B1", "B2"} and variant in HYBRID_FAMILY:
        variant = "Hybrid"
    if int(cfg.micro_batch_size) != 1:
        raise ValueError(
            "micro_batch_size>1 is not implemented; keep 1 until a real collate exists."
        )
    require_init_ckpt(stage, str(cfg.init_ckpt or ""))
    paths = make_output_tree(cfg.out_dir)
    set_seeds(cfg.seed)
    cfg.variant = variant
    cfg.stage = stage  # type: ignore[assignment]
    run_name = cfg.run_name
    run_log_dir = paths["logs"] / run_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_log_dir / "config.json")
    cfg.save(cfg.out_dir / "config.json")
    local_on = stage_local_enabled(variant, stage)

    print(
        f"\n=== Training FrameWeave Hybrid v6.0/Ovis formulation={FORMULATION} "
        f"variant={variant} stage={stage} local={local_on} run={run_name} mode={cfg.run_mode} ==="
    )
    if stage == "joint" and local_on and not str(cfg.init_ckpt or "").strip():
        print(
            "Joint from scratch: Global + Local both trainable, single LR, "
            f"checkpoint {run_name}.pt (no init_ckpt). Local has no A_t; P_L is small-Xavier."
        )
    ann, videos_dir = download_perception_lite(cfg.train_data_dir)
    samples, excluded = load_and_filter_samples(ann, videos_dir)
    train_s, val_s = select_for_mode(samples, cfg.run_mode, seed=cfg.seed)
    assert_no_herbench_in_training(train_s + val_s)
    save_manifests(train_s, val_s, excluded, paths["data"])
    print(f"Train n={len(train_s)} Val n={len(val_s)}")

    model, tokenizer = load_ovis(cfg)
    llm_dim = ovis_hidden_size(model)
    model_dtype = next(model.parameters()).dtype
    if cfg.dtype != str(model_dtype).replace("torch.", ""):
        print(f"Overriding cfg.dtype={cfg.dtype} → model dtype={model_dtype}")
    dtype = model_dtype
    joining = build_joining_module(
        llm_dim,
        variant=variant,
        rel_dim=cfg.rel_dim,
        residual_eps=cfg.residual_eps,
        rms_eps=cfg.rms_eps,
        time_eps=cfg.time_eps,
        local_token_chunk=int(getattr(cfg, "local_token_chunk", 0) or 0),
        local_enabled=local_on,
    ).to(device=model.device, dtype=torch.float32)
    if cfg.init_ckpt:
        load_joining_checkpoint(joining, cfg.init_ckpt)
        joining.default_local_enabled = local_on

    model.reljoin = joining
    install_reljoin_hook(model)
    freeze_backbone(model)
    apply_stage_trainability(joining, stage, variant=variant)
    trainable_names = assert_only_reljoin_trainable(model, joining)
    if getattr(cfg, "use_gradient_checkpointing", True):
        enable_frozen_backbone_checkpointing(model)
    configure_frozen_backbone_for_training(model)
    joining.train()
    budget = report_parameter_budget(model, joining)
    print("Trainable params:", budget["trainable_count"], f"({budget['trainable_pct']:.4f}%)")
    print("Using dtype:", dtype, "llm_dim:", llm_dim)
    print(
        "FrameWeave dtype: float32; V5.1 loss: logsumexp(L_cand, L_corr)-log2; "
        "logits in fp32. Ovis backbone frozen."
    )
    print(
        "VRAM opts:",
        f"micro_batch={cfg.micro_batch_size}",
        f"accum={cfg.gradient_accumulation_steps}",
        f"effective_batch={cfg.effective_batch_size}",
        f"max_pixels={cfg.max_pixels}",
        f"sdpa={getattr(cfg, 'use_sdpa', True)}",
        f"grad_ckpt={getattr(cfg, 'use_gradient_checkpointing', True)}",
        f"num_workers={getattr(cfg, 'num_workers', 0)}",
        f"local_chunk_p={getattr(cfg, 'local_token_chunk', 0)}",
    )
    print("NOTE: val_acc is argmax over candidate letter logits (not greedy generate)")
    wo_zero = float(joining.W_o.weight.detach().abs().sum().cpu())
    wo_bias = float(joining.W_o.bias.detach().abs().sum().cpu()) if joining.W_o.bias is not None else 0.0
    pl_zero = float(joining.head.local_adapter.weight.detach().abs().sum().cpu())
    wl_zero = float(joining.local.w_L.detach().abs().sum().cpu())
    wabs = float(joining.abs_position.weight.detach().abs().sum().cpu())
    ak = float(joining.global_relation.A_K.weight.detach().abs().sum().cpu())
    print(
        f"W_o abs-sum (expect 0 at clean init): {wo_zero}; bias {wo_bias}; "
        f"P_L {pl_zero} (expect >0 small-Xavier); w_L {wl_zero}; W_abs {wabs}; A_K {ak}"
    )
    select_note = "PerceptionTest val L_V5.1"
    (run_log_dir / "val_acc_note.txt").write_text(
        "val_acc / vanilla_val_acc: argmax over candidate letter logits at the first "
        "answer position (not greedy generate).\n"
        "V5.1 loss is per-sample logsumexp(L_cand, L_corr)-log2, then batch mean. "
        "Vanilla scores are stopgrad hook-bypass on the same frames, prompt, and "
        "option permutation. Residual stats are taken after FrameWeave, before Vanilla. "
        f"Epoch selection uses {select_note} only — never HERBench.\n"
        f"stage={stage} variant={variant} run_name={run_name} "
        f"method_version={METHOD_VERSION} formulation={FORMULATION}. "
        "Backbone: ATH-MaaS/Ovis2.5-9B. Thinking mode off.\n",
        encoding="utf-8",
    )
    shutil.copy2(run_log_dir / "val_acc_note.txt", paths["logs"] / "val_acc_note.txt")
    (paths["diagnostics"] / "gradient_report.json").write_text(
        json.dumps(
            {
                "trainable_names": trainable_names,
                "W_o_abs_sum_init": wo_zero,
                "W_o_bias_abs_sum_init": wo_bias,
                "P_L_abs_sum": pl_zero,
                "w_L_abs_sum": wl_zero,
                "W_abs_abs_sum": wabs,
                "A_K_abs_sum": ak,
                "stage": stage,
                "variant": variant,
                "local_enabled": local_on,
                "llm_dim": llm_dim,
                "method_version": METHOD_VERSION,
                "formulation": FORMULATION,
                **{k: v for k, v in budget.items() if k != "trainable_names"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    optimizer = build_optimizer(joining, cfg)

    max_epochs = cfg.overfit_max_epochs if cfg.run_mode == "overfit" else cfg.lite_epochs
    if cfg.run_mode == "smoke":
        max_epochs = 1
    updates_per_epoch, total_updates, warmup_updates = count_scheduler_updates(
        len(train_s),
        cfg.gradient_accumulation_steps,
        max_epochs,
        cfg.warmup_ratio,
    )
    scheduler = cosine_scheduler(optimizer, warmup_updates, total_updates)
    print(
        f"Schedule: updates/epoch={updates_per_epoch} total={total_updates} "
        f"warmup={warmup_updates} epochs={max_epochs} "
        f"patience={cfg.early_stop_patience}"
    )

    device = model.device
    num_workers = int(getattr(cfg, "num_workers", 0))
    metrics_path = run_log_dir / "train_metrics.csv"
    mem_path = run_log_dir / "memory_profile.csv"
    legacy_metrics = paths["logs"] / "train_metrics.csv"
    legacy_mem = paths["logs"] / "memory_profile.csv"

    with metrics_path.open("w", newline="", encoding="utf-8") as mf, mem_path.open(
        "w", newline="", encoding="utf-8"
    ) as memf:
        mwriter = csv.DictWriter(mf, fieldnames=METRIC_FIELDS)
        memwriter = csv.DictWriter(memf, fieldnames=["variant", "run_name", "epoch", "gpu_mem_gb", "time"])
        mwriter.writeheader()
        memwriter.writeheader()

        best_val = float("inf")
        epochs_no_improve = 0
        best_path = paths["checkpoints_best"] / f"{run_name}.pt"
        last_path = paths["checkpoints_last"] / f"{run_name}.pt"
        first_ans_nll: float | None = None
        global_step = 0

        for epoch in range(1, max_epochs + 1):
            joining.train()
            configure_reljoin(joining, enabled=True, bypass=False)
            tot_v5 = 0.0
            tot_train = 0.0
            tot_cand = 0.0
            tot_corr = 0.0
            n_van_ok = 0
            n_van_wrong = 0
            correct = 0
            n_seen = 0
            n_acc_probes = 0
            n_skip = 0
            optimizer.zero_grad(set_to_none=True)
            grad_norms: list[float] = []
            residual_rms_hist: list[float] = []
            gate_hist: list[float] = []
            calib_hist: list[float] = []
            rho_hist: list[float] = []
            ratio_hist: list[float] = []
            global_ent_hist: list[float] = []
            local_ent_hist: list[float] = []
            match_ent_hist: list[float] = []
            norm_match_ent_hist: list[float] = []
            local_logit_hist: list[float] = []
            toc_hist: list[float] = []
            local_ratio_hist: list[float] = []
            time_key_hist: list[float] = []
            time_value_hist: list[float] = []
            abs_pos_hist: list[float] = []
            grad_group_hists: dict[str, list[float]] = {k: [] for k in GRAD_GROUP_KEYS}
            warn_budget = {"collapse": 3, "inactive": 3, "scale": 3}

            n_train = len(train_s)
            for i, item in enumerate(iter_frame_items(_train_jobs(train_s, cfg.seed, epoch), num_workers)):
                batch = prepare_batch_tensors(
                    model,
                    item,
                    device,
                    dtype,
                    min_pixels=cfg.min_pixels,
                    max_pixels=cfg.max_pixels,
                    enable_thinking=cfg.enable_thinking,
                )
                _set_reljoin_batch_context(batch, tokenizer)

                accum = cfg.gradient_accumulation_steps
                skipped = False
                begin_prepare_cache()
                try:
                    outputs, z_rel = _forward_candidates(
                        model, tokenizer, batch, joining, bypass=False
                    )
                    rmetrics = _residual_metrics(joining)
                    _warn_residual_conditions(rmetrics, joining, warn_budget=warn_budget)
                    target = _target_index(batch, device)
                    with torch.no_grad():
                        _outputs_van, z_van = _forward_candidates(
                            model, tokenizer, batch, joining, bypass=True
                        )
                    loss_v5, loss_cand, loss_corr = batch_reljoin_v5_loss(
                        z_rel, z_van.detach(), target
                    )
                    van_ok = (z_van.float().detach().argmax(dim=-1) == target)
                    loss = loss_v5
                    loss_ok = bool(torch.isfinite(loss).item())
                    if n_seen < 3 or not loss_ok:
                        print(
                            f"[v6] epoch={epoch} i={i} stage={stage} "
                            f"train={float(loss.detach()) if loss.numel() else 'empty'} "
                            f"cand={float(loss_cand.detach())} corr={float(loss_corr.detach())} "
                            f"vanilla_correct={int(van_ok.sum().item())} "
                            f"logits={tuple(outputs.logits.shape)} "
                            f"zR={tuple(z_rel.shape)} "
                            f"P_i={get_hook_state().last_tokens_per_frame} "
                            f"K={get_hook_state().last_num_frames}"
                        )
                    if not loss_ok:
                        n_skip += 1
                        skipped = True
                        del outputs, z_rel, z_van, loss, loss_v5, loss_cand, loss_corr, van_ok, _outputs_van
                    else:
                        n_acc_probes += 1
                        pred = _pred_letter_from_z(z_rel, len(batch["choices"]))
                        gold = str(batch["answer_text"]).strip().upper()[:1]
                        if pred == gold:
                            correct += 1
                        train_f = float(loss.detach())
                        v5_f = float(loss_v5.detach())
                        cand_f = float(loss_cand.detach())
                        corr_f = float(loss_corr.detach())
                        n_ok = int(van_ok.sum().item())
                        n_wrong = int(van_ok.numel() - n_ok)
                        (loss / accum).backward()
                        residual_rms_hist.append(rmetrics["residual_rms"])
                        gate_hist.append(rmetrics["gate_mean"])
                        calib_hist.append(rmetrics["calib_mean"])
                        rho_hist.append(rmetrics["rho_max"])
                        ratio_hist.append(rmetrics["residual_over_base_rms"])
                        global_ent_hist.append(rmetrics["global_entropy"])
                        local_ent_hist.append(rmetrics["local_frame_entropy"])
                        match_ent_hist.append(rmetrics["token_match_entropy"])
                        norm_match_ent_hist.append(rmetrics["normalized_token_match_entropy"])
                        local_logit_hist.append(rmetrics["local_logit_abs_mean"])
                        toc_hist.append(rmetrics["temporal_over_content"])
                        local_ratio_hist.append(rmetrics["local_ratio"])
                        time_key_hist.append(rmetrics["time_key_ratio"])
                        time_value_hist.append(rmetrics["time_value_ratio"])
                        abs_pos_hist.append(rmetrics["absolute_position_ratio"])
                        del outputs, z_rel, z_van, loss, loss_v5, loss_cand, loss_corr, van_ok, _outputs_van
                finally:
                    end_prepare_cache()
                    configure_reljoin(joining, enabled=True, bypass=False)

                if not skipped:
                    tot_train += train_f
                    tot_v5 += v5_f
                    tot_cand += cand_f
                    tot_corr += corr_f
                    n_van_ok += n_ok
                    n_van_wrong += n_wrong
                    n_seen += 1

                if (i + 1) % cfg.gradient_accumulation_steps == 0 or (i + 1) == n_train:
                    grad_norm_val = float(
                        torch.nn.utils.clip_grad_norm_(
                            joining.parameters(), cfg.max_grad_norm
                        )
                    )
                    grad_norms.append(grad_norm_val)
                    groups = _grad_group_norms(joining)
                    for key in GRAD_GROUP_KEYS:
                        grad_group_hists[key].append(groups[key])
                    for n, p in model.named_parameters():
                        if p.requires_grad and not n.startswith("reljoin"):
                            raise AssertionError(f"Unexpected trainable: {n}")
                        if (not n.startswith("reljoin")) and p.grad is not None:
                            if float(p.grad.abs().sum()) > 0:
                                raise AssertionError(f"Frozen param got grad: {n}")
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                del batch, item

            train_acc = correct / max(n_acc_probes, 1)
            val_stats = {
                "val_nll": 0.0,
                "val_acc": 0.0,
                "vanilla_val_nll": float("nan"),
                "vanilla_val_acc": float("nan"),
                "delta_val_nll": float("nan"),
                "val_loss_cand": float("nan"),
                "val_loss_corr": float("nan"),
                "val_n_vanilla_correct": 0,
                "paired_wins": 0,
                "paired_losses": 0,
                "paired_ties": 0,
                "perm_consistency": float("nan"),
            }
            for k in range(3, 9):
                val_stats[f"val_acc_k{k}"] = float("nan")
                val_stats[f"val_nll_k{k}"] = float("nan")

            if val_s:
                end_prepare_cache()
                joining.eval()
                v_nlls: list[float] = []
                v_cands: list[float] = []
                v_corrs: list[float] = []
                v_correct = 0
                van_nlls: list[float] = []
                van_correct = 0
                wins = losses = ties = 0
                cons_ok = cons_n = 0
                by_k: dict[int, dict[str, list[float]]] = {
                    k: {"nll": [], "acc": []} for k in range(3, 9)
                }
                per_sample: list[dict] = []
                with torch.no_grad():
                    for item in iter_frame_items(_val_jobs(val_s), num_workers):
                        batch = prepare_batch_tensors(
                            model,
                            item,
                            device,
                            dtype,
                            min_pixels=cfg.min_pixels,
                            max_pixels=cfg.max_pixels,
                            enable_thinking=cfg.enable_thinking,
                        )
                        begin_prepare_cache()
                        try:
                            _, z_rel = _forward_candidates(
                                model, tokenizer, batch, joining, bypass=False
                            )
                            rmetrics = _residual_metrics(joining)
                            tokens_per_frame = list(get_hook_state().last_tokens_per_frame)
                            _, z_van = _forward_candidates(
                                model, tokenizer, batch, joining, bypass=True
                            )
                        finally:
                            end_prepare_cache()
                        target = _target_index(batch, device)
                        loss_each, loss_cand, loss_corr = reljoin_v5_loss(
                            z_rel, z_van, target
                        )
                        rj_nll = float(loss_each[0].detach()) if torch.isfinite(loss_each).all() else float("nan")
                        cand_nll = float(loss_cand[0].detach()) if torch.isfinite(loss_cand).all() else float("nan")
                        corr_nll = float(loss_corr[0].detach()) if torch.isfinite(loss_corr).all() else float("nan")
                        van_ce = torch.nn.functional.cross_entropy(
                            z_van.float(), target, reduction="none"
                        )
                        van_nll = float(van_ce[0].detach()) if torch.isfinite(van_ce).all() else float("nan")
                        n_ch = len(batch["choices"])
                        rj_pred = _pred_letter_from_z(z_rel, n_ch)
                        van_pred = _pred_letter_from_z(z_van, n_ch)
                        gold = str(batch["answer_text"]).strip().upper()[:1]
                        rj_ok = rj_pred == gold
                        van_ok = van_pred == gold
                        if math.isfinite(rj_nll):
                            v_nlls.append(rj_nll)
                        if math.isfinite(cand_nll):
                            v_cands.append(cand_nll)
                        if math.isfinite(corr_nll):
                            v_corrs.append(corr_nll)
                        if rj_ok:
                            v_correct += 1
                        if math.isfinite(van_nll):
                            van_nlls.append(van_nll)
                        if van_ok:
                            van_correct += 1
                        if math.isfinite(rj_nll) and math.isfinite(van_nll):
                            if rj_nll < van_nll - 1e-8:
                                wins += 1
                            elif van_nll < rj_nll - 1e-8:
                                losses += 1
                            else:
                                ties += 1

                        k_real = int(item["K_real"])
                        if k_real in by_k and math.isfinite(rj_nll):
                            by_k[k_real]["nll"].append(rj_nll)
                            by_k[k_real]["acc"].append(1.0 if rj_ok else 0.0)

                        consistent = None
                        orig_choices = list(item.get("original_choices") or batch["choices"])
                        orig_idx = int(item.get("original_answer_index", 0))
                        if len(orig_choices) >= 2:
                            alt_c, alt_i, _perm = deterministic_permute(
                                orig_choices,
                                orig_idx,
                                f"val-alt::{item['sample_id']}",
                            )
                            item_alt = dict(item)
                            item_alt["choices"] = alt_c
                            item_alt["answer_index"] = alt_i
                            batch_alt = prepare_batch_tensors(
                                model,
                                item_alt,
                                device,
                                dtype,
                                min_pixels=cfg.min_pixels,
                                max_pixels=cfg.max_pixels,
                                enable_thinking=cfg.enable_thinking,
                            )
                            begin_prepare_cache()
                            try:
                                _, z_alt = _forward_candidates(
                                    model, tokenizer, batch_alt, joining, bypass=False
                                )
                            finally:
                                end_prepare_cache()
                            alt_pred = _pred_letter_from_z(z_alt, len(alt_c))
                            c1 = letter_to_content(batch["choices"], rj_pred)
                            c2 = letter_to_content(alt_c, alt_pred)
                            consistent = c1 is not None and c1 == c2
                            cons_n += 1
                            if consistent:
                                cons_ok += 1
                            del batch_alt, item_alt, z_alt

                        per_sample.append(
                            {
                                "sample_id": batch["sample_id"],
                                "K_real": k_real,
                                "tokens_per_frame": tokens_per_frame,
                                "reljoin_nll": rj_nll,
                                "reljoin_cand": cand_nll,
                                "reljoin_corr": corr_nll,
                                "reljoin_pred": rj_pred,
                                "reljoin_correct": rj_ok,
                                "vanilla_nll": van_nll,
                                "vanilla_pred": van_pred,
                                "vanilla_correct": van_ok,
                                "perm_consistent": consistent,
                                **rmetrics,
                            }
                        )
                        del batch, item, z_rel, z_van

                val_stats["val_nll"] = sum(v_nlls) / max(len(v_nlls), 1)
                val_stats["val_acc"] = v_correct / max(len(val_s), 1)
                val_stats["val_loss_cand"] = sum(v_cands) / max(len(v_cands), 1) if v_cands else float("nan")
                val_stats["val_loss_corr"] = sum(v_corrs) / max(len(v_corrs), 1) if v_corrs else float("nan")
                val_stats["val_n_vanilla_correct"] = van_correct
                val_stats["vanilla_val_nll"] = (
                    sum(van_nlls) / max(len(van_nlls), 1) if van_nlls else float("nan")
                )
                val_stats["vanilla_val_acc"] = van_correct / max(len(val_s), 1)
                if v_nlls and van_nlls:
                    val_stats["delta_val_nll"] = val_stats["val_nll"] - val_stats["vanilla_val_nll"]
                val_stats["paired_wins"] = wins
                val_stats["paired_losses"] = losses
                val_stats["paired_ties"] = ties
                val_stats["perm_consistency"] = cons_ok / cons_n if cons_n else float("nan")
                for k in range(3, 9):
                    accs = by_k[k]["acc"]
                    nlls = by_k[k]["nll"]
                    val_stats[f"val_acc_k{k}"] = sum(accs) / len(accs) if accs else float("nan")
                    val_stats[f"val_nll_k{k}"] = sum(nlls) / len(nlls) if nlls else float("nan")
                val_epoch_path = run_log_dir / f"val_epoch_{epoch}.json"
                val_epoch_path.write_text(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "stage": stage,
                            "variant": variant,
                            "run_name": run_name,
                            "samples": per_sample,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                if run_name in {"GlobalTimeFix", "Hybrid", "RelJoin"}:
                    shutil.copy2(val_epoch_path, paths["logs"] / f"val_epoch_{epoch}.json")
                joining.train()
                configure_reljoin(joining, enabled=True, bypass=False)

            gn = _summarize_floats(grad_norms)
            rr = _summarize_floats(residual_rms_hist)
            gg = _summarize_floats(gate_hist)
            cc = _summarize_floats(calib_hist)
            rh = _summarize_floats(rho_hist)
            mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
            combined = tot_v5 / max(n_seen, 1)
            row = {
                "variant": variant,
                "run_name": run_name,
                "stage": stage,
                "epoch": epoch,
                "ans_nll": tot_train / max(n_seen, 1),
                "loss_v5": combined,
                "loss_cand": tot_cand / max(n_seen, 1),
                "loss_corr": tot_corr / max(n_seen, 1),
                "n_vanilla_correct": n_van_ok,
                "n_vanilla_wrong": n_van_wrong,
                "train_acc": train_acc,
                "lr": scheduler.get_last_lr()[0],
                "gpu_mem_gb": mem,
                "skip_count": n_skip,
                "updates_per_epoch": updates_per_epoch,
                "global_step": global_step,
                "grad_norm_mean": gn["mean"],
                "grad_norm_max": gn["max"],
                "grad_norm_p95": gn["p95"],
                "residual_rms_mean": rr["mean"],
                "residual_rms_max": rr["max"],
                "residual_rms_p95": rr["p95"],
                "gate_mean": gg["mean"],
                "gate_p50": gg["p50"],
                "gate_p95": gg["p95"],
                "calib_mean": cc["mean"],
                "rho_max": rh["max"],
                "residual_over_base_rms_mean": _summarize_floats(ratio_hist)["mean"],
                "global_entropy_mean": _summarize_floats(global_ent_hist)["mean"],
                "local_frame_entropy_mean": _summarize_floats(local_ent_hist)["mean"],
                "token_match_entropy_mean": _summarize_floats(match_ent_hist)["mean"],
                "normalized_token_match_entropy_mean": _summarize_floats(norm_match_ent_hist)["mean"],
                "local_logit_abs_mean": _summarize_floats(local_logit_hist)["mean"],
                "temporal_over_content_mean": _summarize_floats(toc_hist)["mean"],
                "local_ratio_mean": _summarize_floats(local_ratio_hist)["mean"],
                "time_key_ratio_mean": _summarize_floats(time_key_hist)["mean"],
                "time_value_ratio_mean": _summarize_floats(time_value_hist)["mean"],
                "absolute_position_ratio_mean": _summarize_floats(abs_pos_hist)["mean"],
                **{k: _summarize_floats(grad_group_hists[k])["mean"] for k in GRAD_GROUP_KEYS},
                **val_stats,
            }
            mwriter.writerow(row)
            mf.flush()
            memwriter.writerow(
                {
                    "variant": variant,
                    "run_name": run_name,
                    "epoch": epoch,
                    "gpu_mem_gb": mem,
                    "time": time.time(),
                }
            )
            memf.flush()
            if run_name in {"GlobalTimeFix", "Hybrid", "RelJoin"}:
                shutil.copy2(metrics_path, legacy_metrics)
                shutil.copy2(mem_path, legacy_mem)
            print(row)

            ckpt = {
                "reljoin": joining.state_dict(),
                "variant": variant,
                "run_name": run_name,
                "stage": stage,
                "local_enabled": local_on,
                "epoch": epoch,
                "config": cfg.to_dict(),
                "model_revision": MODEL_REVISION,
                "model_id": MODEL_ID,
                "method": "FrameWeave-Hybrid",
                "method_version": METHOD_VERSION,
                "formulation": FORMULATION,
                "backbone": "Ovis2.5-9B",
                "val_nll": row["val_nll"],
                "val_loss_v5": row.get("val_nll"),
            }
            epoch_dir = paths["checkpoints"] / f"epoch_{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, epoch_dir / f"{run_name}.pt")
            torch.save(ckpt, last_path)
            if val_s:
                metric_for_best = row["val_nll"]
            else:
                metric_for_best = row["ans_nll"]
            if not math.isfinite(metric_for_best):
                metric_for_best = float("inf")
            if metric_for_best <= best_val:
                best_val = metric_for_best
                epochs_no_improve = 0
                torch.save(ckpt, best_path)
                print(f"Saved best → {best_path}")
            else:
                epochs_no_improve += 1
            _sync_persist(cfg, paths)

            if first_ans_nll is None and math.isfinite(row["ans_nll"]):
                first_ans_nll = row["ans_nll"]
            if cfg.run_mode == "overfit" and first_ans_nll and first_ans_nll > 0:
                drop = (first_ans_nll - row["ans_nll"]) / first_ans_nll
                explosion = (
                    (not math.isfinite(row["grad_norm_mean"]))
                    or row["grad_norm_mean"] > 1e3
                    or (
                        math.isfinite(row["residual_rms_mean"])
                        and row["residual_rms_mean"] > 1e3
                    )
                )
                print(
                    f"Overfit gate: objective_drop={drop:.3f} train_acc={train_acc:.3f} "
                    f"explosion={explosion}"
                )
                if (not explosion) and drop >= 0.30 and train_acc >= 0.90:
                    print("Overfit engineering target reached; stopping early.")
                    break
            if (
                cfg.run_mode in {"lite", "full"}
                and val_s
                and epochs_no_improve >= int(cfg.early_stop_patience)
            ):
                print(
                    f"Early stop: PerceptionTest val {select_note} did not improve for "
                    f"{epochs_no_improve} epochs (best={best_val:.4f})."
                )
                break

    del model, joining
    gc.collect()
    torch.cuda.empty_cache()
    return best_path if best_path.exists() else last_path


def main_train(cfg: ExperimentConfig) -> None:
    cfg.variant = architecture_variant(cfg.variant)
    cfg.save()
    if cfg.run_mode not in {"overfit", "smoke", "lite", "full"}:
        raise ValueError(f"train not for mode {cfg.run_mode}")
    path = train_variant(cfg, cfg.variant)
    results = {cfg.run_name: str(path), "stage": cfg.stage, "variant": cfg.variant}
    (cfg.out_dir / "train_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Training complete:", results)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--run-mode", default="overfit")
    p.add_argument("--root", default="/content")
    p.add_argument("--variant", default="Hybrid")
    p.add_argument("--stage", default="joint", choices=list(VALID_STAGES))
    p.add_argument("--init-ckpt", default="")
    args = p.parse_args()
    cfg = ExperimentConfig(
        run_mode=args.run_mode,
        root=Path(args.root),
        variant=args.variant,
        stage=args.stage,
        init_ckpt=args.init_ckpt,
    )
    main_train(cfg)
