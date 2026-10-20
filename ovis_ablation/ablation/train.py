"""Train ablation adapters on the shared schedule. Vanilla is not trained."""
from __future__ import annotations

import math
import platform
import random
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from .common import ensure_v6_on_path, git_revision, record_blocker, write_json
from .config import TRAINABLE_VARIANTS, AblationConfig
from .modules import build_adapter, count_parameters, parameter_manifest, select_token_mlp_hidden
from .paths import ResolvedPaths
from .schedule import load_schedule_files

ensure_v6_on_path()

from reljoin.config import ExperimentConfig  # noqa: E402
from reljoin.herbench_oracle import extract_frames_rgb  # noqa: E402
from reljoin.frame_io import frames_to_pil  # noqa: E402
from reljoin.losses import batch_reljoin_v5_loss  # noqa: E402
from reljoin.ovis_hooks import (  # noqa: E402
    assert_only_reljoin_trainable,
    begin_prepare_cache,
    configure_frozen_backbone_for_training,
    configure_reljoin,
    enable_frozen_backbone_checkpointing,
    end_prepare_cache,
    freeze_backbone,
    install_reljoin_hook,
)
from reljoin.train import (  # noqa: E402
    _candidate_logits_from_outputs,
    _set_reljoin_batch_context,
    _target_index,
    cosine_scheduler,
    count_scheduler_updates,
    load_ovis,
    prepare_batch_tensors,
    set_seeds,
)

LOSS_FN = batch_reljoin_v5_loss


def environment_snapshot(cfg: AblationConfig) -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": git_revision(),
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "enable_thinking": cfg.enable_thinking,
        "min_pixels": cfg.min_pixels,
        "max_pixels": cfg.max_pixels,
        "dtype": cfg.dtype,
        "adapter_dtype": cfg.adapter_dtype,
        "use_sdpa": cfg.use_sdpa,
        "use_gradient_checkpointing": cfg.use_gradient_checkpointing,
    }
    try:
        import torch as torch_mod

        info["torch"] = torch_mod.__version__
        info["cuda"] = torch_mod.version.cuda
        info["cuda_available"] = bool(torch_mod.cuda.is_available())
        if torch_mod.cuda.is_available():
            info["gpu"] = torch_mod.cuda.get_device_name(0)
        else:
            info["gpu"] = "NONE"
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = str(exc)
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = "missing"
    try:
        import flash_attn

        info["flash_attn"] = getattr(flash_attn, "__version__", "present")
    except Exception:
        info["flash_attn"] = "missing"
    return info


def variant_dir(cfg: AblationConfig, variant: str) -> Path:
    return cfg.output_dir() / f"seed_{cfg.seed}" / variant


def _runtime_blockers(resolved: ResolvedPaths) -> list[str]:
    reasons = []
    if not torch.cuda.is_available():
        reasons.append("CUDA is not available. Refusing a CPU fallback for Ovis2.5-9B.")
    try:
        import flash_attn  # noqa: F401
    except Exception:
        reasons.append("flash-attn is missing. The canonical Ovis loader imports it and this suite does not install packages.")
    return reasons


def _frames_blockers(bundle: dict) -> list[str]:
    meta = bundle["meta"]["frames"]
    if not meta.get("probed"):
        return ["Frame indices were not probed. Run prepare on the machine that has the PerceptionTest videos."]
    missing = meta.get("missing_videos") or {}
    if missing:
        preview = list(missing)[:8]
        return [f"{len(missing)} videos missing or unreadable. First ids: {preview}. See shared/missing_videos.json."]
    if int(meta.get("rows_ready", 0)) != int(meta.get("rows_total", -1)):
        return ["Schedule frame plans are incomplete."]
    return []


def _v6_config(cfg: AblationConfig, resolved: ResolvedPaths) -> ExperimentConfig:
    v6 = ExperimentConfig(run_mode="lite", root=cfg.output_dir())
    if resolved.model_dir:
        v6.model_dir = Path(resolved.model_dir)
    v6.use_sdpa = bool(cfg.use_sdpa)
    v6.use_gradient_checkpointing = bool(cfg.use_gradient_checkpointing)
    v6.min_pixels = int(cfg.min_pixels)
    v6.max_pixels = int(cfg.max_pixels)
    v6.enable_thinking = bool(cfg.enable_thinking)
    return v6


def _decode_cached(cache: dict, path: str, indices: list[int]):
    key = (path, tuple(int(index) for index in indices))
    if key not in cache:
        cache[key] = frames_to_pil(extract_frames_rgb(Path(path), list(key[1])))
    return cache[key]


def row_to_item(row: dict, cache: dict) -> dict:
    if not row.get("frames_ready"):
        raise RuntimeError(f"Sample {row['sample_id']} has no frame plan ({row.get('frame_error')}).")
    frames = _decode_cached(cache, str(row["video_path"]), list(row["frame_indices"]))
    return {
        "frames": frames,
        "question": row["question"],
        "choices": list(row["choices"]),
        "answer_index": int(row["answer_index"]),
        "timestamps": torch.tensor(row["timestamps"], dtype=torch.float32),
        "frame_mask": torch.ones(len(row["timestamps"]), dtype=torch.bool),
        "K_real": int(row["K"]),
        "sample_id": row["sample_id"],
        "original_choices": list(row["original_choices"]),
        "original_answer_index": int(row["original_answer_index"]),
    }


def forward_loss(model, tokenizer, batch, module, *, require_grad: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    begin_prepare_cache()
    try:
        configure_reljoin(module, enabled=True, bypass=False)
        _set_reljoin_batch_context(batch, tokenizer)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            grid_thws=batch["grid_thws"],
            labels=batch["labels"],
            use_cache=False,
        )
        if require_grad and not outputs.logits.requires_grad:
            raise RuntimeError("Adapter logits were detached. The multimodal pass must keep autograd into the adapter.")
        z_theta = _candidate_logits_from_outputs(
            outputs, tokenizer, len(batch["choices"]), labels=batch["labels"]
        )
        with torch.no_grad():
            configure_reljoin(module, enabled=True, bypass=True)
            _set_reljoin_batch_context(batch, tokenizer)
            vanilla_out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                grid_thws=batch["grid_thws"],
                labels=batch["labels"],
                use_cache=False,
            )
            z_zero = _candidate_logits_from_outputs(
                vanilla_out, tokenizer, len(batch["choices"]), labels=batch["labels"]
            )
    finally:
        end_prepare_cache()
        configure_reljoin(module, enabled=True, bypass=False)
    target = _target_index(batch, z_theta.device)
    return LOSS_FN(z_theta, z_zero.detach(), target)


def _save_adapter(path: Path, module, *, variant: str, epoch: int, seed: int, llm_dim: int, rel_dim: int, hidden, val_loss: float, cfg: AblationConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": module.state_dict(),
            "variant": variant,
            "epoch": epoch,
            "seed": seed,
            "llm_dim": llm_dim,
            "rel_dim": rel_dim,
            "token_mlp_hidden": hidden,
            "val_loss": val_loss,
            "model_id": cfg.model_id,
            "model_revision": cfg.model_revision,
            "trainable_params": count_parameters(module),
        },
        path,
    )


def _load_module(path: Path, device) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    module = build_adapter(
        payload["variant"],
        int(payload["llm_dim"]),
        int(payload["rel_dim"]),
        hidden=payload.get("token_mlp_hidden"),
    )
    if module is None:
        raise RuntimeError(f"Checkpoint {path} is vanilla and has no adapter")
    module.load_state_dict(payload["state_dict"], strict=True)
    return module.to(device=device, dtype=torch.float32), payload


def train_variants(
    cfg: AblationConfig,
    resolved: ResolvedPaths,
    variants: list[str],
    *,
    resume: bool,
) -> dict:
    reasons = _runtime_blockers(resolved)
    try:
        bundle = load_schedule_files(cfg)
    except FileNotFoundError:
        reasons.append("Shared schedule is missing. Run prepare first.")
        bundle = None
    if bundle is not None:
        reasons.extend(_frames_blockers(bundle))
    if reasons:
        path = record_blocker(cfg.output_dir(), "train", reasons)
        print("TRAIN BLOCKED:\n- " + "\n- ".join(reasons))
        print(f"Wrote {path}")
        return {"status": "blocked", "reasons": reasons}
    selected = [name for name in variants if name in TRAINABLE_VARIANTS]
    if not selected:
        raise ValueError("No trainable variant selected. Vanilla has no adapter to train.")
    env = environment_snapshot(cfg)
    write_json(cfg.output_dir() / "shared" / "environment.json", env)
    v6_cfg = _v6_config(cfg, resolved)
    model, tokenizer = load_ovis(v6_cfg)
    freeze_backbone(model)
    install_reljoin_hook(model)
    if cfg.use_gradient_checkpointing:
        enable_frozen_backbone_checkpointing(model)
    configure_frozen_backbone_for_training(model)
    from reljoin.ovis_hooks import ovis_hidden_size

    llm_dim = ovis_hidden_size(model)
    if llm_dim != cfg.hidden_size:
        raise RuntimeError(f"Ovis hidden size {llm_dim} != config {cfg.hidden_size}")
    hidden = cfg.token_mlp_hidden
    if "token_mlp" in selected and hidden is None:
        hidden = int(select_token_mlp_hidden(llm_dim, cfg.rel_dim, cfg.rms_eps, cfg.time_eps)["hidden"])
    updates_per_epoch, total_updates, warmup_updates = count_scheduler_updates(
        cfg.expected_train_questions,
        cfg.gradient_accumulation_steps,
        cfg.max_epochs,
        cfg.warmup_ratio,
    )
    runs = {}
    for name in selected:
        set_seeds(cfg.seed)
        module = build_adapter(
            name,
            llm_dim,
            cfg.rel_dim,
            hidden=hidden if name == "token_mlp" else None,
            rms_eps=cfg.rms_eps,
            time_eps=cfg.time_eps,
        )
        module = module.to(device=model.device, dtype=torch.float32)
        model.reljoin = module
        assert_only_reljoin_trainable(model, module)
        optimizer = AdamW(module.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = cosine_scheduler(optimizer, warmup_updates, total_updates)
        run_dir = variant_dir(cfg, name)
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = parameter_manifest(module, name)
        manifest["hybrid_reference"] = select_token_mlp_hidden(llm_dim, cfg.rel_dim, cfg.rms_eps, cfg.time_eps)
        write_json(run_dir / "parameter_manifest.json", manifest)
        write_json(run_dir / "config.json", cfg.to_dict())
        write_json(run_dir / "environment.json", env)
        state = {
            "module": module,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "next_epoch": 1,
            "global_step": 0,
            "best_val": float("inf"),
            "best_epoch": 0,
            "epochs_no_improve": 0,
            "stopped": False,
            "hidden": hidden if name == "token_mlp" else None,
        }
        resume_path = run_dir / "resume.pt"
        if resume and resume_path.is_file():
            payload = torch.load(resume_path, map_location="cpu", weights_only=False)
            if payload.get("schedule_sha256") != bundle["meta"]["schedule_sha256"]:
                raise RuntimeError(f"Resume schedule hash mismatch for {name}")
            module.load_state_dict(payload["state_dict"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            state.update(
                {
                    "next_epoch": int(payload["next_epoch"]),
                    "global_step": int(payload["global_step"]),
                    "best_val": float(payload["best_val"]),
                    "best_epoch": int(payload["best_epoch"]),
                    "epochs_no_improve": int(payload["epochs_no_improve"]),
                    "stopped": bool(payload["stopped"]),
                }
            )
            random.setstate(payload["python_random"])
            torch.set_rng_state(payload["torch_random"])
            if torch.cuda.is_available() and payload.get("cuda_random") is not None:
                torch.cuda.set_rng_state_all(payload["cuda_random"])
        runs[name] = state
        print(
            f"Initialized {name}: trainable={manifest['trainable']} active={manifest['active']}",
            flush=True,
        )

    train_rows = bundle["train_rows"]
    val_rows = bundle["val_rows"]
    for epoch in range(1, cfg.max_epochs + 1):
        active = [name for name, state in runs.items() if not state["stopped"] and state["next_epoch"] == epoch]
        if not active:
            continue
        epoch_rows = [row for row in train_rows if int(row["epoch"]) == epoch]
        if len(epoch_rows) != cfg.expected_train_questions:
            raise RuntimeError(f"Epoch {epoch} has {len(epoch_rows)} rows, expected {cfg.expected_train_questions}")
        cache: dict = {}
        items = [row_to_item(row, cache) for row in epoch_rows]
        for name in active:
            _run_epoch(cfg, model, tokenizer, runs[name], name, items, epoch, updates_per_epoch, bundle["meta"]["schedule_sha256"])
        del items, cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del model
    return {"status": "ok", "variants": selected}


def _run_epoch(cfg, model, tokenizer, state, variant, items, epoch, updates_per_epoch, schedule_sha) -> None:
    module = state["module"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    module.train()
    model.reljoin = module
    optimizer.zero_grad(set_to_none=True)
    accum = int(cfg.gradient_accumulation_steps)
    pending = 0
    running = 0.0
    seen = 0
    run_dir = variant_dir(cfg, variant)
    log_path = run_dir / "train.jsonl"
    for index, item in enumerate(items):
        batch = prepare_batch_tensors(
            model,
            item,
            model.device,
            next(model.parameters()).dtype,
            min_pixels=cfg.min_pixels,
            max_pixels=cfg.max_pixels,
            enable_thinking=cfg.enable_thinking,
        )
        loss, _cand, _corr = forward_loss(model, tokenizer, batch, module)
        (loss / accum).backward()  # tail group is also divided by accum, matching V6
        pending += 1
        seen += 1
        running += float(loss.detach())
        if pending == accum or index + 1 == len(items):
            for name, param in model.named_parameters():
                if param.requires_grad and not name.startswith("reljoin") and id(param) not in {id(p) for p in module.parameters()}:
                    raise AssertionError(f"Backbone parameter became trainable: {name}")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.max_grad_norm))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["global_step"] += 1
            pending = 0
            from .common import append_jsonl

            append_jsonl(
                log_path,
                {
                    "event": "update",
                    "variant": variant,
                    "epoch": epoch,
                    "global_step": state["global_step"],
                    "loss": running / seen,
                    "grad_norm": grad_norm,
                    "lr": scheduler.get_last_lr()[0],
                },
            )
        del batch
    if state["global_step"] <= 0 or (seen != len(items)):
        raise RuntimeError(f"{variant} epoch {epoch} did not consume every sample")
    val_loss = _validate(cfg, model, tokenizer, module, variant)
    improved = val_loss <= state["best_val"]
    if improved:
        state["best_val"] = val_loss
        state["best_epoch"] = epoch
        state["epochs_no_improve"] = 0
    else:
        state["epochs_no_improve"] += 1
    _save_adapter(
        run_dir / "last" / "adapter.pt",
        module,
        variant=variant,
        epoch=epoch,
        seed=cfg.seed,
        llm_dim=module.llm_dim,
        rel_dim=module.rel_dim,
        hidden=state["hidden"],
        val_loss=val_loss,
        cfg=cfg,
    )
    if improved:
        _save_adapter(
            run_dir / "best" / "adapter.pt",
            module,
            variant=variant,
            epoch=epoch,
            seed=cfg.seed,
            llm_dim=module.llm_dim,
            rel_dim=module.rel_dim,
            hidden=state["hidden"],
            val_loss=val_loss,
            cfg=cfg,
        )
        write_json(
            run_dir / "selected.json",
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "checkpoint": str(run_dir / "best" / "adapter.pt"),
                "rule": "lowest validation L; ties keep the later epoch",
            },
        )
    state["next_epoch"] = epoch + 1
    if state["epochs_no_improve"] >= int(cfg.early_stop_patience):
        state["stopped"] = True
        print(f"Early stop {variant} at epoch {epoch}; best epoch {state['best_epoch']} L={state['best_val']:.4f}", flush=True)
    torch.save(
        {
            "state_dict": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "next_epoch": state["next_epoch"],
            "global_step": state["global_step"],
            "best_val": state["best_val"],
            "best_epoch": state["best_epoch"],
            "epochs_no_improve": state["epochs_no_improve"],
            "stopped": state["stopped"],
            "schedule_sha256": schedule_sha,
            "python_random": random.getstate(),
            "torch_random": torch.get_rng_state(),
            "cuda_random": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        run_dir / "resume.pt",
    )
    from .common import append_jsonl

    append_jsonl(
        run_dir / "val.jsonl",
        {
            "variant": variant,
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val": state["best_val"],
            "best_epoch": state["best_epoch"],
            "global_step": state["global_step"],
            "updates_this_epoch_expected": updates_per_epoch,
            "stopped": state["stopped"],
        },
    )
    print(
        f"{variant} epoch {epoch}: val_L={val_loss:.4f} best={state['best_val']:.4f} step={state['global_step']}",
        flush=True,
    )


def _validate(cfg, model, tokenizer, module, variant: str) -> float:
    module.eval()
    bundle_rows = load_schedule_files(cfg)["val_rows"]
    cache: dict = {}
    losses = []
    with torch.no_grad():
        for row in bundle_rows:
            item = row_to_item(row, cache)
            batch = prepare_batch_tensors(
                model,
                item,
                model.device,
                next(model.parameters()).dtype,
                min_pixels=cfg.min_pixels,
                max_pixels=cfg.max_pixels,
                enable_thinking=cfg.enable_thinking,
            )
            loss, _, _ = forward_loss(model, tokenizer, batch, module, require_grad=False)
            if not math.isfinite(float(loss)):
                raise RuntimeError(f"Non-finite validation loss for {variant} sample {row['sample_id']}")
            losses.append(float(loss))
            del batch
    if len(losses) != len(bundle_rows):
        raise RuntimeError("Validation did not score every scheduled row")
    return sum(losses) / len(losses)


def smoke_variants(cfg: AblationConfig, resolved: ResolvedPaths, variants: list[str]) -> dict:
    reasons = _runtime_blockers(resolved)
    try:
        bundle = load_schedule_files(cfg)
    except FileNotFoundError:
        reasons.append("Shared schedule is missing. Run prepare first.")
        bundle = None
    if bundle is not None:
        reasons.extend(_frames_blockers(bundle))
    if reasons:
        path = record_blocker(cfg.output_dir(), "smoke", reasons)
        report = {
            "paper_result": False,
            "status": "blocked",
            "reasons": reasons,
        }
        write_json(cfg.output_dir() / "smoke" / "smoke_report.json", report)
        print("SMOKE BLOCKED:\n- " + "\n- ".join(reasons))
        print(f"Wrote {path}")
        return report
    selected = [name for name in variants if name in TRAINABLE_VARIANTS]
    v6_cfg = _v6_config(cfg, resolved)
    model, tokenizer = load_ovis(v6_cfg)
    freeze_backbone(model)
    install_reljoin_hook(model)
    from reljoin.ovis_hooks import ovis_hidden_size

    llm_dim = ovis_hidden_size(model)
    hidden = cfg.token_mlp_hidden
    if any(name == "token_mlp" for name in selected) and hidden is None:
        hidden = int(select_token_mlp_hidden(llm_dim, cfg.rel_dim, cfg.rms_eps, cfg.time_eps)["hidden"])
    epoch_rows = [row for row in bundle["train_rows"] if int(row["epoch"]) == 1][:2]
    if len(epoch_rows) < 1:
        raise RuntimeError("Smoke found no scheduled training rows")
    if len(epoch_rows) == 1:
        epoch_rows = epoch_rows + epoch_rows
    cache: dict = {}
    items = [row_to_item(row, cache) for row in epoch_rows]
    reports = []
    for name in selected:
        set_seeds(cfg.seed)
        module = build_adapter(
            name,
            llm_dim,
            cfg.rel_dim,
            hidden=hidden if name == "token_mlp" else None,
            rms_eps=cfg.rms_eps,
            time_eps=cfg.time_eps,
        ).to(device=model.device, dtype=torch.float32)
        model.reljoin = module
        assert_only_reljoin_trainable(model, module)
        optimizer = AdamW(module.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        upstream_name = {
            "token_mlp": "B_x.weight",
            "local_only": "local.local_q.weight",
            "frame_only": "frame_pooler.token_key.weight",
        }.get(name, "global_relation.W_Q.weight")
        upstream = dict(module.named_parameters())[upstream_name]
        for step_index, item in enumerate(items, start=1):
            optimizer.zero_grad(set_to_none=True)
            batch = prepare_batch_tensors(
                model,
                item,
                model.device,
                next(model.parameters()).dtype,
                min_pixels=cfg.min_pixels,
                max_pixels=cfg.max_pixels,
                enable_thinking=cfg.enable_thinking,
            )
            loss, _, _ = forward_loss(model, tokenizer, batch, module)
            loss.backward()
            head_grad = module.W_o.weight.grad
            if head_grad is None or not bool(torch.isfinite(head_grad).all()) or float(head_grad.abs().sum()) == 0:
                raise RuntimeError(f"{name} smoke step {step_index} did not give a finite nonzero output-head gradient")
            backbone_grads = [
                param_name
                for param_name, param in model.named_parameters()
                if (not param_name.startswith("reljoin")) and param.grad is not None and float(param.grad.abs().sum()) > 0
            ]
            if backbone_grads:
                raise RuntimeError(f"Frozen backbone received gradients: {backbone_grads[:5]}")
            if step_index == 1:
                first_upstream = 0.0 if upstream.grad is None else float(upstream.grad.abs().sum())
            optimizer.step()
            del batch
        if upstream.grad is None or float(upstream.grad.abs().sum()) <= 0:
            raise RuntimeError(f"{name} upstream gradient stayed zero after the second real step")
        reports.append(
            {
                "variant": name,
                "steps": len(items),
                "first_upstream_abs": first_upstream,
                "second_upstream_abs": float(upstream.grad.abs().sum()),
                "paper_result": False,
            }
        )
        print(f"Smoke {name}: two real updates, upstream grad now nonzero", flush=True)
        del module
    report = {"paper_result": False, "status": "ok", "variants": reports}
    write_json(cfg.output_dir() / "smoke" / "smoke_report.json", report)
    del model
    return report
