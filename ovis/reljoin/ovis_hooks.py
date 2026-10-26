"""Hooks for RelJoin-Residual after the frozen Ovis Visual Embedding Table.

Primary path (multi-image, not video packing):
  frames → frozen VisualTokenizer → softmax visual words → VET → X_i
  → RelJoin-Residual (ragged P_i via token_mask) → X̃ written back into VISUAL_ATOM slots
  → frozen Qwen3

Vanilla bypass leaves the VET embeddings unchanged. Sequence length and
attention mask stay identical to Vanilla.

``inject_delta`` replaces that addition with a tensor already scaled after
gate, RMS calibration, and mask. The residual head is not run again.
"""
from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

VISUAL_ATOM_ID = -300
INDICATOR_IDS = (-301, -302, -303, -304)


@dataclass
class RelJoinHookState:
    enabled: bool = False
    bypass: bool = True  # True = legacy Vanilla (no pack, P_i=[])
    module: Optional[nn.Module] = None
    question_hidden: Optional[torch.Tensor] = None  # [B, d_L]
    timestamps: Optional[torch.Tensor] = None  # [B, K]
    frame_mask: Optional[torch.Tensor] = None  # [B, K]
    last_visual_embeds: Optional[torch.Tensor] = None
    last_updated_embeds: Optional[torch.Tensor] = None
    last_residual_norm: float = 0.0
    last_labels: Optional[torch.Tensor] = None
    last_num_frames: int = 0
    last_tokens_per_frame: list[int] = field(default_factory=list)
    last_visual_token_count: int = 0
    last_input_shape: tuple[int, ...] = ()
    last_effective_timestamps: Optional[list[float]] = None
    last_effective_frame_order: Optional[list[int]] = None
    cache_prepare: bool = False
    cached_result: Optional[torch.Tensor] = None
    answer_ids: Optional[torch.Tensor] = None
    eval_mode: str = "full"
    last_token_mask: Optional[torch.Tensor] = None
    inject_delta: Optional[torch.Tensor] = None  # pre-scaled residual; None = run the head
    blocked_sources: Optional[tuple] = None  # read-source block; None = official pair mask
    suppress_tokens: Optional[torch.Tensor] = None  # [K,P] or [B,K,P]; None = no token ablation
    blocked_edges: Optional[tuple] = None  # ((receiver, source), ...); arrow source -> receiver
    edge_ablation: str = "pre_softmax"
    check_beta: bool = False
    raw_mask_tokens: Optional[torch.Tensor] = None  # zero/mean the delivered visual tokens
    raw_fill: str = "zero"


_HOOK_STATE = RelJoinHookState()
_HOOK_INSTALL_COUNT = 0


def get_hook_state() -> RelJoinHookState:
    return _HOOK_STATE


def set_reljoin_context(
    question_hidden: Optional[torch.Tensor] = None,
    timestamps: Optional[torch.Tensor] = None,
    frame_mask: Optional[torch.Tensor] = None,
    answer_ids: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    text_embeds: Optional[torch.Tensor] = None,
    text_mask: Optional[torch.Tensor] = None,
) -> None:
    """Set per-forward RelJoin context."""
    del text_embeds, text_mask
    st = _HOOK_STATE
    st.question_hidden = question_hidden
    st.timestamps = timestamps
    st.frame_mask = frame_mask
    st.answer_ids = answer_ids
    if labels is not None:
        st.last_labels = labels


def configure_reljoin(
    module: Optional[nn.Module],
    *,
    enabled: bool,
    bypass: bool,
    eval_mode: str = "full",
    inject_delta: Optional[torch.Tensor] = None,
    blocked_sources: Optional[tuple] = None,
    suppress_tokens: Optional[torch.Tensor] = None,
    blocked_edges: Optional[tuple] = None,
    edge_ablation: str = "pre_softmax",
    check_beta: bool = False,
    raw_mask_tokens: Optional[torch.Tensor] = None,
    raw_fill: str = "zero",
) -> None:
    st = _HOOK_STATE
    st.module = module
    st.enabled = enabled
    st.bypass = bypass
    st.eval_mode = eval_mode or "full"
    st.inject_delta = inject_delta
    if blocked_sources:
        st.blocked_sources = tuple(int(j) for j in blocked_sources)
    else:
        st.blocked_sources = None
    st.suppress_tokens = suppress_tokens
    if blocked_edges:
        st.blocked_edges = tuple((int(i), int(j)) for i, j in blocked_edges)
    else:
        st.blocked_edges = None
    st.edge_ablation = edge_ablation or "pre_softmax"
    st.check_beta = bool(check_beta)
    st.raw_mask_tokens = raw_mask_tokens
    st.raw_fill = raw_fill or "zero"


def hook_install_count() -> int:
    return int(_HOOK_INSTALL_COUNT)


def no_reljoin_hooks_remaining(model: nn.Module) -> bool:
    return not bool(getattr(model, "_reljoin_hook_installed", False))


def reset_hook_diagnostics() -> None:
    """Clear cached per-forward diagnostics. Does not uninstall the hook."""
    st = _HOOK_STATE
    st.last_visual_embeds = None
    st.last_updated_embeds = None
    st.last_residual_norm = 0.0
    st.last_tokens_per_frame = []
    st.last_num_frames = 0
    st.last_visual_token_count = 0
    st.last_input_shape = ()
    st.last_effective_timestamps = None
    st.last_effective_frame_order = None
    st.last_token_mask = None
    mod = st.module
    if mod is not None:
        if hasattr(mod, "last_stats"):
            mod.last_stats = {}
        if hasattr(mod, "last_delta_x"):
            mod.last_delta_x = None
        if hasattr(mod, "last_beta"):
            mod.last_beta = None
        if hasattr(mod, "last_alpha"):
            mod.last_alpha = None
        if hasattr(mod, "last_has_neighbor"):
            mod.last_has_neighbor = None
        if hasattr(mod, "last_phi_dt"):
            mod.last_phi_dt = None
        if hasattr(mod, "last_r"):
            mod.last_r = None


def begin_prepare_cache() -> None:
    """Reuse one frozen VT/VET merge for subsequent RelJoin forwards."""
    st = _HOOK_STATE
    st.cache_prepare = True
    st.cached_result = None


def end_prepare_cache() -> None:
    st = _HOOK_STATE
    st.cache_prepare = False
    st.cached_result = None


def freeze_backbone(model: nn.Module) -> None:
    """Freeze visual tokenizer, VET, embeddings, LLM, lm_head."""
    for p in model.parameters():
        p.requires_grad = False


def enable_reljoin_grads(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def require_flash_attn() -> None:
    """Ovis2.5 ViT imports flash_attn at module load. Fail loudly if missing."""
    try:
        import flash_attn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "flash-attn is required by Ovis2.5 (hard import in modeling_ovis2_5.py). "
            "Install flash-attn==2.7.0.post2 --no-build-isolation before loading the model."
        ) from exc


def visual_atom_id(model: Optional[nn.Module] = None) -> int:
    if model is not None:
        try:
            mod = __import__(type(model).__module__, fromlist=["VISUAL_ATOM_ID"])
            return int(getattr(mod, "VISUAL_ATOM_ID", VISUAL_ATOM_ID))
        except Exception:
            return VISUAL_ATOM_ID
    return VISUAL_ATOM_ID


def visual_atom_spans(
    input_ids: torch.Tensor,
    atom_id: int = VISUAL_ATOM_ID,
) -> list[tuple[int, int]]:
    """Half-open (start, end) runs of consecutive visual atoms in row 0."""
    if input_ids.dim() != 2:
        raise ValueError(f"expected input_ids [B, T], got {tuple(input_ids.shape)}")
    ids = input_ids[0].detach().tolist()
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(ids)
    while i < n:
        if int(ids[i]) == int(atom_id):
            j = i + 1
            while j < n and int(ids[j]) == int(atom_id):
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def pack_variable_frame_tokens(
    groups: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad per-frame tokens with different P_i.

    Args:
        groups: list of [B, P_i, d]
    Returns:
        frame_tokens: [B, K, P_max, d]
        token_mask: [B, K, P_max]
    """
    if not groups:
        raise ValueError("pack_variable_frame_tokens received no frame groups")
    B, _, d = groups[0].shape
    device = groups[0].device
    dtype = groups[0].dtype
    lengths = [int(g.size(1)) for g in groups]
    p_max = max(lengths)
    k = len(groups)
    frame_tokens = groups[0].new_zeros(B, k, p_max, d)
    token_mask = torch.zeros(B, k, p_max, dtype=torch.bool, device=device)
    for i, (g, p) in enumerate(zip(groups, lengths)):
        if g.shape[0] != B or g.shape[-1] != d:
            raise ValueError(f"group {i} shape {tuple(g.shape)} != [B={B}, P, d={d}]")
        if p > 0:
            frame_tokens[:, i, :p] = g
            token_mask[:, i, :p] = True
    return frame_tokens.to(dtype=dtype), token_mask


def scatter_variable_frame_tokens(
    embeds: torch.Tensor,
    spans: list[tuple[int, int]],
    x_tilde: torch.Tensor,
) -> torch.Tensor:
    """Write padded [B, K, P_max, d] back into visual-atom slots of embeds."""
    out = embeds
    for i, (start, end) in enumerate(spans):
        p = end - start
        out[:, start:end, :] = x_tilde[:, i, :p, :]
    return out


def masked_inject(
    frame_tokens: torch.Tensor,
    token_mask: torch.Tensor,
    inject_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add a residual that was already gated, RMS-calibrated, and masked.

    Does not call the residual head and does not recompute RMS calibration.
    Padding positions (``token_mask`` false) stay at the original token.
    """
    delta = inject_delta.to(device=frame_tokens.device, dtype=torch.float32)
    if tuple(delta.shape) != tuple(frame_tokens.shape):
        raise AssertionError(
            f"inject_delta shape {tuple(delta.shape)} != packed X {tuple(frame_tokens.shape)}"
        )
    delta = delta * token_mask.to(device=delta.device, dtype=torch.float32).unsqueeze(-1)
    x_tilde = frame_tokens + delta.to(dtype=frame_tokens.dtype)
    return x_tilde, delta


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, n_kv, slen, hd = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, n_kv, n_rep, slen, hd)
    return hidden_states.reshape(bsz, n_kv * n_rep, slen, hd)


def _qwen_gqa_dims(attn: nn.Module, query_states: torch.Tensor, key_states: torch.Tensor):
    """Qwen3-8B is GQA: q is 32×128, k/v are 8×128. Never assume n_kv == n_heads."""
    cfg = getattr(attn, "config", None)
    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None and cfg is not None:
        head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        n_heads_guess = getattr(attn, "num_heads", None) or getattr(cfg, "num_attention_heads", 32)
        head_dim = query_states.size(-1) // int(n_heads_guess)
    head_dim = int(head_dim)
    n_heads = int(query_states.size(-1) // head_dim)
    n_kv = int(key_states.size(-1) // head_dim)
    if n_heads < 1 or n_kv < 1 or query_states.size(-1) % head_dim != 0:
        raise RuntimeError(
            f"Bad Qwen GQA dims: q={tuple(query_states.shape)} k={tuple(key_states.shape)} head_dim={head_dim}"
        )
    return n_heads, n_kv, head_dim


def _qwen_sdpa_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    """Qwen2-style self-attn via SDPA. Qwen3 is left on native attention (see patch)."""
    if output_attentions:
        raise RuntimeError("SDPA RelJoin patch does not support output_attentions=True")

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    n_heads, n_kv, head_dim = _qwen_gqa_dims(self, query_states, key_states)

    query_states = query_states.view(bsz, q_len, n_heads, head_dim)
    key_states = key_states.view(bsz, q_len, n_kv, head_dim)
    value_states = value_states.view(bsz, q_len, n_kv, head_dim)
    q_norm = getattr(self, "q_norm", None)
    k_norm = getattr(self, "k_norm", None)
    if q_norm is not None:
        query_states = q_norm(query_states)
    if k_norm is not None:
        key_states = k_norm(key_states)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    apply_rotary_pos_emb = None
    try:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    except Exception:
        try:
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        except Exception:
            apply_rotary_pos_emb = None
    pos_emb = kwargs.get("position_embeddings")
    rotary = getattr(self, "rotary_emb", None)
    if apply_rotary_pos_emb is not None and pos_emb is not None:
        cos, sin = pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    elif apply_rotary_pos_emb is not None and rotary is not None:
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            past_k = past_key_value[0] if isinstance(past_key_value, tuple) else None
            if past_k is not None:
                kv_seq_len += past_k.shape[-2]
        try:
            cos, sin = rotary(value_states, seq_len=kv_seq_len)
        except TypeError:
            cos, sin = rotary(value_states, position_ids)
        try:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
        except TypeError:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None and isinstance(past_key_value, tuple):
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    past_out = (key_states, value_states) if use_cache else None

    n_rep = n_heads // n_kv if n_kv else 1
    if n_rep != 1:
        key_states = _repeat_kv(key_states, n_rep)
        value_states = _repeat_kv(value_states, n_rep)

    orig_dtype = query_states.dtype
    query_states = query_states.float()
    key_states = key_states.float()
    value_states = value_states.float()

    attn_mask = None
    is_causal = False
    kv_len = key_states.shape[-2]
    if attention_mask is None:
        is_causal = q_len > 1 and past_key_value is None
    elif attention_mask.dim() == 2:
        if bool(attention_mask.all().item()) and q_len == kv_len and q_len > 1:
            is_causal = True
        else:
            attn_mask = torch.zeros(
                attention_mask.size(0),
                1,
                q_len,
                attention_mask.size(-1),
                device=query_states.device,
                dtype=query_states.dtype,
            )
            attn_mask = attn_mask.masked_fill(
                attention_mask[:, None, None, :] == 0,
                torch.finfo(query_states.dtype).min,
            )
    else:
        if q_len == kv_len and q_len > 1:
            is_causal = True
        else:
            attn_mask = attention_mask.to(dtype=query_states.dtype)

    hidden = getattr(self, "hidden_size", n_heads * head_dim)
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
    )
    attn_output = attn_output.to(orig_dtype)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, hidden)
    attn_output = self.o_proj(attn_output)
    # Qwen3 decoder unpacks 2 values; Qwen2 may expect a cache triple.
    if hasattr(self, "q_norm") or hasattr(self, "k_norm"):
        return attn_output, None
    return attn_output, None, past_out


def _enable_native_sdpa(module: Optional[nn.Module]) -> bool:
    if module is None:
        return False
    fn = getattr(module, "set_attn_implementation", None)
    if callable(fn):
        try:
            fn("sdpa")
            return True
        except Exception:
            pass
    cfg = getattr(module, "config", None)
    if cfg is not None and hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = "sdpa"
        return True
    return False


def patch_qwen_sdpa_attention(model: nn.Module) -> int:
    """Use native Qwen3 SDPA. Do not replace Qwen3Attention.forward (GQA + q/k-norm)."""
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    llm = getattr(model, "llm", model)
    inner = getattr(llm, "model", llm)
    native = _enable_native_sdpa(llm) or _enable_native_sdpa(inner)
    layers = getattr(inner, "layers", None)
    if layers is None:
        print("WARN: could not find Qwen layers for SDPA")
        return 0

    n_native = 0
    n_patched = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if hasattr(attn, "q_norm") or hasattr(attn, "k_norm"):
            if hasattr(attn, "config") and hasattr(attn.config, "_attn_implementation"):
                attn.config._attn_implementation = "sdpa"
            n_native += 1
            continue
        attn.forward = types.MethodType(_qwen_sdpa_forward, attn)
        n_patched += 1
    print(
        f"Qwen attention: native SDPA layers={n_native} "
        f"custom_patch={n_patched} llm_native={native}"
    )
    return n_native + n_patched


def enable_frozen_backbone_checkpointing(model: nn.Module) -> bool:
    """Enable HF gradient checkpointing on the frozen Qwen3 LLM only."""
    llm = getattr(model, "llm", model)
    if hasattr(llm, "config"):
        llm.config.use_cache = False
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass

    def _enable(m: Optional[nn.Module]) -> bool:
        if m is None:
            return False
        fn = getattr(m, "gradient_checkpointing_enable", None)
        if callable(fn):
            try:
                fn(gradient_checkpointing_kwargs={"use_reentrant": False})
                return True
            except TypeError:
                fn()
                return True
        return False

    ok = _enable(llm) or _enable(getattr(llm, "model", None))
    print(f"Gradient checkpointing enabled={ok}")
    return ok


def configure_frozen_backbone_for_training(model: nn.Module) -> None:
    """Keep checkpointing active (model.train) but freeze dropout/vision eval behavior."""
    model.train()
    llm = getattr(model, "llm", None)
    if llm is not None and hasattr(llm, "config"):
        llm.config.use_cache = False
    for name, mod in model.named_modules():
        if name.startswith("reljoin"):
            continue
        if isinstance(mod, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            mod.eval()
    vt = getattr(model, "visual_tokenizer", None)
    if vt is not None:
        vt.eval()
    vit = getattr(vt, "vit", None) if vt is not None else None
    if vit is not None:
        vit.eval()


def assert_only_reljoin_trainable(model: nn.Module, reljoin: nn.Module) -> list[str]:
    """Assert no parameter outside RelJoin receives gradients / is trainable."""
    reljoin_ids = {id(p) for p in reljoin.parameters()}
    bad = []
    for name, p in model.named_parameters():
        if p.requires_grad and id(p) not in reljoin_ids:
            if not name.startswith("reljoin"):
                bad.append(name)
    if bad:
        raise AssertionError(f"Non-RelJoin trainable params: {bad[:20]}")
    return [n for n, p in reljoin.named_parameters() if p.requires_grad]


def report_parameter_budget(model: nn.Module, reljoin: nn.Module) -> dict:
    all_params = {}
    for n, p in model.named_parameters():
        all_params[id(p)] = (n, p)
    for n, p in reljoin.named_parameters():
        all_params[id(p)] = (f"reljoin.{n}", p)
    total = sum(p.numel() for _, p in all_params.values())
    trainable_names = [n for n, p in all_params.values() if p.requires_grad]
    frozen_names = [n for n, p in all_params.values() if not p.requires_grad]
    trainable = sum(p.numel() for _, p in all_params.values() if p.requires_grad)
    return {
        "trainable_names": trainable_names,
        "frozen_count": len(frozen_names),
        "trainable_count": trainable,
        "total_count": total,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }


def _question_text(question: str, choices: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = [question.strip()]
    for i, c in enumerate(choices):
        if not str(c).strip():
            continue
        lines.append(f"{letters[i]}. {str(c).strip()}")
    return "\n".join(lines)


def _last_nonpad_hidden(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
    bidx = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[bidx, lengths]


def _llm_embed_tokens(model: nn.Module):
    llm = getattr(model, "llm", None)
    if llm is not None:
        getter = getattr(llm, "get_input_embeddings", None)
        if callable(getter):
            return getter()
    getter = getattr(model, "get_wte", None)
    if callable(getter):
        return getter()
    raise RuntimeError("Could not find Ovis/Qwen3 input embeddings")


def _llm_backbone(model: nn.Module) -> nn.Module:
    llm = getattr(model, "llm", model)
    inner = getattr(llm, "model", None)
    return inner if inner is not None else llm


def encode_question_hidden(
    model: nn.Module,
    tokenizer,
    question: str,
    choices: list[str],
    device: torch.device,
    dtype: torch.dtype,
    **kwargs,
) -> torch.Tensor:
    """Frozen text-only question vector. Last non-pad Qwen3 hidden (stopgrad)."""
    del kwargs
    text = _question_text(question, choices)
    toks = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = toks["input_ids"].to(device)
    mask = toks.get("attention_mask")
    if mask is None:
        mask = torch.ones_like(input_ids)
    else:
        mask = mask.to(device)
    with torch.no_grad():
        embeds = _llm_embed_tokens(model)(input_ids)
        llama = _llm_backbone(model)
        try:
            out = llama(
                inputs_embeds=embeds,
                attention_mask=mask,
                use_cache=False,
                return_dict=True,
            )
        except TypeError:
            out = llama(
                inputs_embeds=embeds,
                attention_mask=mask,
                use_cache=False,
            )
        seq = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        hidden = _last_nonpad_hidden(seq, mask)
    return hidden.to(dtype=dtype)


def _apply_raw_visual_mask(x_tilde, token_mask, raw_mask, fill: str):
    """Replace selected delivered visual tokens with zeros or the kept-token mean.

    This is the secondary out-of-distribution mask. It is not the adapter-internal
    ablation, which never writes these values into the frozen LLM.
    """
    raw = raw_mask.to(device=x_tilde.device, dtype=torch.bool)
    if raw.shape == token_mask.shape[1:]:
        raw = raw.unsqueeze(0).expand_as(token_mask)
    if raw.shape != token_mask.shape:
        raise ValueError(
            f"raw_mask_tokens shape {tuple(raw.shape)} != token_mask {tuple(token_mask.shape)}"
        )
    raw = raw & token_mask
    out = x_tilde.clone()
    mode = (fill or "zero").strip().lower()
    if mode not in ("zero", "mean"):
        raise ValueError(f"raw_fill must be zero or mean, got {fill!r}")
    if mode == "zero":
        return out.masked_fill(raw.unsqueeze(-1), 0)
    for b in range(out.shape[0]):
        for i in range(out.shape[1]):
            drop = raw[b, i]
            keep = token_mask[b, i] & ~raw[b, i]
            if not bool(drop.any()):
                continue
            if bool(keep.any()):
                mu = out[b, i, keep].float().mean(dim=0)
                out[b, i, drop] = mu.to(dtype=out.dtype)
            else:
                out[b, i, drop] = 0
    return out


def _apply_reljoin(
    model: nn.Module,
    input_ids: torch.Tensor,
    embeds: torch.Tensor,
) -> torch.Tensor:
    from .reljoin_modules import EVAL_FULL, EVAL_IDENTITY

    st = _HOOK_STATE
    st.last_visual_embeds = None
    st.last_updated_embeds = None
    st.last_residual_norm = 0.0
    st.last_tokens_per_frame = []
    st.last_num_frames = 0
    st.last_visual_token_count = 0
    st.last_input_shape = tuple(int(x) for x in embeds.shape)
    st.last_effective_timestamps = None
    st.last_effective_frame_order = None
    st.last_token_mask = None

    if not st.enabled or st.bypass:
        return embeds
    if st.module is None and st.inject_delta is None:
        return embeds

    atom_id = visual_atom_id(model)
    spans = visual_atom_spans(input_ids, atom_id=atom_id)
    if not spans:
        return embeds

    groups = [embeds[:, start:end, :].clone() for start, end in spans]
    K = len(groups)
    eval_mode = (st.eval_mode or EVAL_FULL).strip().lower()
    original_order = list(range(K))

    frame_tokens, token_mask = pack_variable_frame_tokens(groups)
    B, K_pack, _P, _d = frame_tokens.shape
    st.last_num_frames = K_pack
    st.last_tokens_per_frame = [int(g.size(1)) for g in groups]
    st.last_visual_token_count = int(sum(st.last_tokens_per_frame))
    st.last_visual_embeds = frame_tokens.detach()
    st.last_token_mask = token_mask.detach()
    st.last_effective_frame_order = original_order

    if st.frame_mask is not None:
        frame_mask = st.frame_mask.to(device=embeds.device)
        if frame_mask.dim() == 1:
            frame_mask = frame_mask.unsqueeze(0)
        if frame_mask.shape[0] != B:
            frame_mask = frame_mask[:B]
        if frame_mask.shape[1] < K:
            pad = torch.zeros(
                B, K - frame_mask.shape[1], device=embeds.device, dtype=torch.bool
            )
            frame_mask = torch.cat([frame_mask, pad], dim=1)
        frame_mask = frame_mask[:, :K]
    else:
        frame_mask = token_mask.any(dim=-1)

    if st.timestamps is not None:
        timestamps = st.timestamps.to(device=embeds.device, dtype=embeds.dtype)
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(0)
        if timestamps.shape[0] != B:
            timestamps = timestamps[:B]
        if timestamps.shape[1] < K:
            pad_t = torch.zeros(
                B, K - timestamps.shape[1], device=embeds.device, dtype=embeds.dtype
            )
            timestamps = torch.cat([timestamps, pad_t], dim=1)
        timestamps = timestamps[:, :K]
    else:
        timestamps = (
            torch.arange(K, device=embeds.device, dtype=embeds.dtype).unsqueeze(0).expand(B, K)
        )

    st.last_effective_timestamps = [float(x) for x in timestamps[0].detach().cpu().tolist()]

    if st.inject_delta is not None:
        # λ, R, and D^parallel are applied to the captured ΔX. Do not re-enter the head.
        x_tilde, delta_applied = masked_inject(frame_tokens, token_mask, st.inject_delta)
        if st.module is not None and hasattr(st.module, "last_delta_x"):
            st.module.last_delta_x = delta_applied.detach()
    else:
        if st.question_hidden is None:
            raise RuntimeError(
                "RelJoin-Residual enabled but question_hidden not set. "
                "Call set_reljoin_context(question_hidden=...) before forward/generate."
            )
        question_hidden = st.question_hidden.to(device=embeds.device)
        if question_hidden.dim() == 1:
            question_hidden = question_hidden.unsqueeze(0)
        if question_hidden.shape[0] != B:
            question_hidden = question_hidden[:B]

        forward_kwargs = {}
        if st.blocked_sources:
            forward_kwargs["blocked_sources"] = st.blocked_sources
        if st.suppress_tokens is not None:
            forward_kwargs["suppress_tokens"] = st.suppress_tokens
        if st.blocked_edges:
            forward_kwargs["blocked_edges"] = st.blocked_edges
            forward_kwargs["edge_ablation"] = st.edge_ablation
        if st.check_beta:
            forward_kwargs["check_beta"] = True
        x_tilde = st.module(
            frame_tokens=frame_tokens.float(),
            token_mask=token_mask,
            frame_mask=frame_mask,
            timestamps=timestamps.float(),
            question_hidden=question_hidden.float(),
            eval_mode=eval_mode,
            **forward_kwargs,
        )
        if eval_mode == EVAL_IDENTITY:
            x_tilde = frame_tokens
    if x_tilde.shape != frame_tokens.shape:
        raise AssertionError(
            f"RelJoin-Residual changed token shape {tuple(frame_tokens.shape)} "
            f"→ {tuple(x_tilde.shape)}"
        )
    if st.raw_mask_tokens is not None:
        x_tilde = _apply_raw_visual_mask(x_tilde, token_mask, st.raw_mask_tokens, st.raw_fill)
    x_tilde = x_tilde.to(dtype=embeds.dtype)
    new_embeds = embeds.clone()
    new_embeds = scatter_variable_frame_tokens(new_embeds, spans, x_tilde)
    if new_embeds.shape != embeds.shape:
        raise AssertionError(
            f"RelJoin-Residual changed sequence shape {tuple(embeds.shape)} "
            f"→ {tuple(new_embeds.shape)}"
        )
    st.last_updated_embeds = new_embeds.detach()
    delta = (x_tilde - frame_tokens.float()).float()
    st.last_residual_norm = float(delta.norm().detach().cpu())
    from .frame_viz.collector import note_delivered_tokens

    note_delivered_tokens(
        frame_tokens=frame_tokens,
        delivered=x_tilde,
        token_mask=token_mask,
        frame_mask=frame_mask,
        module=st.module,
        timestamps=st.last_effective_timestamps,
    )
    return new_embeds


def install_reljoin_hook(model: nn.Module) -> nn.Module:
    """Wrap merge_multimodal to apply RelJoin-Residual on visual-atom embeddings."""
    global _HOOK_INSTALL_COUNT
    if getattr(model, "_reljoin_hook_installed", False):
        return model

    original = model.merge_multimodal

    def hooked_merge(self, input_ids, pixel_values=None, grid_thws=None, **kwargs):
        st = _HOOK_STATE
        if st.cache_prepare and st.cached_result is not None:
            embeds = st.cached_result
        else:
            with torch.no_grad():
                embeds = original(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    **kwargs,
                )
            if st.cache_prepare:
                st.cached_result = embeds.detach().clone()
        return _apply_reljoin(self, input_ids, embeds)

    model.merge_multimodal = types.MethodType(hooked_merge, model)
    model._reljoin_hook_installed = True
    model._reljoin_original_merge = original
    _HOOK_INSTALL_COUNT += 1
    return model


def restore_original_merge(model: nn.Module) -> None:
    if getattr(model, "_reljoin_hook_installed", False) and hasattr(
        model, "_reljoin_original_merge"
    ):
        model.merge_multimodal = model._reljoin_original_merge
        model._reljoin_hook_installed = False


def remove_all_reljoin_hooks(model: nn.Module) -> None:
    restore_original_merge(model)


def ovis_hidden_size(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "hidden_size", None):
        return int(cfg.hidden_size)
    llm = getattr(model, "llm", None)
    if llm is not None and getattr(llm, "config", None) is not None:
        return int(llm.config.hidden_size)
    raise RuntimeError("Could not read Ovis/Qwen3 hidden_size")
