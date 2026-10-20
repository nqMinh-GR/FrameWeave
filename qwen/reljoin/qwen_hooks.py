"""Hooks for FrameWeave Hybrid v6 after the frozen Qwen2.5-VL visual merger.

Primary path (K still images, not native video packing):
  frames → frozen ViT + merger → per-image tokens X_i
  → FrameWeave V5.1 (ragged P_i via token_mask) → X̃ written back into image-pad slots
  → frozen Qwen2.5 LLM

Vanilla bypass leaves merger embeddings unchanged. Sequence length and
attention mask stay identical to Vanilla. eval_mode is identity or full.
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
) -> None:
    st = _HOOK_STATE
    st.module = module
    st.enabled = enabled
    st.bypass = bypass
    st.eval_mode = eval_mode or "full"


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
    """Freeze vision encoder, merger, embeddings, LLM, lm_head."""
    for p in model.parameters():
        p.requires_grad = False


def enable_reljoin_grads(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def require_flash_attn() -> None:
    """Optional on Qwen2.5-VL (SDPA is enough)."""
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return


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


def _keep_native_qwen_attention(attn: nn.Module) -> bool:
    """True when the layer already has a correct SDPA / 3D-MRoPE forward.

    Qwen2.5-VL ships `Qwen2_5_VLSdpaAttention` with 3D MRoPE. The Ovis/Qwen2
    custom forward applies 1D RoPE and turns K/V into 5D, which then crashes
    in `_repeat_kv` (`ValueError: too many values to unpack (expected 4)`).
    """
    name = type(attn).__name__
    if "Qwen2_5_VL" in name or "Sdpa" in name:
        return True
    if hasattr(attn, "q_norm") or hasattr(attn, "k_norm"):
        return True
    for obj in (attn, getattr(attn, "config", None)):
        if obj is None:
            continue
        rope = getattr(obj, "rope_scaling", None)
        if isinstance(rope, dict) and "mrope_section" in rope:
            return True
    return False


def patch_qwen_sdpa_attention(model: nn.Module) -> int:
    """Prefer native SDPA on Qwen2.5-VL language layers."""
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    llm = getattr(model, "llm", None) or getattr(model, "model", model)
    inner = getattr(llm, "language_model", None) or getattr(llm, "model", llm)
    native = _enable_native_sdpa(model) or _enable_native_sdpa(llm) or _enable_native_sdpa(inner)
    layers = getattr(inner, "layers", None)
    if layers is None:
        layers = getattr(llm, "layers", None)
    if layers is None:
        print("WARN: could not find Qwen layers for SDPA")
        return 0

    n_native = 0
    n_patched = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if _keep_native_qwen_attention(attn):
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
    """Enable HF gradient checkpointing on the frozen language model."""
    llm = getattr(model, "llm", None) or getattr(model, "model", model)
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
    llm = getattr(model, "llm", None) or getattr(model, "model", None)
    if llm is not None and hasattr(llm, "config"):
        llm.config.use_cache = False
    for name, mod in model.named_modules():
        if name.startswith("reljoin"):
            continue
        if isinstance(mod, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            mod.eval()
    visual = getattr(model, "visual", None)
    if visual is not None:
        visual.eval()
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
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        return getter()
    inner = getattr(model, "model", None)
    if inner is not None:
        getter = getattr(inner, "get_input_embeddings", None)
        if callable(getter):
            return getter()
        if hasattr(inner, "embed_tokens"):
            return inner.embed_tokens
        lang = getattr(inner, "language_model", None)
        if lang is not None and hasattr(lang, "embed_tokens"):
            return lang.embed_tokens
    raise RuntimeError("Could not find Qwen2.5-VL input embeddings")


def _llm_backbone(model: nn.Module) -> nn.Module:
    inner = getattr(model, "model", None)
    if inner is None:
        return model
    lang = getattr(inner, "language_model", None)
    if lang is not None:
        return lang
    return inner


def encode_question_hidden(
    model: nn.Module,
    tokenizer,
    question: str,
    choices: list[str],
    device: torch.device,
    dtype: torch.dtype,
    *,
    use_raw_mean: bool = False,
) -> torch.Tensor:
    """Frozen text-only question vector. Last non-pad Qwen2.5 hidden, or raw mean."""
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
        if use_raw_mean:
            m = mask.to(dtype=embeds.dtype).unsqueeze(-1)
            hidden = (embeds * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)
            return hidden.to(dtype=dtype)
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


def encode_question_embeds(
    model: nn.Module,
    tokenizer,
    question: str,
    choices: list[str],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Back-compat: raw token embeddings + mask (used by RawMean control)."""
    text = _question_text(question, choices)
    toks = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = toks["input_ids"].to(device)
    with torch.no_grad():
        embeds = _llm_embed_tokens(model)(input_ids).to(dtype=dtype)
    mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
    return embeds, mask


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

    if not st.enabled or st.module is None:
        return embeds
    # Legacy Vanilla: skip packing entirely (P_i=[]). Used only for the
    # 20-sample identity-vs-bypass reproduction check.
    if st.bypass:
        return embeds

    atom_id = visual_atom_id(model)
    spans = visual_atom_spans(input_ids, atom_id=atom_id)
    if not spans:
        return embeds

    # Clone so the VT/VET cache is never aliased by RelJoin or scatter.
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

    x_tilde = st.module(
        frame_tokens=frame_tokens.float(),
        token_mask=token_mask,
        frame_mask=frame_mask,
        timestamps=timestamps.float(),
        question_hidden=question_hidden.float(),
        eval_mode=eval_mode,
    )
    if eval_mode == EVAL_IDENTITY:
        x_tilde = frame_tokens
    if x_tilde.shape != frame_tokens.shape:
        raise AssertionError(
            f"RelJoin-Residual changed token shape {tuple(frame_tokens.shape)} "
            f"→ {tuple(x_tilde.shape)}"
        )
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
    return new_embeds


def _spatial_merge_size(model: nn.Module) -> int:
    if getattr(model, "spatial_merge_size", None):
        return int(model.spatial_merge_size)
    visual = getattr(model, "visual", None)
    if visual is None:
        inner = getattr(model, "model", None)
        visual = getattr(inner, "visual", None) if inner is not None else None
    if visual is not None and getattr(visual, "spatial_merge_size", None):
        return int(visual.spatial_merge_size)
    cfg = getattr(model, "config", None)
    vision = getattr(cfg, "vision_config", None) if cfg is not None else None
    if vision is not None and getattr(vision, "spatial_merge_size", None):
        return int(vision.spatial_merge_size)
    return 2


def _callable_attr(obj, name: str):
    return obj is not None and callable(getattr(obj, name, None))


def _feature_host(model: nn.Module):
    """Module that owns get_image_features (newer transformers), if any."""
    if _callable_attr(model, "get_image_features"):
        return model
    inner = getattr(model, "model", None)
    if _callable_attr(inner, "get_image_features"):
        return inner
    return None


def _visual_module(model: nn.Module):
    """Vision tower + merger. transformers 4.51 calls this from ForConditionalGeneration.forward."""
    for cand in (model, getattr(model, "model", None)):
        if cand is None:
            continue
        visual = getattr(cand, "visual", None)
        if visual is not None and callable(getattr(visual, "forward", None)):
            return visual
    return None


def _clone_image_features(feats):
    if isinstance(feats, (tuple, list)):
        return type(feats)(t.detach().clone() for t in feats)
    return feats.detach().clone()


def _split_image_feature_groups(feats, image_grid_thw, merge_size: int) -> list[torch.Tensor]:
    if isinstance(feats, (tuple, list)):
        groups = []
        for g in feats:
            groups.append(g.reshape(-1, g.shape[-1]) if g.dim() == 3 else g)
        return groups
    if image_grid_thw is None:
        raise RuntimeError("Qwen RelJoin needs image_grid_thw to split visual tokens.")
    sizes = (image_grid_thw.prod(-1) // (int(merge_size) ** 2)).to(torch.long).tolist()
    sizes = [max(int(s), 0) for s in sizes]
    if sum(sizes) != int(feats.shape[0]):
        raise RuntimeError(
            f"Visual token split mismatch: sum(P_i)={sum(sizes)} != N={int(feats.shape[0])}"
        )
    return list(torch.split(feats, sizes, dim=0))


def _apply_reljoin_groups(model: nn.Module, groups: list[torch.Tensor]) -> list[torch.Tensor]:
    """Run RelJoin on a list of [P_i, d] visual token groups."""
    from .reljoin_modules import EVAL_FULL, EVAL_IDENTITY

    st = _HOOK_STATE
    if not st.enabled or st.module is None or st.bypass or not groups:
        return groups
    packed = [g.unsqueeze(0).contiguous() if g.dim() == 2 else g.contiguous() for g in groups]
    frame_tokens, token_mask = pack_variable_frame_tokens(packed)
    B, K, _P, _d = frame_tokens.shape
    st.last_num_frames = K
    st.last_tokens_per_frame = [int(g.size(1)) for g in packed]
    st.last_visual_token_count = int(sum(st.last_tokens_per_frame))
    st.last_visual_embeds = frame_tokens.detach()
    st.last_token_mask = token_mask.detach()
    st.last_effective_frame_order = list(range(K))

    if st.frame_mask is not None:
        frame_mask = st.frame_mask.to(device=frame_tokens.device)
        if frame_mask.dim() == 1:
            frame_mask = frame_mask.unsqueeze(0)
        if frame_mask.shape[1] < K:
            pad = torch.zeros(
                frame_mask.shape[0], K - frame_mask.shape[1],
                device=frame_mask.device, dtype=torch.bool,
            )
            frame_mask = torch.cat([frame_mask, pad], dim=1)
        frame_mask = frame_mask[:, :K]
    else:
        frame_mask = token_mask.any(dim=-1)

    if st.timestamps is not None:
        timestamps = st.timestamps.to(device=frame_tokens.device, dtype=frame_tokens.dtype)
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(0)
        if timestamps.shape[1] < K:
            pad_t = torch.zeros(
                timestamps.shape[0], K - timestamps.shape[1],
                device=timestamps.device, dtype=timestamps.dtype,
            )
            timestamps = torch.cat([timestamps, pad_t], dim=1)
        timestamps = timestamps[:, :K]
    else:
        timestamps = torch.arange(K, device=frame_tokens.device, dtype=frame_tokens.dtype)
        timestamps = timestamps.unsqueeze(0).expand(B, K)

    st.last_effective_timestamps = [float(x) for x in timestamps[0].detach().cpu().tolist()]
    if st.question_hidden is None:
        raise RuntimeError(
            "RelJoin-Residual enabled but question_hidden not set. "
            "Call set_reljoin_context(question_hidden=...) before forward/generate."
        )
    question_hidden = st.question_hidden.to(device=frame_tokens.device)
    if question_hidden.dim() == 1:
        question_hidden = question_hidden.unsqueeze(0)

    eval_mode = (st.eval_mode or EVAL_FULL).strip().lower()
    x_tilde = st.module(
        frame_tokens=frame_tokens.float(),
        token_mask=token_mask,
        frame_mask=frame_mask,
        timestamps=timestamps.float(),
        question_hidden=question_hidden.float(),
        eval_mode=eval_mode,
    )
    if eval_mode == EVAL_IDENTITY:
        x_tilde = frame_tokens
    if x_tilde.shape != frame_tokens.shape:
        raise AssertionError(
            f"RelJoin-Residual changed token shape {tuple(frame_tokens.shape)} "
            f"→ {tuple(x_tilde.shape)}"
        )
    x_tilde = x_tilde.to(dtype=frame_tokens.dtype)
    st.last_updated_embeds = x_tilde.detach()
    delta = (x_tilde - frame_tokens.float()).float()
    st.last_residual_norm = float(delta.norm().detach().cpu())
    return [x_tilde[0, i, : packed[i].size(1), :] for i in range(K)]


def _apply_reljoin_image_features(model: nn.Module, feats, image_grid_thw):
    st = _HOOK_STATE
    if not st.enabled or st.module is None or st.bypass:
        return feats
    merge_size = _spatial_merge_size(model)
    groups = _split_image_feature_groups(feats, image_grid_thw, merge_size)
    updated = _apply_reljoin_groups(model, groups)
    if isinstance(feats, (tuple, list)):
        return type(feats)(updated)
    return torch.cat(updated, dim=0)


def _run_frozen_visual(original, *args, **kwargs):
    st = _HOOK_STATE
    if st.cache_prepare and st.cached_result is not None:
        return st.cached_result
    with torch.no_grad():
        feats = original(*args, **kwargs)
    if st.cache_prepare:
        st.cached_result = _clone_image_features(feats)
    return feats


def install_reljoin_hook(model: nn.Module) -> nn.Module:
    """Hook after frozen visual merger.

    transformers 4.51 Qwen2.5-VL calls ``self.visual(pixel_values, grid_thw=...)``
    from ForConditionalGeneration.forward — there is no ``get_image_features``
    and no Ovis ``merge_multimodal``. Newer transformers may expose
    ``get_image_features`` on the inner VL model. Dummy unit-test objects keep
    the ``merge_multimodal`` path.
    """
    global _HOOK_INSTALL_COUNT
    if getattr(model, "_reljoin_hook_installed", False):
        return model

    host = _feature_host(model)
    visual = _visual_module(model)

    if host is not None:
        original = host.get_image_features

        def hooked_features(self, pixel_values, image_grid_thw=None, **kwargs):
            feats = _run_frozen_visual(
                original, pixel_values, image_grid_thw=image_grid_thw, **kwargs
            )
            return _apply_reljoin_image_features(self, feats, image_grid_thw)

        host.get_image_features = types.MethodType(hooked_features, host)
        model._reljoin_original_features = original
        model._reljoin_hook_kind = "features"
        model._reljoin_hook_target = host
        print("RelJoin hook: get_image_features", type(host).__name__, flush=True)
    elif visual is not None:
        original = visual.forward

        def hooked_visual(self, hidden_states, grid_thw=None, **kwargs):
            feats = _run_frozen_visual(original, hidden_states, grid_thw=grid_thw, **kwargs)
            return _apply_reljoin_image_features(self, feats, grid_thw)

        visual.forward = types.MethodType(hooked_visual, visual)
        model._reljoin_original_visual = original
        model._reljoin_hook_kind = "visual"
        model._reljoin_hook_target = visual
        print("RelJoin hook: visual.forward", type(visual).__name__, flush=True)
    elif _callable_attr(model, "merge_multimodal"):
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
        model._reljoin_original_merge = original
        model._reljoin_hook_kind = "merge"
        model._reljoin_hook_target = model
        print("RelJoin hook: merge_multimodal (dummy)", flush=True)
    else:
        raise RuntimeError(
            "Qwen RelJoin needs visual.forward or get_image_features. "
            f"{type(model).__name__} has neither; merge_multimodal is Ovis-only."
        )

    model._reljoin_hook_installed = True
    _HOOK_INSTALL_COUNT += 1
    return model


def restore_original_merge(model: nn.Module) -> None:
    if not getattr(model, "_reljoin_hook_installed", False):
        return
    kind = getattr(model, "_reljoin_hook_kind", None)
    target = getattr(model, "_reljoin_hook_target", model)
    if kind == "features" and hasattr(model, "_reljoin_original_features"):
        target.get_image_features = model._reljoin_original_features
        delattr(model, "_reljoin_original_features")
    elif kind == "visual" and hasattr(model, "_reljoin_original_visual"):
        target.forward = model._reljoin_original_visual
        delattr(model, "_reljoin_original_visual")
    elif hasattr(model, "_reljoin_original_merge"):
        model.merge_multimodal = model._reljoin_original_merge
        delattr(model, "_reljoin_original_merge")
    model._reljoin_hook_installed = False
    if hasattr(model, "_reljoin_hook_kind"):
        delattr(model, "_reljoin_hook_kind")
    if hasattr(model, "_reljoin_hook_target"):
        delattr(model, "_reljoin_hook_target")


def remove_all_reljoin_hooks(model: nn.Module) -> None:
    restore_original_merge(model)


def qwen_hidden_size(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "hidden_size", None):
        return int(cfg.hidden_size)
    text = getattr(cfg, "text_config", None) if cfg is not None else None
    if text is not None and getattr(text, "hidden_size", None):
        return int(text.hidden_size)
    raise RuntimeError("Could not read Qwen2.5-VL hidden_size")


def ovis_hidden_size(model: nn.Module) -> int:
    """Alias so copied unit tests can call either name."""
    try:
        return qwen_hidden_size(model)
    except RuntimeError:
        llm = getattr(model, "llm", None)
        if llm is not None and getattr(llm, "config", None) is not None:
            return int(llm.config.hidden_size)
        raise
