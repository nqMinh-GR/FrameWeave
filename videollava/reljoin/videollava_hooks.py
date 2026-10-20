"""Hooks for FrameWeave Hybrid v6 after the frozen Video-LLaVA projector.

Primary path:
  frames → frozen vision tower → frozen mm_projector → frame-token groups X
  → FrameWeave Hybrid v6 → X̃ (same shape) written back into the visual slice
  → frozen LLM

Vanilla bypass leaves the projected visual tokens unchanged. Sequence length,
position IDs, and attention mask stay identical to Vanilla. Frozen LLM weights
do not receive optimizer steps, but the multimodal forward is not no_grad so
gradients flow back to X̃.
"""
from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class RelJoinHookState:
    enabled: bool = False
    bypass: bool = True  # True = Vanilla (no residual)
    module: Optional[nn.Module] = None
    question_hidden: Optional[torch.Tensor] = None  # [B, d_L]
    timestamps: Optional[torch.Tensor] = None  # [B, K]
    frame_mask: Optional[torch.Tensor] = None  # [B, K]
    last_visual_embeds: Optional[torch.Tensor] = None
    last_updated_embeds: Optional[torch.Tensor] = None
    last_residual_norm: float = 0.0
    last_labels: Optional[torch.Tensor] = None
    last_num_frames: int = 0
    last_tokens_per_frame: int = 0
    cache_prepare: bool = False
    cached_result: Optional[tuple] = None
    answer_ids: Optional[torch.Tensor] = None
    eval_mode: str = "full"


_HOOK_STATE = RelJoinHookState()


def get_hook_state() -> RelJoinHookState:
    return _HOOK_STATE


def set_reljoin_context(
    question_hidden: Optional[torch.Tensor] = None,
    timestamps: Optional[torch.Tensor] = None,
    frame_mask: Optional[torch.Tensor] = None,
    answer_ids: Optional[torch.Tensor] = None,
    text_embeds: Optional[torch.Tensor] = None,
    text_mask: Optional[torch.Tensor] = None,
) -> None:
    """Set per-forward RelJoin context.

    `text_embeds` / `text_mask` are accepted as unused kwargs so older call
    sites fail loudly only if question_hidden is missing at hook time.
    """
    del text_embeds, text_mask
    st = _HOOK_STATE
    st.question_hidden = question_hidden
    st.timestamps = timestamps
    st.frame_mask = frame_mask
    st.answer_ids = answer_ids


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


def begin_prepare_cache() -> None:
    """Reuse one frozen vision/projector prepare for subsequent RelJoin forwards."""
    st = _HOOK_STATE
    st.cache_prepare = True
    st.cached_result = None


def end_prepare_cache() -> None:
    st = _HOOK_STATE
    st.cache_prepare = False
    st.cached_result = None


def freeze_backbone(model: nn.Module) -> None:
    """Freeze vision tower, projector, embeddings, LLM, lm_head."""
    for p in model.parameters():
        p.requires_grad = False


def _ignore_index() -> int:
    try:
        from videollava.constants import IGNORE_INDEX

        return int(IGNORE_INDEX)
    except ImportError:
        return -100


def _maybe_pin_answer_labels(
    labels: Optional[torch.Tensor],
    answer_ids: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """If IMAGE expansion left every label IGNORE, unmask the answer at the sequence tail."""
    if labels is None:
        return labels
    ignore = _ignore_index()
    if int((labels != ignore).sum().item()) > 0:
        return labels
    if answer_ids is None or answer_ids.numel() == 0:
        return labels
    ids = answer_ids.to(device=labels.device, dtype=labels.dtype).view(-1)
    length = min(int(ids.numel()), int(labels.size(1)))
    if length <= 0:
        return labels
    out = labels.clone()
    out[:, -length:] = ids[-length:]
    return out


def enable_reljoin_grads(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, n_kv, slen, hd = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, n_kv, n_rep, slen, hd)
    return hidden_states.reshape(bsz, n_kv * n_rep, slen, hd)


def _llama_sdpa_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[tuple] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    """Llama attention via scaled_dot_product_attention (Flash/mem-efficient, no LxL store)."""
    if output_attentions:
        raise RuntimeError("SDPA RelJoin patch does not support output_attentions=True")

    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    n_kv = getattr(self, "num_key_value_heads", self.num_heads)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, n_kv, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, n_kv, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    has_past = past_key_value is not None
    if has_past:
        kv_seq_len += past_key_value[0].shape[-2]

    try:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
    except TypeError:
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if has_past:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    past_key_value = (key_states, value_states) if use_cache else None

    n_rep = getattr(self, "num_key_value_groups", 1)
    if n_rep != 1:
        try:
            from transformers.models.llama.modeling_llama import repeat_kv as _hf_repeat_kv

            key_states = _hf_repeat_kv(key_states, n_rep)
            value_states = _hf_repeat_kv(value_states, n_rep)
        except ImportError:
            key_states = _repeat_kv(key_states, n_rep)
            value_states = _repeat_kv(value_states, n_rep)

    kv_len = key_states.shape[-2]
    orig_dtype = query_states.dtype
    query_states = query_states.float()
    key_states = key_states.float()
    value_states = value_states.float()

    attn_mask = None
    is_causal = False
    if attention_mask is None:
        is_causal = q_len > 1 and not has_past
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

    dropout_p = 0.0
    hidden = getattr(self, "hidden_size", self.num_heads * self.head_dim)
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )
    attn_output = attn_output.to(orig_dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def patch_llama_sdpa_attention(model: nn.Module) -> int:
    """Replace Llama self-attn with SDPA/Flash on every layer. Returns patched count."""
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    llama = getattr(model, "model", model)
    layers = getattr(llama, "layers", None)
    if layers is None:
        inner = getattr(model, "get_model", None)
        if callable(inner):
            llama = inner()
            layers = getattr(llama, "layers", None)
    if layers is None:
        print("WARN: could not find Llama layers to patch SDPA")
        return 0

    n = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        attn.use_frame_attn = False
        attn.forward = types.MethodType(_llama_sdpa_forward, attn)
        n += 1
    print(f"Patched {n} Llama attention layers → SDPA/Flash")
    return n


def enable_frozen_backbone_checkpointing(model: nn.Module) -> bool:
    """Enable HF gradient checkpointing. Requires model.train() to actually run."""
    if hasattr(model, "config"):
        model.config.use_cache = False
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.use_cache = False

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
        if hasattr(m, "gradient_checkpointing"):
            m.gradient_checkpointing = True
            return True
        return False

    ok = _enable(model)
    inner = getattr(model, "model", None)
    ok = _enable(inner) or ok
    get_model = getattr(model, "get_model", None)
    if callable(get_model):
        try:
            ok = _enable(get_model()) or ok
        except Exception:
            pass
    print(f"Gradient checkpointing enabled={ok}")
    return ok


def configure_frozen_backbone_for_training(model: nn.Module) -> None:
    """Keep checkpointing active (model.train) but freeze dropout/vision eval behavior."""
    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False
    for name, mod in model.named_modules():
        if name.startswith("reljoin"):
            continue
        if isinstance(mod, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            mod.eval()
    for getter in ("get_video_tower", "get_image_tower"):
        fn = getattr(model, getter, None)
        if callable(fn):
            try:
                tower = fn()
                if tower is not None:
                    tower.eval()
            except Exception:
                pass


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
    trainable = [n for n, p in reljoin.named_parameters() if p.requires_grad]
    return trainable


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
    """Frozen text-only question vector. Last non-pad LLM hidden, or raw mean control.

    Does not go through the RelJoin multimodal hook. LLM weights stay frozen;
    this pass is no_grad because h_q does not depend on video.
    """
    text = _question_text(question, choices)
    toks = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = toks["input_ids"].to(device)
    mask = toks.get("attention_mask")
    if mask is None:
        mask = torch.ones_like(input_ids)
    else:
        mask = mask.to(device)
    inner = model.get_model()
    with torch.no_grad():
        embeds = inner.embed_tokens(input_ids)
        if use_raw_mean:
            m = mask.to(dtype=embeds.dtype).unsqueeze(-1)
            hidden = (embeds * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)
            return hidden.to(dtype=dtype)
        llama = inner if hasattr(inner, "layers") else getattr(inner, "model", inner)
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
        embeds = model.get_model().embed_tokens(input_ids).to(dtype=dtype)
    mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
    return embeds, mask


def install_reljoin_hook(model: nn.Module) -> nn.Module:
    """Wrap prepare_inputs_labels_for_multimodal to apply FrameWeave Hybrid in place."""
    if getattr(model, "_reljoin_hook_installed", False):
        return model

    original = model.prepare_inputs_labels_for_multimodal

    def hooked_prepare(self, *args, **kwargs):
        st = _HOOK_STATE
        if st.cache_prepare and st.cached_result is not None:
            result = st.cached_result
        else:
            with torch.no_grad():
                result = original(*args, **kwargs)
            if st.cache_prepare:
                st.cached_result = result
        (
            out_input_ids,
            out_position_ids,
            out_attention_mask,
            out_past,
            new_input_embeds,
            new_labels,
        ) = result

        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]

        if new_input_embeds is None:
            st.last_labels = None
            return result

        st.last_visual_embeds = None
        st.last_updated_embeds = None
        st.last_residual_norm = 0.0

        if not st.enabled or st.bypass or st.module is None:
            st.last_labels = _maybe_pin_answer_labels(new_labels, st.answer_ids)
            return result

        visual_start, visual_end, K, P = _estimate_visual_span(self, input_ids, new_input_embeds)
        st.last_num_frames = K
        st.last_tokens_per_frame = P

        visual = new_input_embeds[:, visual_start:visual_end, :]
        B, Nv, d = visual.shape
        if K <= 0 or P <= 0 or Nv != K * P:
            K = 1
            P = Nv
        frame_tokens = visual.view(B, K, P, d)
        st.last_visual_embeds = visual.detach()

        if st.frame_mask is not None:
            frame_mask = st.frame_mask.to(device=visual.device)
            if frame_mask.shape[0] != B:
                frame_mask = frame_mask[:B]
            if frame_mask.shape[1] < K:
                pad = torch.zeros(B, K - frame_mask.shape[1], device=visual.device, dtype=torch.bool)
                frame_mask = torch.cat([frame_mask, pad], dim=1)
            frame_mask = frame_mask[:, :K]
        else:
            frame_mask = torch.ones(B, K, device=visual.device, dtype=torch.bool)

        token_mask = frame_mask.unsqueeze(-1).expand(B, K, P)

        if st.timestamps is not None:
            timestamps = st.timestamps.to(device=visual.device, dtype=visual.dtype)
            if timestamps.shape[0] != B:
                timestamps = timestamps[:B]
            if timestamps.shape[1] < K:
                pad_t = torch.zeros(
                    B, K - timestamps.shape[1], device=visual.device, dtype=visual.dtype
                )
                timestamps = torch.cat([timestamps, pad_t], dim=1)
            timestamps = timestamps[:, :K]
        else:
            timestamps = (
                torch.arange(K, device=visual.device, dtype=visual.dtype).unsqueeze(0).expand(B, K)
            )

        if st.question_hidden is None:
            raise RuntimeError(
                "FrameWeave enabled but question_hidden not set. "
                "Call set_reljoin_context(question_hidden=...) before forward/generate."
            )
        question_hidden = st.question_hidden.to(device=visual.device)
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
            eval_mode=st.eval_mode or "full",
        )
        if x_tilde.shape != frame_tokens.shape:
            raise AssertionError(
                f"FrameWeave changed token shape {tuple(frame_tokens.shape)} "
                f"→ {tuple(x_tilde.shape)}"
            )
        x_tilde = x_tilde.to(dtype=visual.dtype)

        new_embeds = new_input_embeds.clone()
        new_embeds[:, visual_start:visual_end, :] = x_tilde.reshape(B, K * P, d)
        st.last_updated_embeds = new_embeds[:, visual_start:visual_end, :].detach()
        delta = (x_tilde - frame_tokens.float()).float()
        st.last_residual_norm = float(delta.norm().detach().cpu())

        if new_embeds.shape != new_input_embeds.shape:
            raise AssertionError(
                f"FrameWeave changed sequence shape {tuple(new_input_embeds.shape)} "
                f"→ {tuple(new_embeds.shape)}"
            )

        new_labs = _maybe_pin_answer_labels(new_labels, st.answer_ids)
        st.last_labels = new_labs
        return (
            out_input_ids,
            out_position_ids,
            out_attention_mask,
            out_past,
            new_embeds,
            new_labs,
        )

    model.prepare_inputs_labels_for_multimodal = types.MethodType(hooked_prepare, model)
    model._reljoin_hook_installed = True
    model._reljoin_original_prepare = original
    return model


def _estimate_visual_span(
    model: nn.Module,
    input_ids: Optional[torch.Tensor],
    new_input_embeds: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Estimate (visual_start, visual_end, K, P) for the multimodal embeds."""
    try:
        from videollava.constants import IMAGE_TOKEN_INDEX
    except ImportError:
        IMAGE_TOKEN_INDEX = -200

    try:
        import run_inference_dtr as dtr

        mod = dtr.FrameAttentionModifier
        if mod._instance is not None and mod.get_num_frames() > 0:
            K = mod.get_num_frames()
            start = mod.get_video_token_start_pos()
            end = mod.get_video_token_end_pos()
            Nv = end - start
            P = Nv // max(K, 1)
            return start, end, K, P
    except Exception:
        pass

    if input_ids is not None:
        video_token_indices = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        K = int(video_token_indices.numel())
    else:
        K = int(getattr(model.get_video_tower().config, "num_frames", 8))

    try:
        num_patches = int(model.get_video_tower().num_patches) + 1
    except Exception:
        num_patches = new_input_embeds.shape[1] // max(K, 1)

    visual_start = 0
    if input_ids is not None:
        ids = input_ids[0]
        visual_start = 0
        for tok in ids:
            if int(tok.item()) == IMAGE_TOKEN_INDEX:
                break
            visual_start += 1
    visual_end = visual_start + K * num_patches
    visual_end = min(visual_end, new_input_embeds.shape[1])
    P = (visual_end - visual_start) // max(K, 1)
    return visual_start, visual_start + K * P, K, P


def generate_videollava(
    model: nn.Module,
    tokenizer,
    video_tensor,
    prompt: str,
    num_video_frames: int,
    joining: Optional[nn.Module] = None,
    question_hidden: Optional[torch.Tensor] = None,
    timestamps: Optional[torch.Tensor] = None,
    frame_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 32,
    eval_mode: str | None = None,
    bypass: bool = False,
) -> str:
    """Greedy decode. Vanilla is hook bypass; Hybrid writes the in-place residual."""
    from videollava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from videollava.conversation import SeparatorStyle, conv_templates
    from videollava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token

    configure_reljoin(
        joining,
        enabled=not bypass,
        bypass=bypass,
        eval_mode=eval_mode or "full",
    )
    if not bypass:
        set_reljoin_context(
            question_hidden=question_hidden,
            timestamps=timestamps,
            frame_mask=frame_mask,
        )

    conv = conv_templates["llava_v1"].copy()
    instruction = "".join([DEFAULT_IMAGE_TOKEN] * int(num_video_frames)) + "\n" + prompt
    conv.append_message(conv.roles[0], instruction)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=video_tensor,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    text = tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
    if text.endswith(stop_str):
        text = text[: -len(stop_str)].strip()
    return text


def restore_original_prepare(model: nn.Module) -> None:
    if getattr(model, "_reljoin_hook_installed", False) and hasattr(
        model, "_reljoin_original_prepare"
    ):
        model.prepare_inputs_labels_for_multimodal = model._reljoin_original_prepare
        model._reljoin_hook_installed = False
