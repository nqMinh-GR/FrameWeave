"""Ablation adapters.

Hybrid is the canonical V6 ``RelJoinResidual``. The other adapters register
only parameters that participate in the forward.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ensure_v6_on_path

ensure_v6_on_path()

from reljoin.config import LOCAL_ADAPTER_XAVIER_GAIN  # noqa: E402
from reljoin.reljoin_modules import (  # noqa: E402
    EVAL_IDENTITY,
    FramePooler,
    GlobalRelation,
    HybridResidualHead,
    LocalRelation,
    QuestionEncoder,
    RelJoinResidual,
    _collect_stats,
    init_xavier,
    masked_softmax,
    normalize_time_and_rank,
    pair_mask_from_frames,
    relative_time_features,
    rms_fp32,
)

PAPER_HIDDEN_SIZE = 4096
PAPER_REL_DIM = 256


def gated_rms_delta(
    frame_tokens: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    token_mask: torch.Tensor,
    has_neighbor: torch.Tensor,
    rms_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Canonical RMS calibration. Update is zero when input RMS does not exceed epsilon."""
    sx = rms_fp32(frame_tokens, dim=-1, keepdim=True)
    su = rms_fp32(update, dim=-1, keepdim=True)
    scale = torch.where(
        sx > rms_eps,
        sx / (sx + su + rms_eps),
        torch.zeros_like(sx),
    ).detach()
    valid = token_mask.unsqueeze(-1) & has_neighbor[:, :, None, None]
    delta = valid.to(dtype=torch.float32) * gate.float() * scale * update.float()
    return delta, scale, sx


def _prep_masks(
    token_mask: torch.Tensor,
    frame_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    token_mask = token_mask.to(dtype=torch.bool)
    frame_mask = frame_mask.to(dtype=torch.bool) & token_mask.any(dim=-1)
    pair_mask = pair_mask_from_frames(frame_mask)
    has_neighbor = pair_mask.any(dim=-1)
    return token_mask, frame_mask, pair_mask, has_neighbor


def init_frameweave(module: nn.Module) -> None:
    """Xavier, zero output, zero gate, zero explicit time maps, small-Xavier P_L."""
    module.apply(init_xavier)
    abs_position = getattr(module, "abs_position", None)
    if isinstance(abs_position, nn.Linear):
        nn.init.zeros_(abs_position.weight)
    relation = getattr(module, "global_relation", None)
    for name in ("A_K", "A_V", "a_b"):
        layer = getattr(relation, name, None) if relation is not None else None
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
    local = getattr(module, "local", None)
    if local is not None and hasattr(local, "w_L"):
        nn.init.zeros_(local.w_L)
    head = module.head
    if hasattr(head, "local_adapter"):
        nn.init.xavier_uniform_(head.local_adapter.weight, gain=LOCAL_ADAPTER_XAVIER_GAIN)
    nn.init.zeros_(head.W_o.weight)
    nn.init.zeros_(head.W_o.bias)
    nn.init.zeros_(head.gate.weight)
    nn.init.zeros_(head.gate.bias)


class ContentGlobalRelation(nn.Module):
    """Directed global relation without A_K, A_V, or a_b."""

    def __init__(self, rel_dim: int):
        super().__init__()
        self.rel_dim = rel_dim
        self.W_Q = nn.Linear(rel_dim, rel_dim, bias=False)
        self.W_K = nn.Linear(rel_dim, rel_dim, bias=False)
        self.U_Q = nn.Linear(rel_dim, rel_dim, bias=False)

    def forward(
        self,
        s_bar: torch.Tensor,
        q: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.W_Q(s_bar) + self.U_Q(q)[:, None, :]
        key = self.W_K(s_bar)[:, None, :, :]
        value = s_bar[:, None, :, :]
        logits = torch.einsum("bkr,bkjr->bkj", query, key) / math.sqrt(self.rel_dim)
        beta = masked_softmax(logits, pair_mask, dim=-1)
        message = torch.einsum("bkj,bkjr->bkr", beta, value)
        has_neighbor = pair_mask.any(dim=-1)
        message = message * has_neighbor.unsqueeze(-1).to(dtype=message.dtype)
        return logits, beta, message, has_neighbor


class PureLocalRelation(nn.Module):
    """Token matching that cannot read rG, eG, or target-frame summaries."""

    def __init__(self, rel_dim: int):
        super().__init__()
        self.rel_dim = rel_dim
        self.ln_L = nn.LayerNorm(rel_dim)
        self.local_q = nn.Linear(rel_dim, rel_dim, bias=False)
        self.local_k = nn.Linear(rel_dim, rel_dim, bias=False)
        self.local_v = nn.Linear(rel_dim, rel_dim, bias=False)
        self.local_frame = nn.Linear(rel_dim, rel_dim, bias=False)
        self.local_question = nn.Linear(rel_dim, rel_dim, bias=False)
        self.ln_Q = nn.LayerNorm(rel_dim)
        self.ln_K = nn.LayerNorm(rel_dim)
        self.A_c = nn.Linear(rel_dim, rel_dim, bias=False)
        self.A_e = nn.Linear(rel_dim, rel_dim, bias=False)
        self.A_d = nn.Linear(rel_dim, rel_dim, bias=False)
        self.A_m = nn.Linear(rel_dim, rel_dim, bias=False)
        self.A_q = nn.Linear(rel_dim, rel_dim, bias=False)
        self.b_L = nn.Parameter(torch.zeros(rel_dim))
        self.w_L = nn.Parameter(torch.zeros(rel_dim))

    def forward(
        self,
        v: torch.Tensor,
        s_bar: torch.Tensor,
        q: torch.Tensor,
        token_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        *,
        chunk_p: int = 0,
    ) -> dict[str, torch.Tensor]:
        _batch, _frames, tokens, width = v.shape
        nu = self.ln_L(v)
        # s_bar is broadcast per source frame. There is no s_bar_j term.
        query_raw = (
            self.local_q(nu)
            + self.local_frame(s_bar)[:, :, None, :]
            + self.local_question(q)[:, None, None, :]
        )
        query = self.ln_Q(query_raw)
        key = self.ln_K(self.local_k(nu))
        value = self.local_v(nu)
        scale = math.sqrt(width)
        token_pair_mask = (
            token_mask[:, None, None, :, :]
            & pair_mask[:, :, None, :, None]
            & token_mask[:, :, :, None, None]
        )
        chunk = int(chunk_p) if int(chunk_p) > 0 else tokens
        chunk = max(1, min(chunk, tokens))
        matched_parts: list[torch.Tensor] = []
        for start in range(0, tokens, chunk):
            stop = min(tokens, start + chunk)
            logits = torch.einsum("bipr,bjlr->bipjl", query[:, :, start:stop], key) / scale
            pi = masked_softmax(logits, token_pair_mask[:, :, start:stop], dim=-1)
            matched_parts.append(torch.einsum("bipjl,bjlr->bipjr", pi, value))
        matched = torch.cat(matched_parts, dim=2)
        source = value[:, :, :, None, :]
        message = (
            self.A_c(source)
            + self.A_e(matched)
            + self.A_d(matched - source)
            + self.A_m(source * matched)
            + self.A_q(q)[:, None, None, None, :]
            + self.b_L
        )
        activated = F.silu(message)
        frame_logits = torch.einsum("bipjr,r->bipj", activated, self.w_L) / scale
        frame_mask = token_mask.unsqueeze(-1) & pair_mask.unsqueeze(2)
        gamma = masked_softmax(frame_logits, frame_mask, dim=-1)
        local_message = torch.einsum("bipj,bipjr->bipr", gamma, activated)
        has_neighbor = pair_mask.any(dim=-1)
        valid = token_mask & has_neighbor.unsqueeze(-1)
        local_message = torch.where(valid.unsqueeze(-1), local_message, torch.zeros_like(local_message))
        return {"r_local": local_message, "gamma": gamma, "local_logits": frame_logits}


class GatedResidualHead(nn.Module):
    """Residual head with optional global-message and local-message terms."""

    def __init__(
        self,
        llm_dim: int,
        rel_dim: int,
        *,
        use_global: bool,
        use_local: bool,
        rms_eps: float,
    ):
        super().__init__()
        self.rms_eps = float(rms_eps)
        self.use_global = bool(use_global)
        self.use_local = bool(use_local)
        self.head_token = nn.Linear(rel_dim, rel_dim, bias=False)
        self.head_frame = nn.Linear(rel_dim, rel_dim, bias=False)
        self.head_question = nn.Linear(rel_dim, rel_dim, bias=False)
        self.b_h = nn.Parameter(torch.zeros(rel_dim))
        if self.use_global:
            self.head_global = nn.Linear(rel_dim, rel_dim, bias=False)
        if self.use_local:
            self.local_adapter = nn.Linear(rel_dim, rel_dim, bias=False)
        self.W_o = nn.Linear(rel_dim, llm_dim, bias=True)
        self.gate = nn.Linear(rel_dim, 1)

    def forward(
        self,
        frame_tokens: torch.Tensor,
        v: torch.Tensor,
        s_bar: torch.Tensor,
        r_global: torch.Tensor | None,
        r_local: torch.Tensor | None,
        q: torch.Tensor,
        token_mask: torch.Tensor,
        has_neighbor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        hidden = (
            self.head_token(v)
            + self.head_frame(s_bar)[:, :, None, :]
            + self.head_question(q)[:, None, None, :]
            + self.b_h
        )
        if self.use_global:
            if r_global is None:
                raise RuntimeError("global residual head requires r_global")
            hidden = hidden + self.head_global(r_global)[:, :, None, :]
        if self.use_local:
            if r_local is None:
                raise RuntimeError("local residual head requires r_local")
            hidden = hidden + self.local_adapter(r_local)
        hidden = F.silu(hidden)
        update = self.W_o(hidden)
        gate = torch.sigmoid(self.gate(hidden))
        delta, scale, sx = gated_rms_delta(
            frame_tokens, update, gate, token_mask, has_neighbor, self.rms_eps
        )
        updated = frame_tokens + delta.to(dtype=frame_tokens.dtype)
        stats = _collect_stats(delta, gate, scale, sx, token_mask, has_neighbor, self.rms_eps)
        return updated, delta, stats


class _AdapterBase(nn.Module):
    def __init__(self, llm_dim: int, rel_dim: int, rms_eps: float, time_eps: float, variant: str):
        super().__init__()
        self.llm_dim = int(llm_dim)
        self.rel_dim = int(rel_dim)
        self.rms_eps = float(rms_eps)
        self.time_eps = float(time_eps)
        self.variant = variant
        self.forward_trace: list[str] = []
        self.last_stats: dict[str, float] = {}
        self.last_delta_x: torch.Tensor | None = None

    def _begin(self, eval_mode: str, frame_tokens: torch.Tensor) -> bool:
        mode = (eval_mode or "full").strip().lower()
        self.forward_trace = []
        if mode == EVAL_IDENTITY:
            self.forward_trace.append("identity")
            self.last_delta_x = torch.zeros_like(frame_tokens)
            return True
        if mode != "full":
            raise ValueError(f"Unknown eval_mode {eval_mode!r}")
        return False

    @property
    def W_o(self) -> nn.Linear:
        return self.head.W_o

    def _finish(self, frame_tokens: torch.Tensor, updated: torch.Tensor, delta: torch.Tensor, stats: dict) -> torch.Tensor:
        if not bool(torch.isfinite(delta).all()):
            raise RuntimeError(f"Non-finite residual in {self.variant}")
        self.last_stats = dict(stats)
        self.last_delta_x = delta.detach()
        return updated


class FrameOnlyAdapter(_AdapterBase):
    """Question encoder, within-frame pooling, and a three-input residual head. No frame relation."""

    def __init__(self, llm_dim: int, rel_dim: int, rms_eps: float, time_eps: float):
        super().__init__(llm_dim, rel_dim, rms_eps, time_eps, "frame_only")
        self.question_encoder = QuestionEncoder(llm_dim, rel_dim)
        self.frame_pooler = FramePooler(llm_dim, rel_dim)
        self.abs_position = nn.Linear(2, rel_dim, bias=False)
        self.head = GatedResidualHead(
            llm_dim, rel_dim, use_global=False, use_local=False, rms_eps=rms_eps
        )
        self.last_has_neighbor: torch.Tensor | None = None
        init_frameweave(self)

    def forward(
        self,
        frame_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        frame_mask: torch.Tensor,
        timestamps: torch.Tensor,
        question_hidden: torch.Tensor,
        *,
        eval_mode: str = "full",
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if self._begin(eval_mode, frame_tokens):
            return frame_tokens
        frame_tokens = frame_tokens.float()
        # chi_i is the valid-frame neighbor mask. Having no relation branch must not zero it.
        token_mask, frame_mask, _pair_mask, has_neighbor = _prep_masks(token_mask, frame_mask)
        question = self.question_encoder(question_hidden.float())
        self.forward_trace.append("pool")
        summary, _alpha, values = self.frame_pooler(frame_tokens, token_mask, question, frame_mask)
        self.forward_trace.append("time")
        tau, rho = normalize_time_and_rank(timestamps.float(), frame_mask, eps=self.time_eps)
        absolute = self.abs_position(torch.stack([tau, rho], dim=-1))
        absolute = absolute * frame_mask.unsqueeze(-1).to(dtype=absolute.dtype)
        summary_bar = summary + absolute
        self.forward_trace.append("head")
        updated, delta, stats = self.head(
            frame_tokens, values, summary_bar, None, None, question, token_mask, has_neighbor
        )
        self.last_has_neighbor = has_neighbor.detach()
        return self._finish(frame_tokens, updated, delta, stats)


class GlobalOnlyAdapter(_AdapterBase):
    def __init__(self, llm_dim: int, rel_dim: int, rms_eps: float, time_eps: float):
        super().__init__(llm_dim, rel_dim, rms_eps, time_eps, "global_only")
        self.question_encoder = QuestionEncoder(llm_dim, rel_dim)
        self.frame_pooler = FramePooler(llm_dim, rel_dim)
        self.abs_position = nn.Linear(2, rel_dim, bias=False)
        self.global_relation = GlobalRelation(rel_dim)
        self.head = GatedResidualHead(
            llm_dim, rel_dim, use_global=True, use_local=False, rms_eps=rms_eps
        )
        init_frameweave(self)

    def forward(
        self,
        frame_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        frame_mask: torch.Tensor,
        timestamps: torch.Tensor,
        question_hidden: torch.Tensor,
        *,
        eval_mode: str = "full",
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if self._begin(eval_mode, frame_tokens):
            return frame_tokens
        frame_tokens = frame_tokens.float()
        token_mask, frame_mask, pair_mask, has_neighbor = _prep_masks(token_mask, frame_mask)
        question = self.question_encoder(question_hidden.float())
        self.forward_trace.append("pool")
        summary, _alpha, values = self.frame_pooler(frame_tokens, token_mask, question, frame_mask)
        self.forward_trace.append("time")
        tau, rho = normalize_time_and_rank(timestamps.float(), frame_mask, eps=self.time_eps)
        absolute = self.abs_position(torch.stack([tau, rho], dim=-1))
        absolute = absolute * frame_mask.unsqueeze(-1).to(dtype=absolute.dtype)
        summary_bar = summary + absolute
        phi = relative_time_features(tau, rho)
        self.forward_trace.append("global")
        _logits, _beta, message, has_neighbor = self.global_relation(
            summary_bar, question, phi, pair_mask
        )
        self.forward_trace.append("head")
        updated, delta, stats = self.head(
            frame_tokens, values, summary_bar, message, None, question, token_mask, has_neighbor
        )
        return self._finish(frame_tokens, updated, delta, stats)


class LocalOnlyAdapter(_AdapterBase):
    def __init__(self, llm_dim: int, rel_dim: int, rms_eps: float, time_eps: float, local_token_chunk: int):
        super().__init__(llm_dim, rel_dim, rms_eps, time_eps, "local_only")
        self.local_token_chunk = int(local_token_chunk)
        self.question_encoder = QuestionEncoder(llm_dim, rel_dim)
        self.frame_pooler = FramePooler(llm_dim, rel_dim)
        self.abs_position = nn.Linear(2, rel_dim, bias=False)
        self.local = PureLocalRelation(rel_dim)
        self.head = GatedResidualHead(
            llm_dim, rel_dim, use_global=False, use_local=True, rms_eps=rms_eps
        )
        init_frameweave(self)

    def forward(
        self,
        frame_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        frame_mask: torch.Tensor,
        timestamps: torch.Tensor,
        question_hidden: torch.Tensor,
        *,
        eval_mode: str = "full",
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if self._begin(eval_mode, frame_tokens):
            return frame_tokens
        frame_tokens = frame_tokens.float()
        token_mask, frame_mask, pair_mask, has_neighbor = _prep_masks(token_mask, frame_mask)
        question = self.question_encoder(question_hidden.float())
        self.forward_trace.append("pool")
        summary, _alpha, values = self.frame_pooler(frame_tokens, token_mask, question, frame_mask)
        self.forward_trace.append("time")
        tau, rho = normalize_time_and_rank(timestamps.float(), frame_mask, eps=self.time_eps)
        absolute = self.abs_position(torch.stack([tau, rho], dim=-1))
        absolute = absolute * frame_mask.unsqueeze(-1).to(dtype=absolute.dtype)
        # Source-frame summary only. PureLocalRelation never reads summary_bar of frame j.
        summary_bar = summary + absolute
        self.forward_trace.append("local")
        local_out = self.local(
            values,
            summary_bar,
            question,
            token_mask,
            pair_mask,
            chunk_p=self.local_token_chunk,
        )
        self.forward_trace.append("head")
        updated, delta, stats = self.head(
            frame_tokens,
            values,
            summary_bar,
            None,
            local_out["r_local"],
            question,
            token_mask,
            has_neighbor,
        )
        return self._finish(frame_tokens, updated, delta, stats)


class HybridNoTimeAdapter(_AdapterBase):
    def __init__(self, llm_dim: int, rel_dim: int, rms_eps: float, time_eps: float, local_token_chunk: int):
        super().__init__(llm_dim, rel_dim, rms_eps, time_eps, "hybrid_no_time")
        self.local_token_chunk = int(local_token_chunk)
        self.question_encoder = QuestionEncoder(llm_dim, rel_dim)
        self.frame_pooler = FramePooler(llm_dim, rel_dim)
        self.global_relation = ContentGlobalRelation(rel_dim)
        self.local = LocalRelation(rel_dim)
        self.head = HybridResidualHead(llm_dim, rel_dim, rms_eps=rms_eps)
        init_frameweave(self)

    def forward(
        self,
        frame_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        frame_mask: torch.Tensor,
        timestamps: torch.Tensor,
        question_hidden: torch.Tensor,
        *,
        eval_mode: str = "full",
        **kwargs,
    ) -> torch.Tensor:
        del kwargs, timestamps
        if self._begin(eval_mode, frame_tokens):
            return frame_tokens
        frame_tokens = frame_tokens.float()
        token_mask, frame_mask, pair_mask, has_neighbor = _prep_masks(token_mask, frame_mask)
        question = self.question_encoder(question_hidden.float())
        self.forward_trace.append("pool")
        summary, _alpha, values = self.frame_pooler(frame_tokens, token_mask, question, frame_mask)
        summary_bar = summary
        self.forward_trace.append("global")
        global_logits, _beta, message, has_neighbor = self.global_relation(
            summary_bar, question, pair_mask
        )
        self.forward_trace.append("local")
        local_out = self.local(
            v=values,
            s_bar=summary_bar,
            r_global=message,
            q=question,
            global_logits=global_logits,
            token_mask=token_mask,
            pair_mask=pair_mask,
            chunk_p=self.local_token_chunk,
        )
        self.forward_trace.append("head")
        updated, delta, stats, _extras = self.head(
            frame_tokens,
            values,
            summary_bar,
            message,
            local_out["r_local"],
            question,
            token_mask,
            has_neighbor,
        )
        return self._finish(frame_tokens, updated, delta, stats)


class TokenMLPAdapter(nn.Module):
    """Per-token bottleneck. No pooling, attention, or cross-token normalization."""

    def __init__(
        self,
        llm_dim: int,
        rel_dim: int,
        hidden: int,
        rms_eps: float,
        time_eps: float,
    ):
        super().__init__()
        self.llm_dim = int(llm_dim)
        self.rel_dim = int(rel_dim)
        self.hidden = int(hidden)
        self.rms_eps = float(rms_eps)
        self.time_eps = float(time_eps)
        self.variant = "token_mlp"
        self.forward_trace: list[str] = []
        self.last_stats: dict[str, float] = {}
        self.last_delta_x: torch.Tensor | None = None
        self.question_encoder = QuestionEncoder(llm_dim, rel_dim)
        self.ln_x = nn.LayerNorm(llm_dim)
        self.B_x = nn.Linear(llm_dim, hidden, bias=False)
        self.B_q = nn.Linear(rel_dim, hidden, bias=False)
        self.B_t = nn.Linear(2, hidden, bias=False)
        self.b_h = nn.Parameter(torch.zeros(hidden))
        self.B_o = nn.Linear(hidden, self.llm_dim, bias=True)
        self.gate = nn.Linear(hidden, 1)
        self.reset_parameters()

    @property
    def W_o(self) -> nn.Linear:
        return self.B_o

    def reset_parameters(self) -> None:
        self.apply(init_xavier)
        nn.init.zeros_(self.b_h)
        nn.init.zeros_(self.B_o.weight)
        nn.init.zeros_(self.B_o.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        frame_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        frame_mask: torch.Tensor,
        timestamps: torch.Tensor,
        question_hidden: torch.Tensor,
        *,
        eval_mode: str = "full",
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        self.forward_trace = []
        if (eval_mode or "full").strip().lower() == EVAL_IDENTITY:
            self.forward_trace.append("identity")
            self.last_delta_x = torch.zeros_like(frame_tokens)
            return frame_tokens
        frame_tokens = frame_tokens.float()
        token_mask, frame_mask, _pair, has_neighbor = _prep_masks(token_mask, frame_mask)
        self.forward_trace.append("token_mlp")
        question = self.question_encoder(question_hidden.float())
        tau, rho = normalize_time_and_rank(timestamps.float(), frame_mask, eps=self.time_eps)
        time = torch.stack([tau, rho], dim=-1)
        hidden = F.silu(
            self.B_x(self.ln_x(frame_tokens))
            + self.B_q(question)[:, None, None, :]
            + self.B_t(time)[:, :, None, :]
            + self.b_h
        )
        update = self.B_o(hidden)
        gate = torch.sigmoid(self.gate(hidden))
        delta, scale, sx = gated_rms_delta(
            frame_tokens, update, gate, token_mask, has_neighbor, self.rms_eps
        )
        updated = frame_tokens + delta.to(dtype=frame_tokens.dtype)
        if not bool(torch.isfinite(delta).all()):
            raise RuntimeError("Non-finite token_mlp residual")
        self.last_stats = _collect_stats(delta, gate, scale, sx, token_mask, has_neighbor, self.rms_eps)
        self.last_delta_x = delta.detach()
        return updated


def token_mlp_param_count(llm_dim: int, rel_dim: int, hidden: int) -> int:
    question = llm_dim * rel_dim + rel_dim + 2 * rel_dim
    layer_norm = 2 * llm_dim
    per_hidden = 2 * llm_dim + rel_dim + 4
    constant = question + layer_norm + llm_dim + 1
    return constant + int(hidden) * per_hidden


def count_parameters(module: nn.Module) -> int:
    return sum(int(param.numel()) for param in module.parameters())


def build_hybrid(llm_dim: int, rel_dim: int, rms_eps: float, time_eps: float, local_token_chunk: int) -> RelJoinResidual:
    return RelJoinResidual(
        llm_dim,
        rel_dim=rel_dim,
        residual_eps=rms_eps,
        rms_eps=rms_eps,
        time_eps=time_eps,
        local_token_chunk=local_token_chunk,
        variant="Hybrid",
    )


_HIDDEN_CACHE: dict[tuple[int, int, float], dict] = {}


def select_token_mlp_hidden(
    llm_dim: int = PAPER_HIDDEN_SIZE,
    rel_dim: int = PAPER_REL_DIM,
    rms_eps: float = 1e-8,
    time_eps: float = 1e-8,
) -> dict:
    """Choose H from the real hybrid parameter count. Not tuned on HERBench."""
    key = (int(llm_dim), int(rel_dim), float(rms_eps))
    cached = _HIDDEN_CACHE.get(key)
    if cached is not None:
        return cached
    hybrid = build_hybrid(llm_dim, rel_dim, rms_eps, time_eps, 0)
    target = count_parameters(hybrid)
    # const + H * per = token_mlp_param_count
    per = token_mlp_param_count(llm_dim, rel_dim, 1) - token_mlp_param_count(llm_dim, rel_dim, 0)
    const = token_mlp_param_count(llm_dim, rel_dim, 0)
    guess = max(1, int(round((target - const) / per)))
    best: tuple[int, int, float, int] | None = None
    for hidden in range(max(1, guess - 8), guess + 9):
        total = const + per * hidden
        diff = abs(total - target)
        if best is None or diff < best[0] or (diff == best[0] and hidden < best[1]):
            best = (diff, hidden, diff / target, total)
    assert best is not None
    if best[2] > 0.01:
        raise RuntimeError(
            f"No token_mlp width within 1% of hybrid ({target}). Closest H={best[1]} rel={best[2]:.4f}"
        )
    module = TokenMLPAdapter(llm_dim, rel_dim, best[1], rms_eps, time_eps)
    actual = count_parameters(module)
    if actual != best[3]:
        raise RuntimeError(f"token_mlp formula {best[3]} != registered parameters {actual}")
    payload = {
        "hidden": best[1],
        "token_mlp_params": actual,
        "hybrid_params": target,
        "abs_diff": best[0],
        "rel_diff": best[2],
        "llm_dim": int(llm_dim),
        "rel_dim": int(rel_dim),
    }
    _HIDDEN_CACHE[key] = payload
    return payload


def build_adapter(
    variant: str,
    llm_dim: int,
    rel_dim: int = PAPER_REL_DIM,
    *,
    hidden: Optional[int] = None,
    rms_eps: float = 1e-8,
    time_eps: float = 1e-8,
    local_token_chunk: int = 0,
) -> nn.Module | None:
    name = str(variant).strip()
    if name == "vanilla":
        return None
    if name == "hybrid":
        return build_hybrid(llm_dim, rel_dim, rms_eps, time_eps, local_token_chunk)
    if name == "frame_only":
        return FrameOnlyAdapter(llm_dim, rel_dim, rms_eps, time_eps)
    if name == "global_only":
        return GlobalOnlyAdapter(llm_dim, rel_dim, rms_eps, time_eps)
    if name == "local_only":
        return LocalOnlyAdapter(llm_dim, rel_dim, rms_eps, time_eps, local_token_chunk)
    if name == "hybrid_no_time":
        return HybridNoTimeAdapter(llm_dim, rel_dim, rms_eps, time_eps, local_token_chunk)
    if name == "token_mlp":
        width = hidden
        if width is None:
            width = int(select_token_mlp_hidden(llm_dim, rel_dim, rms_eps, time_eps)["hidden"])
        return TokenMLPAdapter(llm_dim, rel_dim, int(width), rms_eps, time_eps)
    raise ValueError(f"Unknown variant {variant!r}")


def parameter_manifest(module: nn.Module | None, variant: str) -> dict:
    if module is None:
        return {
            "variant": variant,
            "total": 0,
            "trainable": 0,
            "active": 0,
            "names": [],
        }
    names = [name for name, param in module.named_parameters() if param.requires_grad]
    total = count_parameters(module)
    trainable = sum(int(param.numel()) for param in module.parameters() if param.requires_grad)
    return {
        "variant": variant,
        "total": total,
        "trainable": trainable,
        "active": total,
        "names": names,
    }
