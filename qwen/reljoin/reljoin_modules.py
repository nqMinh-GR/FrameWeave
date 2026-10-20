"""FrameWeave Hybrid v6.0 (V5.1): in-place token residual with global time + local match.

Canonical update:

    Δx_ip = m_ip χ_i a_ip c_ip u_ip
    x̃_ip  = x_ip + Δx_ip

Clean frozen Qwen2.5-VL + question-guided pooling + non-periodic (τ, ρ) time
+ directed global relation + V5.1 post-normalized local Q/K (no local time,
small-Xavier P_L). W_o = 0 at init is exact Vanilla. K=1 is identity because
χ_i = 0. Sequence length is unchanged; no join token is inserted.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    LOCAL_ADAPTER_XAVIER_GAIN,
    REL_DIM,
    RESIDUAL_EPS,
    RMS_EPS,
    TIME_EPS,
    VALID_VARIANTS,
    architecture_variant,
)

EVAL_IDENTITY = "identity"
EVAL_FULL = "full"
EVAL_MODES = (EVAL_IDENTITY, EVAL_FULL)

TIME_HYBRID = "hybrid"
TIME_ZERO = "zero"
TIME_DIM = 4


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax over `mask`; all-masked slices become 0 (no NaN). Compute in FP32."""
    mask = mask.to(dtype=torch.bool)
    filled = logits.float().masked_fill(~mask, float("-inf"))
    all_masked = ~mask.any(dim=dim, keepdim=True)
    filled = torch.where(all_masked, torch.zeros_like(filled), filled)
    probs = torch.softmax(filled, dim=dim)
    probs = probs.masked_fill(~mask, 0.0)
    return probs.to(logits.dtype)


def rms_fp32(v: torch.Tensor, dim: int = -1, keepdim: bool = True) -> torch.Tensor:
    """RMS over hidden dim in FP32: sqrt(mean(v^2))."""
    return torch.sqrt(v.float().pow(2).mean(dim=dim, keepdim=keepdim).clamp_min(0.0))


def pair_mask_from_frames(frame_mask: torch.Tensor) -> torch.Tensor:
    """M_ij = f_i f_j 1[i ≠ j]."""
    B, K = frame_mask.shape
    both = frame_mask[:, :, None] & frame_mask[:, None, :]
    eye = torch.eye(K, device=frame_mask.device, dtype=torch.bool).unsqueeze(0)
    return both & ~eye


def normalize_tau(
    timestamps: torch.Tensor,
    frame_mask: torch.Tensor,
    eps_t: float = TIME_EPS,
) -> torch.Tensor:
    """Per-sample τ_i = (t_i - t_min) / (t_max - t_min + ε) in [0, 1].

    <2 valid frames or zero span → 0.
    """
    mask = frame_mask.to(dtype=torch.bool)
    n_valid = mask.sum(dim=-1, keepdim=True)
    pos_inf = torch.finfo(timestamps.dtype).max
    neg_inf = torch.finfo(timestamps.dtype).min
    tmin = timestamps.masked_fill(~mask, pos_inf).min(dim=-1, keepdim=True).values
    tmax = timestamps.masked_fill(~mask, neg_inf).max(dim=-1, keepdim=True).values
    span = tmax - tmin
    bad = (n_valid < 2) | (span.abs() <= eps_t) | ~torch.isfinite(tmin) | ~torch.isfinite(tmax)
    tau = (timestamps - tmin) / (span + eps_t)
    tau = torch.where(bad, torch.zeros_like(tau), tau)
    return tau.masked_fill(~mask, 0.0)


def ordinal_rho(frame_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """ρ_i = o_i / (K_valid - 1) among valid frames. K_valid < 2 → 0."""
    mask = frame_mask.to(dtype=torch.bool)
    order = (mask.to(dtype=torch.long).cumsum(dim=-1) - 1).clamp_min(0)
    k_valid = mask.sum(dim=-1, keepdim=True)
    denom = (k_valid - 1).clamp_min(1).to(dtype=dtype)
    rho = order.to(dtype=dtype) / denom
    rho = torch.where(k_valid < 2, torch.zeros_like(rho), rho)
    return rho.masked_fill(~mask, 0.0)


def normalize_time_and_rank(
    timestamps: torch.Tensor,
    frame_mask: torch.Tensor,
    eps: float = TIME_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (τ, ρ) with invalid / <2-frame samples zeroed."""
    tau = normalize_tau(timestamps, frame_mask, eps_t=eps)
    rho = ordinal_rho(frame_mask, dtype=timestamps.dtype)
    return tau, rho


def relative_time_features(tau: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """φ_ij = [δτ, |δτ|, δρ, |δρ|] with δ_ij = ·_j - ·_i. Shape [B, K, K, 4]."""
    dt = tau[:, None, :] - tau[:, :, None]
    dp = rho[:, None, :] - rho[:, :, None]
    return torch.stack([dt, dt.abs(), dp, dp.abs()], dim=-1)


def variant_flags(variant: str) -> dict:
    arch = architecture_variant(variant)
    if arch not in VALID_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Expected {VALID_VARIANTS}")
    if arch in ("GlobalTimeFix", "HybridLocalOff"):
        return {"local_enabled": False, "use_global": True, "time_mode": TIME_HYBRID}
    if arch == "Hybrid":
        return {"local_enabled": True, "use_global": True, "time_mode": TIME_HYBRID}
    if arch == "HybridNoTime":
        return {"local_enabled": True, "use_global": True, "time_mode": TIME_ZERO}
    return {"local_enabled": True, "use_global": False, "time_mode": TIME_HYBRID}


def _xavier_linear(module: nn.Linear) -> None:
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def init_xavier(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        _xavier_linear(module)


def _zero_linear(module: nn.Linear) -> None:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _empty_residual_stats() -> dict[str, float]:
    return {
        "residual_rms": 0.0,
        "delta_rms": 0.0,
        "max_g": 0.0,
        "gate_mean": 0.0,
        "gate_p50": 0.0,
        "gate_p95": 0.0,
        "calib_mean": 0.0,
        "rho_max": 0.0,
        "base_rms": 0.0,
        "residual_over_base_rms": 0.0,
        "relation_entropy": 0.0,
        "global_entropy": 0.0,
        "local_frame_entropy": 0.0,
        "token_match_entropy": 0.0,
        "normalized_token_match_entropy": 0.0,
        "content_logit_abs_mean": 0.0,
        "temporal_logit_abs_mean": 0.0,
        "temporal_over_content": 0.0,
        "local_ratio": 0.0,
        "local_logit_abs_mean": 0.0,
        "time_key_ratio": 0.0,
        "time_value_ratio": 0.0,
        "absolute_position_ratio": 0.0,
    }


class QuestionEncoder(nn.Module):
    """q = LN_q(W_q sg(h_q))."""

    def __init__(self, llm_dim: int, rel_dim: int):
        super().__init__()
        self.proj = nn.Linear(llm_dim, rel_dim)
        self.ln = nn.LayerNorm(rel_dim)

    def forward(self, question_hidden: torch.Tensor) -> torch.Tensor:
        return self.ln(self.proj(question_hidden.detach()))


class FramePooler(nn.Module):
    """Question-guided local frame pooling.

    z = LN_x(X), v = W_k z, α = MSoftmax_p(v^T q / √R ; m)
    s = f ⊙ LN_s(W_v Σ_p α x)
    """

    def __init__(self, llm_dim: int, rel_dim: int):
        super().__init__()
        self.rel_dim = rel_dim
        self.ln_x = nn.LayerNorm(llm_dim)
        self.token_key = nn.Linear(llm_dim, rel_dim, bias=False)
        self.frame_value = nn.Linear(llm_dim, rel_dim, bias=False)
        self.ln_s = nn.LayerNorm(rel_dim)

    def forward(
        self,
        frame_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        q: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.ln_x(frame_tokens)
        v = self.token_key(z)
        pool_logits = torch.einsum("bkpr,br->bkp", v, q) / math.sqrt(self.rel_dim)
        alpha = masked_softmax(pool_logits, token_mask, dim=-1)
        pooled = torch.einsum("bkp,bkpd->bkd", alpha, frame_tokens)
        s = self.ln_s(self.frame_value(pooled))
        s = s * frame_mask.unsqueeze(-1).to(dtype=s.dtype)
        return s, alpha, v


class GlobalRelation(nn.Module):
    """Directed V3-TimeFix frame relation. A_K = A_V = a_b = 0 at init."""

    def __init__(self, rel_dim: int, time_dim: int = TIME_DIM):
        super().__init__()
        self.rel_dim = rel_dim
        self.W_Q = nn.Linear(rel_dim, rel_dim, bias=False)
        self.W_K = nn.Linear(rel_dim, rel_dim, bias=False)
        self.U_Q = nn.Linear(rel_dim, rel_dim, bias=False)
        self.A_K = nn.Linear(time_dim, rel_dim, bias=False)
        self.A_V = nn.Linear(time_dim, rel_dim, bias=False)
        self.a_b = nn.Linear(time_dim, 1, bias=False)

    def forward(
        self,
        s_bar: torch.Tensor,
        q: torch.Tensor,
        phi: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = self.W_Q(s_bar) + self.U_Q(q)[:, None, :]
        k_pair = self.W_K(s_bar)[:, None, :, :] + self.A_K(phi)
        v_pair = s_bar[:, None, :, :] + self.A_V(phi)
        scale = math.sqrt(self.rel_dim)
        global_logits = torch.einsum("bkr,bkjr->bkj", Q, k_pair) / scale
        global_logits = global_logits - self.a_b(phi).squeeze(-1)
        beta = masked_softmax(global_logits, pair_mask, dim=-1)
        r_global = torch.einsum("bkj,bkjr->bkr", beta, v_pair)
        has_neighbor = pair_mask.any(dim=-1)
        r_global = r_global * has_neighbor.unsqueeze(-1).to(dtype=r_global.dtype)
        return global_logits, beta, r_global, has_neighbor

    def logit_parts(
        self,
        s_bar: torch.Tensor,
        q: torch.Tensor,
        phi: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[float, float]:
        Q = self.W_Q(s_bar) + self.U_Q(q)[:, None, :]
        scale = math.sqrt(self.rel_dim)
        e_c = torch.einsum("bkr,bkjr->bkj", Q, self.W_K(s_bar)[:, None, :, :]) / scale
        e_t = torch.einsum("bkr,bkjr->bkj", Q, self.A_K(phi)) / scale
        e_t = e_t - self.a_b(phi).squeeze(-1)
        if pair_mask.any():
            return (
                float(e_c[pair_mask].detach().abs().mean()),
                float(e_t[pair_mask].detach().abs().mean()),
            )
        return 0.0, 0.0

    def time_ratios(
        self,
        s_bar: torch.Tensor,
        phi: torch.Tensor,
        pair_mask: torch.Tensor,
        rms_eps: float,
    ) -> tuple[float, float]:
        if not bool(pair_mask.any()):
            return 0.0, 0.0
        B, K, _r = s_bar.shape
        wk_pair = self.W_K(s_bar)[:, None, :, :].expand(B, K, K, -1)
        ak_pair = self.A_K(phi)
        av_pair = self.A_V(phi)
        sv_pair = s_bar[:, None, :, :].expand(B, K, K, -1)
        tk = rms_fp32(ak_pair.detach()[pair_mask], dim=-1, keepdim=False).mean()
        ck = rms_fp32(wk_pair.detach()[pair_mask], dim=-1, keepdim=False).mean()
        tv = rms_fp32(av_pair.detach()[pair_mask], dim=-1, keepdim=False).mean()
        cv = rms_fp32(sv_pair.detach()[pair_mask], dim=-1, keepdim=False).mean()
        time_key = float(tk / (ck + rms_eps)) if float(ck) > 0 else 0.0
        time_value = float(tv / (cv + rms_eps)) if float(cv) > 0 else 0.0
        return time_key, time_value


class LocalRelation(nn.Module):
    """V5.1 global-conditioned token matching. No A_t; post-LN query/key."""

    def __init__(self, rel_dim: int):
        super().__init__()
        self.rel_dim = rel_dim
        self.ln_L = nn.LayerNorm(rel_dim)
        self.local_q = nn.Linear(rel_dim, rel_dim, bias=False)
        self.local_k = nn.Linear(rel_dim, rel_dim, bias=False)
        self.local_v = nn.Linear(rel_dim, rel_dim, bias=False)
        self.local_frame = nn.Linear(rel_dim, rel_dim, bias=False)
        self.local_global = nn.Linear(rel_dim, rel_dim, bias=False)
        self.local_question = nn.Linear(rel_dim, rel_dim, bias=False)
        self.ln_Q = nn.LayerNorm(rel_dim)
        self.ln_K = nn.LayerNorm(rel_dim)
        self.A_c = nn.Linear(rel_dim, rel_dim, bias=False)
        self.A_e = nn.Linear(rel_dim, rel_dim, bias=False)
        self.A_d = nn.Linear(rel_dim, rel_dim, bias=False)
        self.A_m = nn.Linear(rel_dim, rel_dim, bias=False)
        self.A_s = nn.Linear(rel_dim, rel_dim, bias=False)
        self.A_r = nn.Linear(rel_dim, rel_dim, bias=False)
        self.A_q = nn.Linear(rel_dim, rel_dim, bias=False)
        self.b_L = nn.Parameter(torch.zeros(rel_dim))
        self.w_L = nn.Parameter(torch.zeros(rel_dim))

    def forward(
        self,
        v: torch.Tensor,
        s_bar: torch.Tensor,
        r_global: torch.Tensor,
        q: torch.Tensor,
        global_logits: torch.Tensor,
        token_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        *,
        chunk_p: int = 0,
    ) -> dict[str, torch.Tensor | float]:
        B, K, P, dr = v.shape
        nu = self.ln_L(v)
        q_raw = (
            self.local_q(nu)
            + self.local_frame(s_bar)[:, :, None, :]
            + self.local_global(r_global)[:, :, None, :]
            + self.local_question(q)[:, None, None, :]
        )
        q_local = self.ln_Q(q_raw)
        k_local = self.ln_K(self.local_k(nu))
        c_local = self.local_v(nu)
        scale = math.sqrt(dr)

        token_pair_mask = (
            token_mask[:, None, None, :, :]
            & pair_mask[:, :, None, :, None]
            & token_mask[:, :, :, None, None]
        )
        chunk = int(chunk_p) if int(chunk_p) > 0 else P
        chunk = max(1, min(chunk, P))

        matched_parts: list[torch.Tensor] = []
        pi_ent_sum = v.new_zeros(())
        pi_ent_n = v.new_zeros(())
        norm_ent_sum = v.new_zeros(())
        norm_ent_n = v.new_zeros(())
        p_valid = token_mask.to(dtype=torch.float32).sum(dim=-1).clamp_min(1.0)
        log_p = p_valid.clamp_min(2.0).log()
        for p0 in range(0, P, chunk):
            p1 = min(P, p0 + chunk)
            q_c = q_local[:, :, p0:p1]
            logits = torch.einsum("bipr,bjlr->bipjl", q_c, k_local) / scale
            mask_c = token_pair_mask[:, :, p0:p1]
            pi = masked_softmax(logits, mask_c, dim=-1)
            matched_parts.append(torch.einsum("bipjl,bjlr->bipjr", pi, c_local))
            p_f = pi.detach().float().clamp_min(1e-12)
            ent = -(p_f * p_f.log()).sum(dim=-1)
            valid = mask_c.any(dim=-1)
            if valid.any():
                pi_ent_sum = pi_ent_sum + ent[valid].sum()
                pi_ent_n = pi_ent_n + valid.to(dtype=ent.dtype).sum()
                enough = valid & (p_valid[:, None, None, :] >= 2)
                if enough.any():
                    denom = log_p[:, None, None, :].expand_as(ent)
                    norm_ent_sum = norm_ent_sum + (ent / denom.clamp_min(1e-8))[enough].sum()
                    norm_ent_n = norm_ent_n + enough.to(dtype=ent.dtype).sum()
        matched = torch.cat(matched_parts, dim=2)
        token_match_entropy = float(pi_ent_sum / pi_ent_n.clamp_min(1.0)) if float(pi_ent_n) else 0.0
        normalized_token_match_entropy = (
            float(norm_ent_sum / norm_ent_n.clamp_min(1.0)) if float(norm_ent_n) else 0.0
        )

        c_pair = c_local[:, :, :, None, :]
        s_diff = (s_bar[:, None, :, :] - s_bar[:, :, None, :])[:, :, None, :, :]
        zeta = (
            self.A_c(c_pair)
            + self.A_e(matched)
            + self.A_d(matched - c_pair)
            + self.A_m(c_pair * matched)
            + self.A_s(s_diff)
            + self.A_r(r_global)[:, :, None, None, :]
            + self.A_q(q)[:, None, None, None, :]
            + self.b_L
        )
        g = F.silu(zeta)
        ell = torch.einsum("bipjr,r->bipj", g, self.w_L) / scale
        frame_logits = global_logits[:, :, None, :] + ell
        frame_mask_tok = token_mask.unsqueeze(-1) & pair_mask.unsqueeze(2)
        gamma = masked_softmax(frame_logits, frame_mask_tok, dim=-1)
        r_local = torch.einsum("bipj,bipjr->bipr", gamma, g)
        has_neighbor = pair_mask.any(dim=-1)
        valid = token_mask & has_neighbor.unsqueeze(-1)
        r_local = torch.where(valid.unsqueeze(-1), r_local, torch.zeros_like(r_local))
        local_logit_abs_mean = 0.0
        if bool(frame_mask_tok.any()):
            local_logit_abs_mean = float(ell.detach()[frame_mask_tok].float().abs().mean())
        return {
            "r_local": r_local,
            "gamma": gamma,
            "local_logits": ell,
            "token_match_entropy": token_match_entropy,
            "normalized_token_match_entropy": normalized_token_match_entropy,
            "local_logit_abs_mean": local_logit_abs_mean,
        }


class HybridResidualHead(nn.Module):
    """Additive hybrid hidden + gated RMS-calibrated residual. W_o = gate = 0; P_L small-Xavier."""

    def __init__(self, llm_dim: int, rel_dim: int, rms_eps: float = RMS_EPS):
        super().__init__()
        self.rel_dim = rel_dim
        self.rms_eps = rms_eps
        self.head_token = nn.Linear(rel_dim, rel_dim, bias=False)
        self.head_frame = nn.Linear(rel_dim, rel_dim, bias=False)
        self.head_global = nn.Linear(rel_dim, rel_dim, bias=False)
        self.head_question = nn.Linear(rel_dim, rel_dim, bias=False)
        self.local_adapter = nn.Linear(rel_dim, rel_dim, bias=False)
        self.b_h = nn.Parameter(torch.zeros(rel_dim))
        self.W_o = nn.Linear(rel_dim, llm_dim, bias=True)
        self.gate = nn.Linear(rel_dim, 1)

    def forward(
        self,
        frame_tokens: torch.Tensor,
        v: torch.Tensor,
        s_bar: torch.Tensor,
        r_global: torch.Tensor,
        r_local: torch.Tensor,
        q: torch.Tensor,
        token_mask: torch.Tensor,
        has_neighbor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        B, K, P, _d = frame_tokens.shape
        del B, K, P
        base_h = (
            self.head_token(v)
            + self.head_frame(s_bar)[:, :, None, :]
            + self.head_global(r_global)[:, :, None, :]
            + self.head_question(q)[:, None, None, :]
            + self.b_h
        )
        local_h = self.local_adapter(r_local)
        h = F.silu(base_h + local_h)
        u = self.W_o(h)
        a = torch.sigmoid(self.gate(h))

        sx = rms_fp32(frame_tokens, dim=-1, keepdim=True)
        su = rms_fp32(u, dim=-1, keepdim=True)
        c = torch.where(
            sx > self.rms_eps,
            sx / (sx + su + self.rms_eps),
            torch.zeros_like(sx),
        ).detach()

        valid = token_mask.unsqueeze(-1) & has_neighbor[:, :, None, None]
        delta_x = valid.to(dtype=torch.float32) * a.float() * c * u.float()
        x_tilde = frame_tokens + delta_x.to(dtype=frame_tokens.dtype)
        stats = _collect_stats(delta_x, a, c, sx, token_mask, has_neighbor, self.rms_eps)
        extras = {"base_h": base_h, "local_h": local_h}
        return x_tilde, delta_x, stats, extras


class RelJoinResidual(nn.Module):
    """Full FrameWeave Hybrid v6 / V5.1 stack. Output shape equals input visual tokens."""

    def __init__(
        self,
        llm_dim: int,
        rel_dim: int = REL_DIM,
        residual_eps: float = RESIDUAL_EPS,
        rms_eps: Optional[float] = None,
        time_eps: float = TIME_EPS,
        local_token_chunk: int = 0,
        variant: str = "Hybrid",
        local_enabled: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        flags = variant_flags(variant)
        self.llm_dim = llm_dim
        self.rel_dim = rel_dim
        self.variant = architecture_variant(variant)
        self.time_mode = flags["time_mode"]
        self.use_global = bool(flags["use_global"])
        self.default_local_enabled = bool(flags["local_enabled"]) if local_enabled is None else bool(local_enabled)
        self.time_eps = float(time_eps)
        self.rms_eps = float(residual_eps if rms_eps is None else rms_eps)
        self.local_token_chunk = int(local_token_chunk)
        self.time_dim = TIME_DIM

        self.question_encoder = QuestionEncoder(llm_dim, rel_dim)
        self.frame_pooler = FramePooler(llm_dim, rel_dim)
        self.abs_position = nn.Linear(2, rel_dim, bias=False)
        self.global_relation = GlobalRelation(rel_dim, self.time_dim)
        self.local = LocalRelation(rel_dim)
        self.head = HybridResidualHead(llm_dim, rel_dim, rms_eps=self.rms_eps)
        self.reset_parameters()

        self.last_stats: dict[str, float] = {}
        self.last_delta_x: Optional[torch.Tensor] = None
        self.last_beta: Optional[torch.Tensor] = None
        self.last_gamma: Optional[torch.Tensor] = None
        self.last_alpha: Optional[torch.Tensor] = None
        self.last_r: Optional[torch.Tensor] = None
        self.last_r_local: Optional[torch.Tensor] = None
        self.last_has_neighbor: Optional[torch.Tensor] = None
        self.last_phi: Optional[torch.Tensor] = None
        self.last_tau: Optional[torch.Tensor] = None
        self.last_rho: Optional[torch.Tensor] = None
        self.last_local_logits: Optional[torch.Tensor] = None

    def reset_parameters(self) -> None:
        self.apply(init_xavier)
        nn.init.zeros_(self.abs_position.weight)
        nn.init.zeros_(self.global_relation.A_K.weight)
        nn.init.zeros_(self.global_relation.A_V.weight)
        nn.init.zeros_(self.global_relation.a_b.weight)
        nn.init.zeros_(self.local.w_L)
        nn.init.xavier_uniform_(
            self.head.local_adapter.weight,
            gain=LOCAL_ADAPTER_XAVIER_GAIN,
        )
        nn.init.zeros_(self.head.W_o.weight)
        nn.init.zeros_(self.head.W_o.bias)
        nn.init.zeros_(self.head.gate.weight)
        nn.init.zeros_(self.head.gate.bias)

    @property
    def W_o(self) -> nn.Linear:
        return self.head.W_o

    def encode_question(self, question_hidden: torch.Tensor) -> torch.Tensor:
        return self.question_encoder(question_hidden)

    def iter_local_parameters(self):
        yield from self.local.parameters()
        yield from self.head.local_adapter.parameters()

    def iter_global_parameters(self):
        yield from self.question_encoder.parameters()
        yield from self.frame_pooler.parameters()
        yield from self.abs_position.parameters()
        yield from self.global_relation.parameters()

    def iter_head_parameters(self):
        for name, param in self.head.named_parameters():
            if name.startswith("local_adapter"):
                continue
            yield param

    def iter_local_qk_parameters(self):
        for name in ("local_q", "local_k", "local_frame", "local_global", "local_question", "ln_L", "ln_Q", "ln_K"):
            yield from getattr(self.local, name).parameters()

    def iter_local_value_parameters(self):
        yield from self.local.local_v.parameters()

    def iter_local_message_parameters(self):
        for name in ("A_c", "A_e", "A_d", "A_m", "A_s", "A_r", "A_q"):
            yield from getattr(self.local, name).parameters()
        yield self.local.b_L

    def iter_local_score_parameters(self):
        yield self.local.w_L

    def _zero_buffers(
        self,
        frame_tokens: torch.Tensor,
        frame_mask: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> None:
        B, K, P, _d = frame_tokens.shape
        self.last_stats = _empty_residual_stats()
        self.last_delta_x = torch.zeros_like(frame_tokens)
        self.last_beta = torch.zeros(B, K, K, device=frame_tokens.device, dtype=frame_tokens.dtype)
        self.last_gamma = torch.zeros(B, K, P, K, device=frame_tokens.device, dtype=frame_tokens.dtype)
        self.last_alpha = torch.zeros_like(token_mask, dtype=frame_tokens.dtype)
        self.last_r = torch.zeros(B, K, self.rel_dim, device=frame_tokens.device, dtype=frame_tokens.dtype)
        self.last_r_local = torch.zeros(
            B, K, P, self.rel_dim, device=frame_tokens.device, dtype=frame_tokens.dtype
        )
        self.last_has_neighbor = pair_mask_from_frames(frame_mask).any(dim=-1)
        self.last_phi = None
        self.last_tau = None
        self.last_rho = None
        self.last_local_logits = None

    def _time_features(
        self,
        timestamps: torch.Tensor,
        frame_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tau, rho = normalize_time_and_rank(timestamps, frame_mask, eps=self.time_eps)
        tau = tau.to(dtype=dtype)
        rho = rho.to(dtype=dtype)
        if self.time_mode == TIME_ZERO:
            tau = torch.zeros_like(tau)
            rho = torch.zeros_like(rho)
        phi = relative_time_features(tau, rho)
        return tau, rho, phi

    def forward(
        self,
        frame_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        frame_mask: torch.Tensor,
        timestamps: torch.Tensor,
        question_hidden: torch.Tensor,
        *,
        eval_mode: str = EVAL_FULL,
        local_enabled: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        mode = (eval_mode or EVAL_FULL).strip().lower()
        if mode not in EVAL_MODES:
            raise ValueError(f"Unknown eval_mode {eval_mode!r}. Expected {EVAL_MODES}")

        token_mask = token_mask.to(dtype=torch.bool)
        frame_mask = frame_mask.to(dtype=torch.bool) & token_mask.any(dim=-1)

        if mode == EVAL_IDENTITY:
            self._zero_buffers(frame_tokens, frame_mask, token_mask)
            return frame_tokens

        use_local = self.default_local_enabled if local_enabled is None else bool(local_enabled)
        pair_mask = pair_mask_from_frames(frame_mask)
        has_neighbor = pair_mask.any(dim=-1)

        q = self.encode_question(question_hidden)
        s, alpha, v = self.frame_pooler(frame_tokens, token_mask, q, frame_mask)
        tau, rho, phi = self._time_features(timestamps.to(dtype=s.dtype), frame_mask, s.dtype)
        abs_pos = self.abs_position(torch.stack([tau, rho], dim=-1))
        abs_pos = abs_pos * frame_mask.unsqueeze(-1).to(dtype=abs_pos.dtype)
        s_bar = s + abs_pos

        B, K, P, _d = frame_tokens.shape
        time_key_ratio = 0.0
        time_value_ratio = 0.0
        if self.use_global:
            global_logits, beta, r_global, has_neighbor = self.global_relation(
                s_bar, q, phi, pair_mask
            )
            content_m, temporal_m = self.global_relation.logit_parts(s_bar, q, phi, pair_mask)
            time_key_ratio, time_value_ratio = self.global_relation.time_ratios(
                s_bar, phi, pair_mask, self.rms_eps
            )
        else:
            global_logits = torch.zeros(B, K, K, device=s.device, dtype=s.dtype)
            beta = torch.zeros(B, K, K, device=s.device, dtype=s.dtype)
            r_global = torch.zeros_like(s_bar)
            content_m, temporal_m = 0.0, 0.0

        token_match_entropy = 0.0
        normalized_token_match_entropy = 0.0
        local_logit_abs_mean = 0.0
        gamma: Optional[torch.Tensor] = None
        ell: Optional[torch.Tensor] = None
        if use_local:
            local_out = self.local(
                v=v,
                s_bar=s_bar,
                r_global=r_global,
                q=q,
                global_logits=global_logits,
                token_mask=token_mask,
                pair_mask=pair_mask,
                chunk_p=self.local_token_chunk,
            )
            r_local = local_out["r_local"]
            gamma = local_out["gamma"]
            ell = local_out["local_logits"]
            token_match_entropy = float(local_out["token_match_entropy"])
            normalized_token_match_entropy = float(local_out["normalized_token_match_entropy"])
            local_logit_abs_mean = float(local_out["local_logit_abs_mean"])
        else:
            r_local = torch.zeros(
                B, K, P, self.rel_dim, device=frame_tokens.device, dtype=s.dtype
            )

        x_tilde, delta_x, stats, extras = self.head(
            frame_tokens, v, s_bar, r_global, r_local, q, token_mask, has_neighbor
        )
        if not bool(torch.isfinite(delta_x).all()):
            raise RuntimeError("Non-finite residual")
        stats = dict(stats)
        stats["content_logit_abs_mean"] = content_m
        stats["temporal_logit_abs_mean"] = temporal_m
        stats["temporal_over_content"] = (
            temporal_m / content_m if content_m > 0 else 0.0
        )
        stats["token_match_entropy"] = token_match_entropy
        stats["normalized_token_match_entropy"] = normalized_token_match_entropy
        stats["local_logit_abs_mean"] = local_logit_abs_mean
        stats["time_key_ratio"] = time_key_ratio
        stats["time_value_ratio"] = time_value_ratio

        if bool(frame_mask.any()):
            ap = rms_fp32(abs_pos.detach()[frame_mask], dim=-1, keepdim=False).mean()
            ss = rms_fp32(s.detach()[frame_mask], dim=-1, keepdim=False).mean()
            stats["absolute_position_ratio"] = float(ap / (ss + self.rms_eps)) if float(ss) > 0 else 0.0
        else:
            stats["absolute_position_ratio"] = 0.0

        if self.use_global and bool(has_neighbor.any()):
            p = beta.detach().float().clamp_min(1e-12)
            ent = -(p * p.log()).sum(dim=-1)
            stats["global_entropy"] = float(ent[has_neighbor].mean())
            stats["relation_entropy"] = stats["global_entropy"]
        else:
            stats["global_entropy"] = 0.0
            stats["relation_entropy"] = 0.0

        if gamma is not None and bool(has_neighbor.any()):
            p_g = gamma.detach().float().clamp_min(1e-12)
            ent_g = -(p_g * p_g.log()).sum(dim=-1)
            valid_g = token_mask & has_neighbor.unsqueeze(-1)
            stats["local_frame_entropy"] = (
                float(ent_g[valid_g].mean()) if bool(valid_g.any()) else 0.0
            )
        else:
            stats["local_frame_entropy"] = 0.0

        local_h = extras["local_h"]
        base_h = extras["base_h"]
        valid = token_mask & has_neighbor.unsqueeze(-1)
        if bool(valid.any()):
            loc_rms = float(rms_fp32(local_h.detach()[valid], dim=-1, keepdim=False).mean())
            base_rms_h = float(rms_fp32(base_h.detach()[valid], dim=-1, keepdim=False).mean())
            stats["local_ratio"] = loc_rms / (base_rms_h + self.rms_eps)
        else:
            stats["local_ratio"] = 0.0

        self.last_stats = stats
        self.last_delta_x = delta_x.detach()
        self.last_beta = beta.detach()
        self.last_gamma = None if gamma is None else gamma.detach()
        self.last_alpha = alpha.detach()
        self.last_r = r_global.detach()
        self.last_r_local = r_local.detach()
        self.last_has_neighbor = has_neighbor.detach()
        self.last_phi = phi.detach()
        self.last_tau = tau.detach()
        self.last_rho = rho.detach()
        self.last_local_logits = None if ell is None else ell.detach()
        return x_tilde


def _collect_stats(
    delta_x: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    sx: torch.Tensor,
    token_mask: torch.Tensor,
    has_neighbor: torch.Tensor,
    rms_eps: float,
) -> dict[str, float]:
    valid = token_mask & has_neighbor.unsqueeze(-1)
    n = valid.to(dtype=torch.float32).sum().clamp_min(1.0)
    d = delta_x.detach().float()
    residual_rms = float(torch.sqrt((d.pow(2).sum() / (n * d.shape[-1])).clamp_min(0)))
    a_d = a.detach().squeeze(-1)
    c_d = c.detach().squeeze(-1)
    a_valid = a_d[valid] if valid.any() else a_d.new_zeros((0,))
    c_valid = c_d[valid] if valid.any() else c_d.new_zeros((0,))
    gate_mean = float(a_valid.float().mean()) if a_valid.numel() else 0.0
    max_g = float(a_valid.float().max()) if a_valid.numel() else 0.0
    calib_mean = float(c_valid.float().mean()) if c_valid.numel() else 0.0
    rho = torch.zeros(valid.shape, device=delta_x.device, dtype=torch.float32)
    sx_flat = sx.detach().squeeze(-1)
    su_d = rms_fp32(delta_x.detach(), dim=-1, keepdim=False)
    nz = valid & (sx_flat > rms_eps)
    rho[nz] = su_d[nz] / sx_flat[nz].clamp_min(rms_eps)
    rho_max = float(rho[nz].max()) if nz.any() else 0.0
    if a_valid.numel():
        a_f = a_valid.float()
        gate_p50 = float(a_f.quantile(0.5))
        gate_p95 = float(a_f.quantile(0.95))
    else:
        gate_p50 = 0.0
        gate_p95 = 0.0
    sx_valid = sx_flat[valid]
    base_rms = float(sx_valid.float().mean()) if sx_valid.numel() else 0.0
    residual_over = residual_rms / base_rms if base_rms > 0 else 0.0
    out = _empty_residual_stats()
    out.update(
        {
            "residual_rms": residual_rms,
            "delta_rms": residual_rms,
            "max_g": max_g,
            "gate_mean": gate_mean,
            "gate_p50": gate_p50,
            "gate_p95": gate_p95,
            "calib_mean": calib_mean,
            "rho_max": rho_max,
            "base_rms": base_rms,
            "residual_over_base_rms": residual_over,
        }
    )
    return out


def build_joining_module(
    llm_dim: int,
    variant: str = "Hybrid",
    rel_dim: int = REL_DIM,
    residual_eps: float = RESIDUAL_EPS,
    rms_eps: Optional[float] = None,
    time_eps: float = TIME_EPS,
    local_token_chunk: int = 0,
    local_enabled: Optional[bool] = None,
    **kwargs,
) -> nn.Module:
    del kwargs
    eps = float(residual_eps if rms_eps is None else rms_eps)
    return RelJoinResidual(
        llm_dim,
        rel_dim=rel_dim,
        residual_eps=eps,
        rms_eps=eps,
        time_eps=time_eps,
        local_token_chunk=local_token_chunk,
        variant=variant,
        local_enabled=local_enabled,
    )


def count_trainable(module: nn.Module) -> tuple[int, list[str]]:
    names = [n for n, p in module.named_parameters() if p.requires_grad]
    n = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return n, names
