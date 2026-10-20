"""Debug collector for FrameWeave Hybrid tensors.

The session is opt-in. With no session, LocalRelation does not copy pi and
ordinary forwards keep their previous outputs. Copies are detached CPU tensors.
"""
from __future__ import annotations

from typing import Optional

import torch

from ..reljoin_modules import register_pi_chunk_sink, rms_fp32, unregister_pi_chunk_sink

_ACTIVE: list["FrameVizSession"] = []


def active_session() -> Optional["FrameVizSession"]:
    return _ACTIVE[-1] if _ACTIVE else None


def note_delivered_tokens(
    *,
    frame_tokens: torch.Tensor,
    delivered: torch.Tensor,
    token_mask: torch.Tensor,
    frame_mask: torch.Tensor,
    module,
    timestamps,
) -> None:
    """Record the cast that is scattered back into the frozen decoder."""
    sess = active_session()
    if sess is None:
        return
    sess.record_delivered(
        frame_tokens=frame_tokens,
        delivered=delivered,
        token_mask=token_mask,
        frame_mask=frame_mask,
        module=module,
        timestamps=timestamps,
    )


class FrameVizSession:
    """Collect scalar maps and, optionally, pi rows for a few query tokens."""

    def __init__(
        self,
        *,
        query_indices: Optional[list[tuple[int, int]]] = None,
        collect_pi: bool = False,
        rms_eps: float = 1e-8,
        keep_updated: bool = True,
    ):
        self.query_indices = [(int(i), int(p)) for i, p in (query_indices or [])]
        self.collect_pi = bool(collect_pi)
        self.rms_eps = float(rms_eps)
        self.keep_updated = bool(keep_updated)
        self._pi_rows: dict[tuple[int, int], torch.Tensor] = {}
        self._bundle: Optional[dict] = None
        self._sink = self._on_pi_chunk

    def __enter__(self) -> "FrameVizSession":
        self._pi_rows = {}
        self._bundle = None
        _ACTIVE.append(self)
        if self.collect_pi:
            register_pi_chunk_sink(self._sink)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.collect_pi:
            unregister_pi_chunk_sink(self._sink)
        if self in _ACTIVE:
            _ACTIVE.remove(self)
        return False

    def _on_pi_chunk(self, pi: torch.Tensor, p0: int) -> None:
        if not self.query_indices:
            return
        p0 = int(p0)
        n_recv = int(pi.shape[1])
        chunk = int(pi.shape[2])
        for frame_i, token_p in self.query_indices:
            if frame_i < 0 or frame_i >= n_recv:
                continue
            local = int(token_p) - p0
            if local < 0 or local >= chunk:
                continue
            # [B, K, P] target frame x target token. Detach before any storage.
            row = pi[:, frame_i, local].detach().to(device="cpu", dtype=torch.float32).clone()
            self._pi_rows[(frame_i, token_p)] = row

    def record_delivered(
        self,
        *,
        frame_tokens: torch.Tensor,
        delivered: torch.Tensor,
        token_mask: torch.Tensor,
        frame_mask: torch.Tensor,
        module,
        timestamps,
    ) -> None:
        original = frame_tokens.detach().float().cpu()
        delivered_f = delivered.detach().float().cpu()
        effective = delivered_f - original
        token_mask_b = token_mask.detach().to(device="cpu", dtype=torch.bool)
        frame_mask_b = frame_mask.detach().to(device="cpu", dtype=torch.bool)
        input_rms = rms_fp32(original, dim=-1, keepdim=False)
        delta_rms = rms_fp32(effective, dim=-1, keepdim=False)
        eta = delta_rms / (input_rms + self.rms_eps)
        eta = eta.masked_fill(~token_mask_b, 0)

        module_delta = getattr(module, "last_delta_x", None) if module is not None else None
        if module_delta is None:
            module_delta_cpu = torch.zeros_like(effective)
            module_delta_rms = torch.zeros_like(delta_rms)
        else:
            module_delta_cpu = module_delta.detach().float().cpu()
            module_delta_rms = rms_fp32(module_delta_cpu, dim=-1, keepdim=False)
            module_delta_rms = module_delta_rms.masked_fill(~token_mask_b, 0)

        denom = original.norm(dim=-1) * effective.norm(dim=-1)
        cosine = (original * effective).sum(dim=-1) / (denom + self.rms_eps)
        cosine = cosine.masked_fill(~token_mask_b, 0)

        def _cpu(name: str):
            tensor = getattr(module, name, None) if module is not None else None
            if tensor is None:
                return None
            return tensor.detach().float().cpu()

        alpha = _cpu("last_alpha")
        beta = _cpu("last_beta")
        gamma = _cpu("last_gamma")
        if alpha is None:
            alpha = torch.zeros(original.shape[:-1], dtype=torch.float32)
        if beta is None:
            k = int(original.shape[1])
            beta = torch.zeros(original.shape[0], k, k, dtype=torch.float32)
        stats = {}
        if module is not None:
            raw_stats = getattr(module, "last_stats", None) or {}
            for key, value in raw_stats.items():
                if isinstance(value, (int, float)):
                    stats[key] = float(value)
        dtype_gap = float((effective - module_delta_cpu).abs().max()) if module_delta_cpu.numel() else 0.0
        self._bundle = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "input_rms": input_rms,
            "delta_rms": delta_rms,
            "module_delta_rms": module_delta_rms,
            "eta": eta,
            "cosine_x_delta": cosine,
            "token_mask": token_mask_b,
            "frame_mask": frame_mask_b,
            "effective_delta": effective if self.keep_updated else None,
            "original_tokens": original if self.keep_updated else None,
            "timestamps": None if timestamps is None else [float(t) for t in timestamps],
            "module_stats": stats,
            "dtype_gap_max_abs": dtype_gap,
            "eta_source": "effective_delivered_correction",
            "module_delta_source": "HybridResidualHead.after_mask_gate_rms_before_dtype_cast",
        }

    def export(self) -> dict:
        if self._bundle is None:
            raise RuntimeError("FrameVizSession saw no delivered residual. The hook did not run.")
        bundle = dict(self._bundle)
        alpha = bundle["alpha"]
        bsz, k, pmax = int(alpha.shape[0]), int(alpha.shape[1]), int(alpha.shape[2])
        queries = list(self.query_indices)
        s = len(queries)
        pi = torch.zeros(bsz, s, k, pmax, dtype=torch.float32)
        captured = []
        for idx, key in enumerate(queries):
            row = self._pi_rows.get(key)
            ok = row is not None and tuple(row.shape) == (bsz, k, pmax)
            captured.append(bool(ok))
            if ok:
                pi[:, idx] = row
        gamma_full = bundle.get("gamma")
        gamma_selected = torch.zeros(bsz, s, k, dtype=torch.float32)
        if gamma_full is not None:
            for idx, (frame_i, token_p) in enumerate(queries):
                if 0 <= frame_i < k and 0 <= token_p < pmax:
                    gamma_selected[:, idx] = gamma_full[:, frame_i, token_p, :]
        bundle["pi_selected"] = pi
        bundle["gamma_selected"] = gamma_selected
        bundle["query_indices"] = torch.tensor(queries, dtype=torch.int64).reshape(s, 2)
        bundle["query_captured"] = captured
        bundle["collect_pi"] = self.collect_pi
        return bundle
