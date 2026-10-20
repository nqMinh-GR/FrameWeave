"""Direction controls for a captured FrameWeave residual.

ΔX is the tensor already produced after the gate, RMS calibration, and mask.
These helpers only rescale that tensor or replace it with a same-norm vector.
They do not call the residual head and do not recompute RMS calibration.
"""
from __future__ import annotations

import hashlib
import math
from typing import Optional

import torch

COND_VANILLA = "vanilla"
COND_LEARNED = "learned"
COND_DOUBLE = "double"
COND_REVERSE = "reverse"
COND_RANDOM = "random"
COND_PARALLEL = "parallel"

CONDITIONS = (
    COND_VANILLA,
    COND_LEARNED,
    COND_DOUBLE,
    COND_REVERSE,
    COND_RANDOM,
    COND_PARALLEL,
)

SEED = 42
_NORM_ATOL = 1e-4


def jobs_for_sample(n_seeds: int = 1) -> list[dict]:
    """One captured ΔX, then these additions. Random uses ``n_seeds`` draws."""
    n_seeds = int(n_seeds)
    if n_seeds < 1:
        raise ValueError(f"n_random_seeds must be >= 1, got {n_seeds}")
    jobs = [
        {"condition": COND_VANILLA, "kind": "scale", "lambda_scale": 0.0, "random_seed": -1},
        {"condition": COND_LEARNED, "kind": "scale", "lambda_scale": 1.0, "random_seed": -1},
        {"condition": COND_DOUBLE, "kind": "scale", "lambda_scale": 2.0, "random_seed": -1},
        {"condition": COND_REVERSE, "kind": "scale", "lambda_scale": -1.0, "random_seed": -1},
    ]
    for seed in range(n_seeds):
        jobs.append(
            {
                "condition": COND_RANDOM,
                "kind": "random",
                "lambda_scale": 1.0,
                "random_seed": int(seed),
            }
        )
    jobs.append(
        {
            "condition": COND_PARALLEL,
            "kind": "parallel",
            "lambda_scale": 1.0,
            "random_seed": -1,
        }
    )
    return jobs


def seed_from_ids(*parts: object, base: int = SEED) -> int:
    blob = ":".join(str(p) for p in (base, *parts)).encode("utf-8")
    digest = hashlib.sha256(blob).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def make_generator(*parts: object, base: int = SEED) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed_from_ids(*parts, base=base))
    return generator


def _token_mask(token_mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    mask = token_mask.to(device=ref.device, dtype=torch.float32)
    if mask.shape != ref.shape[:-1]:
        raise AssertionError(
            f"token_mask shape {tuple(mask.shape)} != residual leading shape {tuple(ref.shape[:-1])}"
        )
    return mask.unsqueeze(-1)


def scale_captured_delta(
    delta_x: torch.Tensor,
    token_mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Multiply the captured ΔX. ``scale=2`` is exactly ``2 ΔX``, not a new RMS pass."""
    delta = delta_x.detach().float()
    out = delta * float(scale)
    return out * _token_mask(token_mask, delta)


def isotropic_matched_residual(
    delta_x: torch.Tensor,
    token_mask: torch.Tensor,
    generator: torch.Generator,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """R_ip = ||Δx_ip||_2 * z_ip / ||z_ip||_2. R_ip = 0 when Δx_ip = 0."""
    delta = delta_x.detach().float().cpu()
    mag = delta.norm(dim=-1, keepdim=True)
    if noise is None:
        drawn = torch.randn(delta.shape, dtype=torch.float32, generator=generator)
    else:
        drawn = noise.detach().float().cpu()
        if tuple(drawn.shape) != tuple(delta.shape):
            raise AssertionError(
                f"noise shape {tuple(drawn.shape)} != ΔX {tuple(delta.shape)}"
            )
    need = (drawn.norm(dim=-1, keepdim=True) == 0) & (mag > 0)
    for _ in range(8):
        if not bool(need.any()):
            break
        fresh = torch.randn(delta.shape, dtype=torch.float32, generator=generator)
        drawn = torch.where(need, fresh, drawn)
        need = (drawn.norm(dim=-1, keepdim=True) == 0) & (mag > 0)
    zn = drawn.norm(dim=-1, keepdim=True)
    direction = torch.zeros_like(drawn)
    positive = zn > 0
    direction = torch.where(positive, drawn / zn, direction)
    out = direction * mag
    out = torch.where(mag > 0, out, torch.zeros_like(out))
    out = out * _token_mask(token_mask, delta).cpu()
    return out.to(device=delta_x.device)


def parallel_matched_residual(
    frame_tokens: torch.Tensor,
    delta_x: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """D^parallel has ||Δx|| and the direction of x. Zero if either norm is zero."""
    base = frame_tokens.detach().float()
    delta = delta_x.detach().float()
    if tuple(base.shape) != tuple(delta.shape):
        raise AssertionError(
            f"X shape {tuple(base.shape)} != ΔX {tuple(delta.shape)}"
        )
    mag = delta.norm(dim=-1, keepdim=True)
    base_norm = base.norm(dim=-1, keepdim=True)
    direction = torch.zeros_like(base)
    positive = base_norm > 0
    direction = torch.where(positive, base / base_norm, direction)
    out = direction * mag
    keep = (mag > 0) & (base_norm > 0)
    out = torch.where(keep, out, torch.zeros_like(out))
    return out * _token_mask(token_mask, out)


def build_injection(
    delta_x: torch.Tensor,
    frame_tokens: torch.Tensor,
    token_mask: torch.Tensor,
    spec: dict,
    sample_id: str,
) -> torch.Tensor:
    """Residual added to the original visual tokens. Built only from captured ΔX."""
    kind = str(spec["kind"])
    if kind == "scale":
        return scale_captured_delta(delta_x, token_mask, float(spec["lambda_scale"]))
    if kind == "random":
        generator = make_generator(sample_id, "random", int(spec["random_seed"]))
        return isotropic_matched_residual(delta_x, token_mask, generator)
    if kind == "parallel":
        return parallel_matched_residual(frame_tokens, delta_x, token_mask)
    raise ValueError(f"Unknown residual-direction kind {kind!r}")


def _finite_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(values.float().mean())


def injection_metrics(
    delta_x: torch.Tensor,
    frame_tokens: torch.Tensor,
    token_mask: torch.Tensor,
    injected: torch.Tensor,
) -> dict[str, float]:
    """Per-token norm ratio and cosine, averaged over tokens where the reference is nonzero."""
    delta = delta_x.detach().float()
    base = frame_tokens.detach().float()
    add = injected.detach().float()
    valid = token_mask.to(dtype=torch.bool)
    mag = delta.norm(dim=-1)
    base_norm = base.norm(dim=-1)
    add_norm = add.norm(dim=-1)
    nonzero = valid & (mag > 0)
    both_x = nonzero & (base_norm > 0)
    matched = nonzero & (add_norm > 0)
    if bool(matched.any()):
        ratio_mean = float((add_norm[matched] / mag[matched]).mean())
    elif bool(nonzero.any()) and float(add.abs().sum()) == 0.0:
        ratio_mean = 0.0
    else:
        ratio_mean = float("nan")
    norm_error = (add_norm - mag).abs()
    norm_error = norm_error[valid]
    dot_delta = (add * delta).sum(dim=-1)
    dot_x = (add * base).sum(dim=-1)
    cos_delta = dot_delta[nonzero] / (add_norm[nonzero] * mag[nonzero]).clamp_min(1e-12)
    cos_delta = cos_delta[add_norm[nonzero] > 0]
    cos_x = dot_x[both_x] / (add_norm[both_x] * base_norm[both_x]).clamp_min(1e-12)
    cos_x = cos_x[add_norm[both_x] > 0]
    return {
        "norm_ratio_vs_delta": ratio_mean,
        "cosine_vs_delta": _finite_mean(cos_delta),
        "cosine_vs_x": _finite_mean(cos_x),
        "max_abs_norm_error": float(norm_error.max()) if norm_error.numel() else 0.0,
        "n_nonzero_delta": float(nonzero.sum()),
    }


def _intended_norm_gap(
    injected: torch.Tensor,
    delta_x: torch.Tensor,
    frame_tokens: torch.Tensor,
    valid: torch.Tensor,
    spec: dict,
) -> float:
    """Max | ||injection|| - target || over valid tokens. Target is |λ| ||Δx||, or 0 for D^parallel when ||x||=0."""
    mag = delta_x.detach().float().norm(dim=-1)
    got = injected.detach().float().norm(dim=-1)
    if str(spec["kind"]) == "parallel":
        base_norm = frame_tokens.detach().float().norm(dim=-1)
        target = torch.where(base_norm > 0, mag, torch.zeros_like(mag))
    else:
        target = mag * abs(float(spec["lambda_scale"]))
    gap = (got - target).abs()[valid]
    if gap.numel() == 0:
        return 0.0
    return float(gap.max())


def _close(actual: float, expected: float, atol: float = _NORM_ATOL) -> bool:
    if not math.isfinite(actual) or not math.isfinite(expected):
        return False
    return abs(actual - expected) <= atol


def assert_injection_matches_spec(
    delta_x: torch.Tensor,
    frame_tokens: torch.Tensor,
    token_mask: torch.Tensor,
    injected: torch.Tensor,
    spec: dict,
) -> dict[str, float]:
    """Fail if scaling re-entered RMS calibration or a control broke the norm/direction."""
    delta = delta_x.detach().float()
    mask = _token_mask(token_mask, delta)
    add = injected.detach().float()
    valid = token_mask.to(dtype=torch.bool)
    masked_delta = delta * mask
    invalid = ~valid
    if bool(invalid.any()) and not torch.equal(add[invalid], torch.zeros_like(add[invalid])):
        raise AssertionError("injection leaked into padding / masked tokens")
    zero_delta = valid & (delta.norm(dim=-1) == 0)
    if bool(zero_delta.any()) and float(add[zero_delta].abs().sum()) != 0.0:
        raise AssertionError("Δx=0 must produce a zero correction")

    kind = str(spec["kind"])
    condition = str(spec["condition"])
    if kind == "scale":
        expected = masked_delta * float(spec["lambda_scale"])
        if not torch.equal(add, expected):
            raise AssertionError(
                f"{condition} is not exactly λΔX (λ={spec['lambda_scale']}). "
                "RMS calibration must not run again after scaling."
            )
    elif kind == "parallel":
        base = frame_tokens.detach().float()
        zero_x = valid & (base.norm(dim=-1) == 0)
        if bool(zero_x.any()) and float(add[zero_x].abs().sum()) != 0.0:
            raise AssertionError("||x||=0 must produce a zero parallel correction")
    metrics = injection_metrics(delta, frame_tokens, token_mask, add)
    metrics["max_abs_norm_error"] = _intended_norm_gap(add, delta, frame_tokens, valid, spec)
    if metrics["n_nonzero_delta"] <= 0:
        if float(add.abs().sum()) != 0.0:
            raise AssertionError(f"{condition} must be zero when ΔX is zero")
        return metrics

    if condition == COND_VANILLA:
        if float(add.abs().sum()) != 0.0:
            raise AssertionError("vanilla injection must be 0")
    elif condition in {COND_LEARNED, COND_DOUBLE, COND_REVERSE}:
        expected_ratio = abs(float(spec["lambda_scale"]))
        expected_cos = 1.0 if float(spec["lambda_scale"]) > 0 else -1.0
        if not _close(metrics["norm_ratio_vs_delta"], expected_ratio):
            raise AssertionError(
                f"{condition} norm ratio {metrics['norm_ratio_vs_delta']} != {expected_ratio}"
            )
        if not _close(metrics["cosine_vs_delta"], expected_cos, atol=1e-5):
            raise AssertionError(
                f"{condition} cosine {metrics['cosine_vs_delta']} != {expected_cos}"
            )
    elif condition == COND_RANDOM:
        if not _close(metrics["norm_ratio_vs_delta"], 1.0):
            raise AssertionError(
                f"random norm ratio {metrics['norm_ratio_vs_delta']} != 1"
            )
    elif condition == COND_PARALLEL:
        base = frame_tokens.detach().float()
        base_norm = base.norm(dim=-1)
        comparable = valid & (masked_delta.norm(dim=-1) > 0) & (base_norm > 0)
        if bool(comparable.any()):
            ratio = add.norm(dim=-1)[comparable] / masked_delta.norm(dim=-1)[comparable]
            dot = (add * base).sum(dim=-1)[comparable]
            cosine = dot / (
                add.norm(dim=-1)[comparable] * base_norm[comparable]
            )
            if not _close(float(ratio.mean()), 1.0):
                raise AssertionError(f"parallel norm ratio {float(ratio.mean())} != 1")
            if not _close(float(cosine.mean()), 1.0, atol=1e-5):
                raise AssertionError(f"parallel cosine vs x {float(cosine.mean())} != 1")
    else:
        raise AssertionError(f"Unknown condition {condition!r}")
    return metrics
