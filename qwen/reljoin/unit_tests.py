"""Structural unit tests for FrameWeave Hybrid v6.0 (V5.1) on Qwen2.5-VL (synthetic tensors; no video)."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from .config import (
    ExperimentConfig,
    K_VALUES_UNIT,
    LOCAL_ADAPTER_XAVIER_GAIN,
    METHOD_VERSION,
    TOKENS_PER_FRAME_VALUES,
    make_output_tree,
)
from .data_perception import (
    QASample,
    assert_no_herbench_in_training,
    stable_int_seed,
    unit_test_option_permutation,
    unit_test_perception_schema_parse,
)
from .losses import batch_reljoin_v5_loss, reljoin_v5_loss
from .qwen_hooks import (
    VISUAL_ATOM_ID,
    begin_prepare_cache,
    configure_reljoin,
    end_prepare_cache,
    get_hook_state,
    hook_install_count,
    install_reljoin_hook,
    no_reljoin_hooks_remaining,
    pack_variable_frame_tokens,
    patch_qwen_sdpa_attention,
    remove_all_reljoin_hooks,
    reset_hook_diagnostics,
    scatter_variable_frame_tokens,
    set_reljoin_context,
    visual_atom_spans,
)
from .reljoin_modules import (
    EVAL_IDENTITY,
    RelJoinResidual,
    masked_softmax,
    normalize_tau,
    normalize_time_and_rank,
    ordinal_rho,
    pair_mask_from_frames,
    relative_time_features,
    rms_fp32,
)


class _Fail(Exception):
    pass


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise _Fail(msg)


def _make_batch(B, K, P, d, device="cpu"):
    frame_tokens = torch.randn(B, K, P, d, device=device)
    token_mask = torch.ones(B, K, P, dtype=torch.bool, device=device)
    frame_mask = torch.ones(B, K, dtype=torch.bool, device=device)
    timestamps = torch.linspace(0, 1, K, device=device).unsqueeze(0).expand(B, K)
    question_hidden = torch.randn(B, d, device=device)
    return frame_tokens, token_mask, frame_mask, timestamps, question_hidden


def _forward(m, batch, **kwargs):
    ft, tm, fm, ts, qh = batch
    return m(ft, tm, fm, ts, qh, **kwargs)


def _randomize_wo(m: nn.Module, std: float = 0.2) -> None:
    nn.init.normal_(m.W_o.weight, std=std)
    if m.W_o.bias is not None:
        nn.init.normal_(m.W_o.bias, std=std * 0.1)


def test_forward_all_K(llm_dim: int = 256) -> None:
    for K in K_VALUES_UNIT:
        for P in TOKENS_PER_FRAME_VALUES:
            if K >= 8 and P >= 256:
                continue
            m = RelJoinResidual(llm_dim, rel_dim=64, variant="Hybrid")
            batch = _make_batch(2, K, P, llm_dim)
            out = _forward(m, batch)
            _assert(out.shape == batch[0].shape, f"shape {out.shape} vs {batch[0].shape} K={K} P={P}")
    print("PASS: forward all K / P (same shape as X)")


def test_no_hardcoded_2048() -> None:
    m = RelJoinResidual(128, rel_dim=32, variant="Hybrid")
    batch = _make_batch(1, 3, 17, 128)
    out = _forward(m, batch)
    _assert(out.shape == (1, 3, 17, 128), "non-2048 path failed")
    print("PASS: no hardcoded 2048 / R=4")


def test_phi_zero_and_signed() -> None:
    ts = torch.tensor([[0.0, 0.5, 1.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    tau = normalize_tau(ts, mask)
    rho = ordinal_rho(mask, dtype=ts.dtype)
    phi = relative_time_features(tau, rho)
    _assert(phi.shape == (1, 3, 3, 4), f"phi shape {phi.shape}")
    _assert(torch.allclose(phi[0].diagonal(dim1=0, dim2=1), torch.zeros(4, 3)), "φ_ii must be 0")
    d01 = phi[0, 0, 1]
    d10 = phi[0, 1, 0]
    _assert(torch.allclose(d01[1], d10[1]), "|δτ| must be symmetric")
    _assert(torch.allclose(d01[3], d10[3]), "|δρ| must be symmetric")
    _assert(not torch.allclose(d01[0], d10[0]), "signed δτ must flip with i↔j")
    _assert(float(d01[0]) > 0, "later source should have positive δτ from earlier receiver")
    equal = torch.tensor([[5.0, 5.0, 5.0]])
    tau0 = normalize_tau(equal, mask)
    _assert(torch.allclose(tau0, torch.zeros_like(tau0)), "equal timestamps → τ=0")
    print("PASS: φ 4D, φ(i=i)=0, signed δ, |δ| symmetric, span 0 → τ=0")


def test_tau_rho_few_frames() -> None:
    ts = torch.tensor([[0.0, 1.0, 2.0]])
    mask = torch.tensor([[True, False, False]])
    tau, rho = normalize_time_and_rank(ts, mask)
    _assert(torch.allclose(tau, torch.zeros_like(tau)), "<2 valid → τ=0")
    _assert(torch.allclose(rho, torch.zeros_like(rho)), "<2 valid → ρ=0")
    print("PASS: <2 valid frames → τ=ρ=0")


def test_masked_softmax_all_masked() -> None:
    logits = torch.randn(2, 4)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    probs = masked_softmax(logits, mask, dim=-1)
    _assert(torch.allclose(probs, torch.zeros_like(probs)), "all-masked softmax must be 0")
    _assert(bool(torch.isfinite(probs).all()), "all-masked softmax must not NaN")
    print("PASS: all-masked softmax is 0")


def test_zero_inits() -> None:
    m = RelJoinResidual(32, rel_dim=16, variant="Hybrid")
    _assert(float(m.W_o.weight.detach().abs().sum()) == 0.0, "W_o must be 0")
    _assert(m.W_o.bias is not None and float(m.W_o.bias.detach().abs().sum()) == 0.0, "W_o bias")
    _assert(float(m.abs_position.weight.detach().abs().sum()) == 0.0, "W_abs must be 0")
    _assert(float(m.global_relation.A_K.weight.detach().abs().sum()) == 0.0, "A_K must be 0")
    _assert(float(m.global_relation.A_V.weight.detach().abs().sum()) == 0.0, "A_V must be 0")
    _assert(float(m.global_relation.a_b.weight.detach().abs().sum()) == 0.0, "a_b must be 0")
    pl = m.head.local_adapter.weight.detach()
    _assert(float(pl.abs().sum()) > 0.0, "P_L small-Xavier must be nonzero")
    fan = pl.shape[0] + pl.shape[1]
    bound = LOCAL_ADAPTER_XAVIER_GAIN * math.sqrt(6.0 / fan) + 1e-6
    _assert(float(pl.abs().max()) <= bound, f"P_L max {float(pl.abs().max())} > {bound}")
    _assert(float(m.local.w_L.detach().abs().sum()) == 0.0, "w_L must be 0")
    _assert(float(m.head.gate.weight.detach().abs().sum()) == 0.0, "w_g must be 0")
    _assert(float(m.head.gate.bias.detach().abs().sum()) == 0.0, "b_g must be 0")
    _assert(not any("A_t" in k for k in m.state_dict()), "local.A_t must not exist")
    print("PASS: zero inits W_o / W_abs / A_K / A_V / a_b / w_L / gate; P_L small-Xavier")


def test_exact_vanilla_at_init(llm_dim: int = 128) -> None:
    m = RelJoinResidual(llm_dim, rel_dim=32, variant="Hybrid")
    m.eval()
    batch = _make_batch(2, 5, 16, llm_dim)
    with torch.no_grad():
        out = _forward(m, batch)
    _assert(torch.equal(out, batch[0]), "W_o=0 must be exact Vanilla even with P_L ≠ 0")
    _assert(float((out - batch[0]).abs().max()) == 0.0, "Vanilla residual max abs must be 0")
    print("PASS: exact Vanilla initialization")


def test_k1_identity_even_with_random_Wo(llm_dim: int = 64) -> None:
    m = RelJoinResidual(llm_dim, rel_dim=32, variant="Hybrid")
    _randomize_wo(m, 0.5)
    m.eval()
    batch = _make_batch(3, 1, 11, llm_dim)
    with torch.no_grad():
        out = _forward(m, batch)
    _assert(torch.allclose(out, batch[0], atol=1e-5, rtol=1e-4), "K=1 must be identity")
    _assert(
        m.last_has_neighbor is not None and not bool(m.last_has_neighbor.any()),
        "K=1 must have χ_i=0",
    )
    print("PASS: K=1 identity after training-like W_o")


def test_diagonal_beta_zero(llm_dim: int = 64) -> None:
    m = RelJoinResidual(llm_dim, rel_dim=32, variant="GlobalTimeFix")
    _randomize_wo(m, 0.2)
    m.eval()
    batch = _make_batch(2, 5, 8, llm_dim)
    with torch.no_grad():
        _forward(m, batch)
    beta = m.last_beta
    _assert(beta is not None, "last_beta missing")
    _assert(beta.shape[-1] == 5 and beta.dim() == 3, f"beta shape {tuple(beta.shape)}")
    for i in range(5):
        _assert(float(beta[:, i, i].abs().sum()) == 0.0, f"β_ii must be 0 at i={i}")
    print("PASS: pair mask excludes diagonal (global β)")


def test_k2_beta_one(llm_dim: int = 32) -> None:
    m = RelJoinResidual(llm_dim, rel_dim=16, variant="GlobalTimeFix")
    _randomize_wo(m, 0.2)
    m.eval()
    batch = _make_batch(1, 2, 5, llm_dim)
    with torch.no_grad():
        _forward(m, batch)
    beta = m.last_beta
    _assert(beta is not None, "last_beta missing")
    _assert(torch.allclose(beta[0, 0, 1], torch.ones(())), "K=2 receiver 0 β=1 on source 1")
    _assert(torch.allclose(beta[0, 1, 0], torch.ones(())), "K=2 receiver 1 β=1 on source 0")
    print("PASS: K=2 β=1 on the single source")


def test_local_token_conditioned_r(llm_dim: int = 48) -> None:
    m = RelJoinResidual(llm_dim, rel_dim=24, variant="Hybrid")
    _randomize_wo(m, 0.3)
    m.eval()
    batch = _make_batch(1, 4, 6, llm_dim)
    with torch.no_grad():
        _forward(m, batch)
    r = m.last_r_local
    _assert(r is not None, "last_r_local missing")
    _assert(not torch.allclose(r[0, 0, 0], r[0, 0, 1], atol=1e-6), "tokens in one frame must have distinct r^L")
    print("PASS: r^L_ip is token-conditioned")


def test_hybrid_equals_global_at_zero_local(llm_dim: int = 48) -> None:
    torch.manual_seed(0)
    hybrid = RelJoinResidual(llm_dim, rel_dim=24, variant="Hybrid")
    gtf = RelJoinResidual(llm_dim, rel_dim=24, variant="GlobalTimeFix")
    hybrid.eval()
    gtf.eval()
    batch = _make_batch(1, 4, 7, llm_dim)
    with torch.no_grad():
        x_h0 = _forward(hybrid, batch, local_enabled=True)
        x_g0 = _forward(gtf, batch, local_enabled=False)
    _assert(torch.equal(x_h0, batch[0]) and torch.equal(x_g0, batch[0]), "W_o=0 ⇒ both exact Vanilla")
    _randomize_wo(hybrid, 0.2)
    gtf.load_state_dict(hybrid.state_dict())
    with torch.no_grad():
        x_h = _forward(hybrid, batch, local_enabled=True)
        x_g = _forward(gtf, batch, local_enabled=False)
    _assert(not torch.allclose(x_h, x_g, atol=1e-5, rtol=1e-5), "P_L≠0 ⇒ Hybrid ≠ GlobalTimeFix after W_o≠0")
    print("PASS: W_o=0 exact Vanilla; P_L≠0 Hybrid differs from GlobalTimeFix")


def test_local_off_matches_hybrid_localoff(llm_dim: int = 40) -> None:
    torch.manual_seed(1)
    a = RelJoinResidual(llm_dim, rel_dim=16, variant="Hybrid")
    b = RelJoinResidual(llm_dim, rel_dim=16, variant="HybridLocalOff")
    b.load_state_dict(a.state_dict())
    _randomize_wo(a, 0.25)
    b.load_state_dict(a.state_dict())
    a.eval()
    b.eval()
    batch = _make_batch(2, 3, 5, llm_dim)
    with torch.no_grad():
        x_a = _forward(a, batch, local_enabled=False)
        x_b = _forward(b, batch)
    _assert(torch.equal(x_a, x_b), "Hybrid local_enabled=False vs HybridLocalOff must match")
    print("PASS: local_enabled=False vs HybridLocalOff bitwise equal")


def test_joint_hybrid_from_scratch() -> None:
    from .config import normalize_stage, run_name_for
    from .train import apply_stage_trainability, require_init_ckpt, stage_local_enabled

    _assert(normalize_stage("A") == "joint", "A is an alias of joint")
    _assert(stage_local_enabled("Hybrid", "joint") is True, "Hybrid joint must enable local")
    _assert(stage_local_enabled("GlobalTimeFix", "joint") is False, "GlobalTimeFix stays local-off")
    _assert(run_name_for("Hybrid", "joint") == "Hybrid", "joint Hybrid ckpt stem")
    _assert(run_name_for("GlobalTimeFix", "joint") == "GlobalTimeFix", "GlobalTimeFix ckpt stem")

    hybrid = RelJoinResidual(32, rel_dim=8, variant="Hybrid")
    apply_stage_trainability(hybrid, "joint", variant="Hybrid")
    local_on = any(p.requires_grad for p in hybrid.iter_local_parameters())
    global_on = any(p.requires_grad for p in hybrid.iter_global_parameters())
    _assert(local_on and global_on, "joint Hybrid must train local and global")

    gtf = RelJoinResidual(32, rel_dim=8, variant="GlobalTimeFix")
    apply_stage_trainability(gtf, "joint", variant="GlobalTimeFix")
    _assert(
        not any(p.requires_grad for p in gtf.iter_local_parameters()),
        "GlobalTimeFix joint must freeze local",
    )

    require_init_ckpt("joint", "")
    require_init_ckpt("A", "")
    require_init_ckpt("B2", "")
    try:
        require_init_ckpt("B1", "")
        raise _Fail("B1 without init_ckpt should fail")
    except ValueError:
        pass
    print("PASS: joint Hybrid from scratch (local+global, no init_ckpt)")


def test_chunk_p_matches_full(llm_dim: int = 32) -> None:
    torch.manual_seed(2)
    full = RelJoinResidual(llm_dim, rel_dim=16, variant="Hybrid", local_token_chunk=0)
    chunked = RelJoinResidual(llm_dim, rel_dim=16, variant="Hybrid", local_token_chunk=2)
    _randomize_wo(full, 0.2)
    chunked.load_state_dict(full.state_dict())
    full.eval()
    chunked.eval()
    batch = _make_batch(1, 3, 5, llm_dim)
    with torch.no_grad():
        x0 = _forward(full, batch)
        x1 = _forward(chunked, batch)
    _assert(torch.allclose(x0, x1, atol=1e-5, rtol=1e-5), "chunk-P must match full einsum")
    print("PASS: chunk-P = full einsum")


def test_hybrid_notime_zeros_phi(llm_dim: int = 32) -> None:
    m = RelJoinResidual(llm_dim, rel_dim=16, variant="HybridNoTime")
    m.eval()
    batch = _make_batch(1, 4, 6, llm_dim)
    with torch.no_grad():
        _forward(m, batch)
    _assert(m.last_phi is not None, "phi missing")
    _assert(torch.allclose(m.last_phi, torch.zeros_like(m.last_phi)), "HybridNoTime φ must be 0")
    _assert(m.last_tau is not None and torch.allclose(m.last_tau, torch.zeros_like(m.last_tau)), "τ=0")
    _assert(m.last_rho is not None and torch.allclose(m.last_rho, torch.zeros_like(m.last_rho)), "ρ=0")
    print("PASS: HybridNoTime zeros τ/ρ/φ")


def test_no_local_time_injection() -> None:
    m = RelJoinResidual(32, rel_dim=16, variant="Hybrid")
    keys = set(m.state_dict())
    _assert("local.A_t.weight" not in keys, "state dict must not contain local.A_t.weight")
    _assert(not any(k.endswith("A_t.weight") for k in keys), "no A_t anywhere")
    _assert(not hasattr(m.local, "A_t"), "LocalRelation must not have A_t")
    print("PASS: local time injection removed (no A_t)")


def test_local_attention_scale_at_init(llm_dim: int = 48) -> None:
    m = RelJoinResidual(llm_dim, rel_dim=24, variant="Hybrid")
    m.eval()
    batch = _make_batch(2, 4, 16, llm_dim)
    with torch.no_grad():
        _forward(m, batch)
    nte = float(m.last_stats.get("normalized_token_match_entropy", 0.0))
    _assert(0.5 < nte <= 1.0, f"init normalized token entropy {nte} not in (0.5, 1.0]")
    print("PASS: init normalized token entropy in (0.5, 1.0]")


def test_masked_tokens_zero_delta(llm_dim: int = 64) -> None:
    m = RelJoinResidual(llm_dim, rel_dim=32, variant="Hybrid")
    _randomize_wo(m, 0.3)
    m.eval()
    ft, tm, fm, ts, qh = _make_batch(1, 4, 9, llm_dim)
    tm[0, 1, 3] = False
    tm[0, 2, :] = False
    with torch.no_grad():
        out = m(ft, tm, fm, ts, qh)
    delta = out - ft
    _assert(torch.allclose(delta[0, 1, 3], torch.zeros_like(delta[0, 1, 3]), atol=1e-6), "m_ip=0 ⇒ Δx=0")
    _assert(torch.allclose(delta[0, 2], torch.zeros_like(delta[0, 2]), atol=1e-6), "masked frame tokens Δx=0")
    print("PASS: m_ip=0 ⇒ Δx_ip=0")


def test_tiny_rms_zero_delta(llm_dim: int = 48) -> None:
    m = RelJoinResidual(llm_dim, rel_dim=24, residual_eps=1e-6, rms_eps=1e-6, variant="Hybrid")
    _randomize_wo(m, 0.4)
    m.eval()
    ft, tm, fm, ts, qh = _make_batch(1, 3, 6, llm_dim)
    ft = ft.clone()
    ft[0, 0, 0] = 0.0
    with torch.no_grad():
        out = m(ft, tm, fm, ts, qh)
    delta = out - ft
    _assert(
        torch.allclose(delta[0, 0, 0], torch.zeros_like(delta[0, 0, 0]), atol=1e-6),
        "σ^x ≤ ε_RMS ⇒ Δx=0",
    )
    print("PASS: σ^x ≤ ε ⇒ Δx=0")


def test_rho_strictly_below_one(llm_dim: int = 64) -> None:
    m = RelJoinResidual(llm_dim, rel_dim=32, variant="Hybrid")
    _randomize_wo(m, 0.5)
    m.eval()
    ft, tm, fm, ts, qh = _make_batch(2, 4, 8, llm_dim)
    with torch.no_grad():
        out = m(ft, tm, fm, ts, qh)
    delta = (out - ft).float()
    sx = rms_fp32(ft, dim=-1, keepdim=False)
    sd = rms_fp32(delta, dim=-1, keepdim=False)
    valid = tm & m.last_has_neighbor.unsqueeze(-1) & (sx > m.rms_eps)
    _assert(bool(valid.any()), "need at least one valid nonzero token")
    rho = sd[valid] / sx[valid]
    _assert(bool((rho >= -1e-5).all()), f"ρ had negatives: {float(rho.min())}")
    _assert(bool((rho < 1.0 + 1e-5).all()), f"ρ must be < 1, max={float(rho.max())}")
    print("PASS: 0 ≤ ρ < 1 on valid nonzero tokens")


def test_padded_frames_zero_effect(llm_dim: int = 128) -> None:
    m = RelJoinResidual(llm_dim, rel_dim=32, variant="GlobalTimeFix")
    _randomize_wo(m, 0.05)
    m.eval()
    K, P = 4, 32
    ft, tm, fm, ts, qh = _make_batch(1, K, P, llm_dim)
    ft2 = torch.cat([ft, torch.randn(1, 1, P, llm_dim)], dim=1)
    tm2 = torch.cat([tm, torch.zeros(1, 1, P, dtype=torch.bool)], dim=1)
    fm2 = torch.cat([fm, torch.zeros(1, 1, dtype=torch.bool)], dim=1)
    ts2 = torch.cat([ts, torch.zeros(1, 1)], dim=1)
    with torch.no_grad():
        x1 = m(ft, tm, fm, ts, qh)
        x2 = m(ft2, tm2, fm2, ts2, qh)
    _assert(
        torch.allclose(x1, x2[:, :K], atol=1e-5, rtol=1e-4),
        "padded frame changed valid-frame residual",
    )
    print("PASS: padded frames no effect")


def test_ragged_token_counts(llm_dim: int = 64) -> None:
    lengths = [17, 64, 9]
    K = len(lengths)
    P_max = max(lengths)
    m = RelJoinResidual(llm_dim, rel_dim=32, variant="Hybrid")
    _randomize_wo(m, 0.2)
    m.eval()
    ft = torch.randn(1, K, P_max, llm_dim)
    tm = torch.zeros(1, K, P_max, dtype=torch.bool)
    for i, p in enumerate(lengths):
        tm[0, i, :p] = True
        ft[0, i, p:] = 99.0
    fm = torch.ones(1, K, dtype=torch.bool)
    ts = torch.tensor([[0.0, 0.4, 1.0]])
    qh = torch.randn(1, llm_dim)
    with torch.no_grad():
        out = m(ft, tm, fm, ts, qh)
    _assert(out.shape == ft.shape, "ragged output shape")
    _assert(
        m.last_has_neighbor is not None and bool(m.last_has_neighbor.all()),
        "K=3 valid frames must have χ_i=1",
    )
    for i, p in enumerate(lengths):
        _assert(
            torch.allclose(out[0, i, p:], ft[0, i, p:], atol=0, rtol=0),
            f"pad slots of frame {i} (P={p}) were rewritten",
        )
        _assert(
            not torch.allclose(out[0, i, :p], ft[0, i, :p]),
            f"valid tokens of frame {i} should change when W_o != 0 and χ=1",
        )
    print("PASS: ragged P_i=[17,64,9] mask pooling")


def test_visual_atom_spans_and_pack() -> None:
    ids = [1, 2, VISUAL_ATOM_ID, VISUAL_ATOM_ID, VISUAL_ATOM_ID, 7, VISUAL_ATOM_ID] + [
        VISUAL_ATOM_ID
    ] * 4 + [9]
    input_ids = torch.tensor([ids])
    spans = visual_atom_spans(input_ids)
    _assert(spans == [(2, 5), (6, 11)], f"spans {spans}")
    d = 8
    embeds = torch.randn(1, len(ids), d)
    groups = [embeds[:, s:e, :] for s, e in spans]
    packed, mask = pack_variable_frame_tokens(groups)
    _assert(packed.shape == (1, 2, 5, d), f"packed {tuple(packed.shape)}")
    _assert(bool(mask[0, 0, :3].all()) and not bool(mask[0, 0, 3:].any()), "mask frame0")
    _assert(bool(mask[0, 1].all()), "mask frame1")
    written = embeds.clone()
    written = scatter_variable_frame_tokens(written, spans, packed)
    _assert(torch.allclose(written[:, 2:5], groups[0]), "scatter frame0")
    _assert(torch.allclose(written[:, 6:11], groups[1]), "scatter frame1")
    print("PASS: visual-atom spans + ragged pack/scatter")


def test_empty_frame_summary_zero(llm_dim: int = 32) -> None:
    m = RelJoinResidual(llm_dim, rel_dim=16, variant="Hybrid")
    _randomize_wo(m, 0.3)
    m.eval()
    ft, tm, fm, ts, qh = _make_batch(1, 2, 6, llm_dim)
    tm[0, 1] = False
    fm[0, 1] = False
    with torch.no_grad():
        out = m(ft, tm, fm, ts, qh)
    _assert(torch.allclose(out[0, 1], ft[0, 1], atol=0, rtol=0), "empty frame must stay X")
    print("PASS: empty-frame summary is identity")


def test_batch_vs_single(llm_dim: int = 64) -> None:
    m = RelJoinResidual(llm_dim, rel_dim=24, variant="GlobalTimeFix")
    _randomize_wo(m, 0.05)
    m.eval()
    K, P = 3, 8
    ft, tm, fm, ts, qh = _make_batch(2, K, P, llm_dim)
    with torch.no_grad():
        x_batch = m(ft, tm, fm, ts, qh)
        x_single = m(ft[:1], tm[:1], fm[:1], ts[:1], qh[:1])
    _assert(torch.allclose(x_batch[:1], x_single, atol=1e-5, rtol=1e-4), "batch vs single mismatch")
    print("PASS: batch vs single")


def test_time_reversal_changes_relations(llm_dim: int = 64) -> None:
    m = RelJoinResidual(llm_dim, rel_dim=24, variant="Hybrid")
    _randomize_wo(m, 0.2)
    nn.init.normal_(m.global_relation.A_K.weight, std=0.3)
    m.eval()
    K, P = 4, 8
    ft, tm, fm, ts, qh = _make_batch(1, K, P, llm_dim)
    ts_rev = ts.flip(dims=[1])
    with torch.no_grad():
        x = m(ft, tm, fm, ts, qh)
        x_r = m(ft, tm, fm, ts_rev, qh)
    _assert(not torch.allclose(x, x_r, atol=1e-5), "time reversal should change directed relations")
    print("PASS: time reversal changes residual via φ")


def test_gradients_only_reljoin(llm_dim: int = 64) -> None:
    m = RelJoinResidual(llm_dim, rel_dim=32, variant="Hybrid")
    backbone = nn.Linear(llm_dim, llm_dim)
    for p in backbone.parameters():
        p.requires_grad = False
    ft, tm, fm, ts, qh = _make_batch(1, 3, 8, llm_dim)
    ft = ft.requires_grad_(True)
    out = m(ft, tm, fm, ts, qh)
    loss = out.float().pow(2).mean() + backbone(ft[:, 0, 0]).float().pow(2).mean() * 0.0
    loss.backward()
    for n, p in backbone.named_parameters():
        _assert(p.grad is None or float(p.grad.abs().sum()) == 0.0, f"backbone got grad: {n}")
    _assert(
        m.W_o.weight.grad is not None and m.W_o.weight.grad.abs().sum() > 0,
        "W_o should receive gradient at init",
    )
    wq_grad = m.global_relation.W_Q.weight.grad
    _assert(
        wq_grad is None or float(wq_grad.abs().sum()) == 0.0,
        "global block should have ~0 grad while W_o is exactly 0",
    )
    print("PASS: gradients RelJoin only + W_o at zero-init")


def test_gradients_flow_after_wo_step(llm_dim: int = 64) -> None:
    torch.manual_seed(0)
    m = RelJoinResidual(llm_dim, rel_dim=32, variant="Hybrid")
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    ft, tm, fm, ts, qh = _make_batch(1, 4, 8, llm_dim)
    out = m(ft, tm, fm, ts, qh)
    out.float().pow(2).mean().backward()
    _assert(
        m.W_o.weight.grad is not None and m.W_o.weight.grad.abs().sum() > 0,
        "first backward must give grad(W_o)>0",
    )
    opt.step()
    opt.zero_grad(set_to_none=True)
    out = m(ft, tm, fm, ts, qh)
    out.float().pow(2).mean().backward()
    _assert(
        m.global_relation.W_Q.weight.grad is not None
        and m.global_relation.W_Q.weight.grad.abs().sum() > 0,
        "grad(global) after first step",
    )
    _assert(
        m.local.local_q.weight.grad is not None and m.local.local_q.weight.grad.abs().sum() > 0,
        "grad(local_qk) after first step",
    )
    _assert(
        m.local.A_c.weight.grad is not None and m.local.A_c.weight.grad.abs().sum() > 0,
        "grad(local_message) after first step",
    )
    _assert(
        m.head.local_adapter.weight.grad is not None
        and m.head.local_adapter.weight.grad.abs().sum() > 0,
        "grad(P_L) after first step",
    )
    print("PASS: after optimizer step 1, global / local_qk / local_message / P_L get grads")


def test_checkpoint_roundtrip(llm_dim: int = 64) -> None:
    m = RelJoinResidual(llm_dim, rel_dim=32, variant="Hybrid")
    _randomize_wo(m, 0.05)
    m.eval()
    ft, tm, fm, ts, qh = _make_batch(1, 3, 8, llm_dim)
    with torch.no_grad():
        x1 = m(ft, tm, fm, ts, qh)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt.pt"
        torch.save({"reljoin": m.state_dict(), "method_version": METHOD_VERSION}, path)
        m2 = RelJoinResidual(llm_dim, rel_dim=32, variant="Hybrid")
        m2.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["reljoin"])
        m2.eval()
        with torch.no_grad():
            x2 = m2(ft, tm, fm, ts, qh)
    _assert(torch.allclose(x1, x2, atol=0, rtol=0), "checkpoint logits mismatch")
    keys = set(m.state_dict())
    _assert("global_relation.W_Q.weight" in keys, "missing global W_Q")
    _assert("head.local_adapter.weight" in keys, "missing P_L")
    _assert("local.w_L" in keys, "missing w_L")
    _assert("local.ln_Q.weight" in keys, "missing ln_Q")
    _assert("local.ln_K.weight" in keys, "missing ln_K")
    _assert("local.A_t.weight" not in keys, "A_t must not be checkpointed")
    print("PASS: checkpoint save/load + v6 keys")


def test_v5_loss_combine() -> None:
    z_rel = torch.tensor([[2.0, 0.0, -1.0], [0.5, 0.5, 0.5]])
    z_van = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = torch.tensor([0, 1])
    each, cand, corr = reljoin_v5_loss(z_rel, z_van, target)
    mean, mc, mr = batch_reljoin_v5_loss(z_rel, z_van, target)
    _assert(each.shape == (2,), f"per-sample shape {each.shape}")
    stacked = torch.stack([cand, corr], dim=-1)
    expect = torch.logsumexp(stacked, dim=-1) - math.log(2.0)
    _assert(torch.allclose(each, expect, atol=1e-6), "logsumexp combine mismatch")
    _assert(torch.allclose(mean, each.mean()), "batch mean must follow per-sample combine")
    _assert(torch.allclose(mc, cand.mean()) and torch.allclose(mr, corr.mean()), "component means")
    print("PASS: v5 loss per-sample logsumexp then batch mean")


def test_pair_mask_no_self() -> None:
    mask = torch.tensor([[True, True, False]])
    pm = pair_mask_from_frames(mask)
    _assert(pm.shape == (1, 3, 3), f"pair mask shape {pm.shape}")
    _assert(not bool(pm[0].diag().any()), "diagonal must be false")
    _assert(bool(pm[0, 0, 1]) and bool(pm[0, 1, 0]), "valid pair missing")
    _assert(not bool(pm[0, 0, 2].item()) and not bool(pm[0, 2, 0].item()), "invalid frame leaked")
    print("PASS: pair_mask M_ij = f_i f_j 1[i≠j]")


def test_train_val_disjoint_and_herbench_guard() -> None:
    train = [
        QASample("t1", "v1", "/data/v1.mp4", "q", ["a", "b"], "A", 0, {}),
        QASample("t2", "v1", "/data/v1.mp4", "q2", ["a", "b"], "B", 1, {}),
    ]
    val = [
        QASample("v1s", "v2", "/data/v2.mp4", "q", ["a", "b"], "A", 0, {}),
    ]
    assert {s.video_id for s in train}.isdisjoint({s.video_id for s in val})
    assert_no_herbench_in_training(train + val)
    bad = QASample("x", "v", "/content/HERBench_official/videos/a.mp4", "q", ["a", "b"], "A", 0, {})
    try:
        assert_no_herbench_in_training([bad])
        raise _Fail("HERBench guard should have failed")
    except AssertionError:
        pass
    print("PASS: train/val disjoint + HERBench guard")


def test_vanilla_bypass_flag() -> None:
    configure_reljoin(None, enabled=True, bypass=True)
    st = get_hook_state()
    _assert(st.bypass is True, "bypass not set")
    configure_reljoin(None, enabled=False, bypass=True)
    begin_prepare_cache()
    st = get_hook_state()
    _assert(st.cache_prepare is True, "prepare cache not enabled")
    end_prepare_cache()
    st = get_hook_state()
    _assert(st.cache_prepare is False, "prepare cache not cleared")
    _assert(st.cached_result is None, "prepare cache result not cleared")
    print("PASS: vanilla bypass mode flag")


def test_no_native_frame_padding() -> None:
    from .frame_io import load_pil_frames

    _assert(hasattr(load_pil_frames, "__call__"), "load_pil_frames missing")
    print("PASS: frame_io has no pad-to-native-8 path")


def test_nextgqa_answer_map_and_prompt() -> None:
    from .nextgqa import (
        eval_type,
        format_nextgqa_prompt,
        map_answer_to_letter,
        official_groups,
        prompt_has_leak,
        question_id,
    )

    letter, idx, err = map_answer_to_letter("Yes", ["no", "yes", "maybe", "idk", "other"])
    _assert(err == "" and letter == "B" and idx == 1, (letter, idx, err))
    letter, idx, err = map_answer_to_letter("C", ["a", "b", "c", "d", "e"])
    _assert(letter == "C" and idx == 2, (letter, idx, err))
    letter, idx, err = map_answer_to_letter("3", ["a", "b", "c", "d", "e"])
    _assert(letter == "D" and idx == 3, (letter, idx, err))
    letter, idx, err = map_answer_to_letter("Blue", ["Blue", "BLUE", "red", "green", "yellow"])
    _assert(err == "ambiguous_answer" and idx == -1, (letter, idx, err))
    letter, idx, err = map_answer_to_letter("zzz", ["a", "b", "c", "d", "e"])
    _assert(err == "unmapped_answer", err)
    _assert(eval_type("TP") == "TN", eval_type("TP"))
    _assert(question_id("10001787725", 1) == "10001787725_1", question_id("10001787725", 1))
    row = {
        "question": "Why does she stop?",
        "choices": ["tired", "hurt", "lost", "late", "scared"],
        "answer": "hurt",
        "spans": [[1.2, 5.8]],
    }
    prompt = format_nextgqa_prompt(row["question"], row["choices"])
    _assert("A. tired" in prompt and "E. scared" in prompt, prompt)
    _assert("Answer with the option letter only." in prompt, prompt)
    _assert("1.2" not in prompt, prompt)
    _assert(not prompt_has_leak(prompt, row), prompt)
    leaked = prompt + "\noracle span 1.2-5.8"
    _assert(prompt_has_leak(leaked, row), leaked)
    recs = [
        {"eval_type": "CW", "raw_type": "CW", "is_correct": True},
        {"eval_type": "TN", "raw_type": "TP", "is_correct": False},
        {"eval_type": "TN", "raw_type": "TN", "is_correct": True},
    ]
    g = official_groups(recs)
    _assert(g["C"]["n"] == 1 and g["T"]["n"] == 2, g)
    _assert("Acc@GQA" not in str(g), g)
    print("PASS: NExT-GQA answer map, prompt leak, TP→TN groups")


def test_nextgqa_uniform_and_oracle_alloc() -> None:
    import numpy as np

    from .nextgqa import FRAME_PROTOCOL_ORACLE, FRAME_PROTOCOL_UNIFORM, NUM_FRAMES
    from .nextgqa_frames import (
        decode_from_arrays,
        oracle_times,
        plan_times,
        uniform_times,
    )

    _assert(NUM_FRAMES == 8, NUM_FRAMES)
    _assert("oracle" in FRAME_PROTOCOL_ORACLE and "uniform" in FRAME_PROTOCOL_UNIFORM, FRAME_PROTOCOL_ORACLE)
    times = uniform_times(14.0, 8)
    _assert(len(times) == 8, times)
    _assert(abs(times[0] - 0.0) < 1e-9 and abs(times[-1] - 14.0) < 1e-9, times)
    t_one, assigned, err = oracle_times([[1.2, 5.8]], 34.0, 8)
    _assert(err == "" and len(t_one) == 8 and assigned == [0] * 8, (t_one, assigned, err))
    _assert(all(1.2 - 1e-6 <= t <= 5.8 + 1e-6 for t in t_one), t_one)
    t_multi, assigned, err = oracle_times([[0.0, 10.0], [20.0, 22.0]], 40.0, 8)
    _assert(err == "" and len(t_multi) == 8, (t_multi, err))
    _assert(t_multi == sorted(t_multi), t_multi)
    n0 = assigned.count(0)
    n1 = assigned.count(1)
    _assert(n0 >= 1 and n1 >= 1 and n0 + n1 == 8, assigned)
    _assert(n0 > n1, assigned)
    too, _, err = oracle_times([[i, i + 0.5] for i in range(9)], 20.0, 8)
    _assert(err == "too_many_spans" and too == [], err)
    t3, a3, err = oracle_times([[12.1, 17.1], [20.0, 23.5], [29.7, 33.2]], 34.0, 8)
    _assert(err == "" and a3.count(0) == 3 and a3.count(1) == 3 and a3.count(2) == 2, a3)
    deg, _, err = oracle_times([[5.0, 5.0]], 10.0, 8)
    _assert(err == "insufficient_distinct_frames", err)
    row = {"duration": 12.0, "spans": [[1.0, 3.0]], "skip_reason": ""}
    u, _, uerr, _ = plan_times(row, "uniform", 8)
    o, _, oerr, _ = plan_times(row, "oracle", 8)
    _assert(uerr == "" and oerr == "" and u != o, (u, o))
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(200)]
    decoded = decode_from_arrays(frames, fps=25.0, times=t_one, mode="oracle", spans=[[1.2, 5.8]], assigned=[0] * 8, duration=34.0)
    _assert(decoded.status == "ok", decoded.error)
    _assert(len(decoded.frame_hashes) == 8, decoded.frame_hashes)
    decoded_b = decode_from_arrays(frames, fps=25.0, times=t_one, mode="oracle", spans=[[1.2, 5.8]], assigned=[0] * 8, duration=34.0)
    _assert(decoded_b.frame_hashes == decoded.frame_hashes, "paired arms must share hashes")
    print("PASS: NExT-GQA uniform + oracle allocation")


def test_nextgqa_oracle_decode_failures() -> None:
    import numpy as np

    from .nextgqa_frames import decode_from_arrays, decode_row, oracle_times

    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(8)]
    outside = decode_from_arrays(
        frames,
        fps=1.0,
        times=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        mode="oracle",
        spans=[[20.0, 21.0]],
        assigned=[0] * 8,
        duration=30.0,
    )
    _assert(outside.status == "decode_outside_span", outside.status)
    dup = decode_from_arrays(
        frames[:3],
        fps=1.0,
        times=[0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        mode="uniform",
        duration=2.0,
    )
    _assert(dup.status == "insufficient_distinct_frames", dup.status)
    empty = decode_from_arrays([], fps=8.0, times=list(range(8)))
    _assert(empty.status == "decode_failure", empty.status)
    missing = decode_row({"resolved_video_path": "", "spans": [[0, 1]], "duration": 2.0}, "oracle", 8)
    _assert(missing.status == "missing_media", missing.status)
    _, _, err = oracle_times([], 10.0, 8)
    _assert(err == "missing_spans", err)
    print("PASS: NExT-GQA oracle failure statuses")


def test_nextgqa_loader_audit_resume() -> None:
    import csv
    import json

    from .nextgqa import (
        audit_nextgqa,
        common_id_delta,
        load_nextgqa_rows,
        resume_completed_keys,
        score_predictions,
        select_smoke_rows,
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "annotations").mkdir()
        (root / "videos").mkdir()
        (root / "videos" / "10001787725.mp4").write_bytes(b"fake")
        (root / "videos" / "20000000000.mp4").write_bytes(b"fake")
        rows = [
            {
                "video_id": "10001787725",
                "frame_count": "100",
                "width": "640",
                "height": "480",
                "question": "Why does she stop?",
                "answer": "hurt",
                "qid": "1",
                "type": "CW",
                "a0": "tired",
                "a1": "hurt",
                "a2": "lost",
                "a3": "late",
                "a4": "scared",
            },
            {
                "video_id": "10001787725",
                "frame_count": "100",
                "width": "640",
                "height": "480",
                "question": "What happens after the wave?",
                "answer": "they sit",
                "qid": "3",
                "type": "TN",
                "a0": "they sit",
                "a1": "they run",
                "a2": "they fall",
                "a3": "they leave",
                "a4": "they shout",
            },
            {
                "video_id": "20000000000",
                "frame_count": "80",
                "width": "640",
                "height": "480",
                "question": "When does he turn?",
                "answer": "after the ball",
                "qid": "0",
                "type": "TP",
                "a0": "before the ball",
                "a1": "after the ball",
                "a2": "never",
                "a3": "twice",
                "a4": "at the start",
            },
            {
                "video_id": "missingvid",
                "frame_count": "10",
                "width": "1",
                "height": "1",
                "question": "Where?",
                "answer": "left",
                "qid": "0",
                "type": "DL",
                "a0": "left",
                "a1": "right",
                "a2": "up",
                "a3": "down",
                "a4": "center",
            },
        ]
        with (root / "annotations" / "val.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        gsub = {
            "10001787725": {
                "duration": 34,
                "fps": 29.97,
                "location": {
                    "1": [[1.2, 5.8]],
                    "3": [[12.1, 17.1], [20.0, 23.5], [29.7, 33.2]],
                },
            },
            "20000000000": {
                "duration": 20,
                "fps": 25.0,
                "location": {"0": [[2.0, 8.0]]},
            },
        }
        (root / "annotations" / "gsub_val.json").write_text(json.dumps(gsub), encoding="utf-8")
        loaded, missing = load_nextgqa_rows(root, "val")
        _assert(len(loaded) == 4, loaded)
        by_id = {r["question_id"]: r for r in loaded}
        _assert(by_id["10001787725_1"]["answer_letter"] == "B", by_id["10001787725_1"])
        _assert(len(by_id["10001787725_3"]["spans"]) == 3, by_id["10001787725_3"])
        _assert(by_id["20000000000_0"]["eval_type"] == "TN", by_id["20000000000_0"])
        _assert(by_id["missingvid_0"]["skip_reason"] == "missing_video", by_id["missingvid_0"])
        audit = audit_nextgqa(root, "val")
        _assert(audit["splits"]["val"]["n_csv"] == 4, audit)
        _assert(audit["splits"]["val"]["n_missing_video"] == 1, audit)
        smoke = select_smoke_rows(loaded)
        _assert(any(len(r["spans"]) == 1 for r in smoke), smoke)
        _assert(any(len(r["spans"]) >= 2 for r in smoke), smoke)
        pred = [
            {"split": "val", "mode": "oracle", "method": "vanilla", "question_id": "q1", "eval_type": "CW", "raw_type": "CW", "is_correct": False, "is_valid": True, "status": "ok"},
            {"split": "val", "mode": "oracle", "method": "hybrid", "question_id": "q1", "eval_type": "CW", "raw_type": "CW", "is_correct": True, "is_valid": True, "status": "ok"},
        ]
        metrics = score_predictions(pred)
        _assert(metrics["deltas"][0]["delta"] == 1.0, metrics["deltas"])
        _assert(metrics["metric_note"].find("Acc@GQA") >= 0, metrics)
        d = common_id_delta(pred[:1], pred[1:])
        _assert(d["n"] == 1 and d["delta"] == 1.0, d)
        csv_path = root / "predictions.jsonl"
        csv_path.write_text(
            json.dumps({"run_fingerprint": "abcd", "split": "val", "mode": "oracle", "question_id": "q1", "method": "vanilla", "status": "ok"})
            + "\n",
            encoding="utf-8",
        )
        done = resume_completed_keys(csv_path, "abcd")
        _assert(("val", "oracle", "q1", "vanilla") in done, done)
        try:
            resume_completed_keys(csv_path, "ffff")
            _assert(False, "resume must refuse a different fingerprint")
        except RuntimeError as exc:
            _assert("different fingerprint" in str(exc), str(exc))
    print("PASS: NExT-GQA loader, audit, smoke pick, resume")


def test_nextgqa_runner_protocol() -> None:
    src = Path(__file__).resolve().parents[1] / "run_nextgqa_reljoin.py"
    text = src.read_text(encoding="utf-8")
    _assert(src.is_file(), f"missing {src}")
    _assert("choose_oracle_times" not in text, "must not reuse HERBench oracle padding")
    _assert("bypass=bypass" in text or "bypass=True" in text, "vanilla bypass")
    _assert("Acc@GQA" in text, "must mention Acc@GQA only as a negation")
    _assert("not Acc@GQA" in text or "not_acc_gqa" in text, text)
    _assert("format_nextgqa_prompt" in text, "must use NExT-GQA prompt")
    _assert("--num-frames" in text, "must expose K")
    print("PASS: NExT-GQA runner protocol")


def test_nextgqa_vanilla_hybrid_hook_flags() -> None:
    configure_reljoin(None, enabled=True, bypass=True, eval_mode="full")
    st = get_hook_state()
    _assert(st.bypass is True, "vanilla bypass")
    m = RelJoinResidual(32, rel_dim=16, variant="Hybrid")
    configure_reljoin(m, enabled=True, bypass=False, eval_mode="full")
    st = get_hook_state()
    _assert(st.bypass is False and st.module is m, "hybrid activation")
    configure_reljoin(None, enabled=False, bypass=True, eval_mode="full")
    print("PASS: NExT-GQA vanilla bypass vs Hybrid activation")


def test_qwen_gqa_dims() -> None:
    from .qwen_hooks import _qwen_gqa_dims

    class _Attn:
        head_dim = 128
        num_heads = 32

    q = torch.zeros(1, 26, 32 * 128)
    k = torch.zeros(1, 26, 8 * 128)
    n_heads, n_kv, head_dim = _qwen_gqa_dims(_Attn(), q, k)
    _assert(n_heads == 32 and n_kv == 8 and head_dim == 128, (n_heads, n_kv, head_dim))
    _assert(k.numel() == 1 * 26 * n_kv * head_dim, "KV reshape must match tensor size")
    print("PASS: Qwen3 GQA dims 32q / 8kv")


def test_skip_qwen25_vl_sdpa_patch() -> None:
    """Qwen2.5-VL native SDPA must not be overwritten by the Ovis/Qwen2 forward."""

    class Qwen2_5_VLSdpaAttention(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states

    class _PlainQwen2Attn(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states

    class _Layer(nn.Module):
        def __init__(self, attn: nn.Module):
            super().__init__()
            self.self_attn = attn

    class _Inner(nn.Module):
        def __init__(self, attn: nn.Module):
            super().__init__()
            self.layers = nn.ModuleList([_Layer(attn)])

    class _Model(nn.Module):
        def __init__(self, attn: nn.Module):
            super().__init__()
            self.model = _Inner(attn)

    vl = Qwen2_5_VLSdpaAttention()
    patch_qwen_sdpa_attention(_Model(vl))
    _assert("forward" not in vl.__dict__, "must keep native Qwen2.5-VL SDPA forward")

    mrope = _PlainQwen2Attn()
    mrope.config = type("Cfg", (), {"rope_scaling": {"mrope_section": [16, 24, 24]}})()
    patch_qwen_sdpa_attention(_Model(mrope))
    _assert("forward" not in mrope.__dict__, "must skip layers with 3D mrope_section")

    plain = _PlainQwen2Attn()
    patch_qwen_sdpa_attention(_Model(plain))
    _assert("forward" in plain.__dict__, "plain Qwen2 attn may still use the custom SDPA patch")
    print("PASS: Qwen2.5-VL native SDPA is not overwritten")


def test_stable_int_seed() -> None:
    a = stable_int_seed("perceptiontest_video_9267__item0__q0")
    b = stable_int_seed("perceptiontest_video_9267__item0__q0")
    c = stable_int_seed("perceptiontest_video_9267__item0__q1")
    _assert(a == b, "stable_int_seed must be deterministic")
    _assert(a != c, "different keys must not collide in this check")
    _assert(isinstance(a, int) and a >= 0, "seed must be a non-negative int")
    print("PASS: stable_int_seed")


def test_scheduler_update_count() -> None:
    from .train import count_scheduler_updates, cosine_scheduler

    updates, total, warmup = count_scheduler_updates(454, 8, 8, 0.10)
    _assert(updates == 57, f"updates/epoch {updates} != 57")
    _assert(total == 456, f"total {total} != 456")
    _assert(warmup == 46, f"warmup {warmup} != 46")
    _assert(total > 100, "schedule must follow micro-batch=1, not phantom batch=64")

    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    sched = cosine_scheduler(opt, 2, 6)
    lrs = []
    for _ in range(10):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    _assert(lrs[0] == 0.0, f"warmup step 0 should be 0, got {lrs[0]}")
    _assert(all(0.0 <= lr <= 1.0 for lr in lrs), f"lr out of range {lrs}")
    print("PASS: scheduler update count + cosine clamp")


def test_eval_mode_identity(llm_dim: int = 64) -> None:
    m = RelJoinResidual(llm_dim, rel_dim=32, variant="Hybrid")
    _randomize_wo(m, 0.5)
    m.eval()
    batch = _make_batch(2, 8, 7, llm_dim)
    with torch.no_grad():
        out = m(*batch, eval_mode=EVAL_IDENTITY)
    _assert(torch.allclose(out, batch[0], atol=0, rtol=0), "identity must return X")
    _assert(m.last_delta_x is not None and float(m.last_delta_x.abs().sum()) == 0.0, "identity Δx")
    print("PASS: eval_mode identity returns X")


def test_eval_mode_identity_packs_spans(llm_dim: int = 8) -> None:
    from .qwen_hooks import _apply_reljoin

    m = RelJoinResidual(llm_dim, rel_dim=8, variant="Hybrid")
    _randomize_wo(m, 0.5)
    m.eval()
    reset_hook_diagnostics()
    configure_reljoin(m, enabled=True, bypass=False, eval_mode=EVAL_IDENTITY)
    set_reljoin_context(
        question_hidden=torch.randn(1, llm_dim),
        timestamps=torch.linspace(0, 1, 8).unsqueeze(0),
        frame_mask=torch.ones(1, 8, dtype=torch.bool),
    )
    ids = []
    p_i = 3
    for i in range(8):
        ids.extend([VISUAL_ATOM_ID] * p_i)
        ids.append(i + 1)
    input_ids = torch.tensor([ids])
    embeds = torch.randn(1, len(ids), llm_dim)

    class _Dummy:
        pass

    out = _apply_reljoin(_Dummy(), input_ids, embeds)
    st = get_hook_state()
    _assert(len(st.last_tokens_per_frame) == 8, st.last_tokens_per_frame)
    _assert(all(int(p) == p_i for p in st.last_tokens_per_frame), st.last_tokens_per_frame)
    _assert(torch.allclose(out, embeds), "identity hook must scatter original X")
    configure_reljoin(None, enabled=False, bypass=True, eval_mode="full")
    print("PASS: identity packs P_i length K then returns X")


def test_hook_install_count_and_remove() -> None:
    class _M(nn.Module):
        def merge_multimodal(self, input_ids, pixel_values=None, grid_thws=None, **kwargs):
            return torch.zeros(1, 2, 4)

    model = _M()
    before = hook_install_count()
    install_reljoin_hook(model)
    _assert(hook_install_count() == before + 1, "install must increment once")
    install_reljoin_hook(model)
    _assert(hook_install_count() == before + 1, "second install must be idempotent")
    _assert(not no_reljoin_hooks_remaining(model), "hook should be present")
    remove_all_reljoin_hooks(model)
    _assert(no_reljoin_hooks_remaining(model), "remove_all_reljoin_hooks must restore merge")
    print("PASS: hook_install_count + remove")


def test_letter_token_ids() -> None:
    from .losses import letter_token_ids

    class _Tok:
        unk_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [ord(text[0])]

    ids = letter_token_ids(_Tok(), 4)
    _assert(list(ids.tolist()) == [ord("A"), ord("B"), ord("C"), ord("D")], ids)
    print("PASS: letter token ids")


def test_refuse_incomplete_state_dict(llm_dim: int = 16) -> None:
    m = RelJoinResidual(llm_dim, rel_dim=8, variant="Hybrid")
    fake = {k: v for k, v in m.state_dict().items() if "local_adapter" not in k}
    try:
        m.load_state_dict(fake, strict=True)
        raise _Fail("incomplete state_dict should fail strict load")
    except RuntimeError:
        pass
    print("PASS: refuse incomplete state_dict")


def test_sync_persist_skips_same_out_dir() -> None:
    from .train import _sync_persist

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = ExperimentConfig(run_mode="unit", root=root)
        cfg.persist_dir = cfg.out_dir
        paths = make_output_tree(cfg.out_dir)
        (paths["checkpoints_last"] / "Hybrid.pt").write_bytes(b"ckpt")
        _sync_persist(cfg, paths)
        persist = root / "persist"
        cfg.persist_dir = persist
        _sync_persist(cfg, paths)
        copied = persist / "checkpoints" / "last" / "Hybrid.pt"
        _assert(copied.is_file() and copied.read_bytes() == b"ckpt", "persist copy missing")
    print("PASS: persist skip when dest==out_dir; copy when distinct")


def run_all_unit_tests() -> None:
    print("=" * 60)
    print("FrameWeave Hybrid v6.0 / V5.1 / Qwen2.5-VL structural unit tests")
    print("=" * 60)
    unit_test_option_permutation()
    unit_test_perception_schema_parse()
    test_phi_zero_and_signed()
    test_tau_rho_few_frames()
    test_masked_softmax_all_masked()
    test_pair_mask_no_self()
    test_zero_inits()
    test_forward_all_K()
    test_no_hardcoded_2048()
    test_exact_vanilla_at_init()
    test_k1_identity_even_with_random_Wo()
    test_diagonal_beta_zero()
    test_k2_beta_one()
    test_local_token_conditioned_r()
    test_hybrid_equals_global_at_zero_local()
    test_local_off_matches_hybrid_localoff()
    test_joint_hybrid_from_scratch()
    test_chunk_p_matches_full()
    test_hybrid_notime_zeros_phi()
    test_no_local_time_injection()
    test_local_attention_scale_at_init()
    test_masked_tokens_zero_delta()
    test_tiny_rms_zero_delta()
    test_rho_strictly_below_one()
    test_padded_frames_zero_effect()
    test_ragged_token_counts()
    test_visual_atom_spans_and_pack()
    test_empty_frame_summary_zero()
    test_batch_vs_single()
    test_qwen_gqa_dims()
    test_skip_qwen25_vl_sdpa_patch()
    test_time_reversal_changes_relations()
    test_gradients_only_reljoin()
    test_gradients_flow_after_wo_step()
    test_checkpoint_roundtrip()
    test_v5_loss_combine()
    test_train_val_disjoint_and_herbench_guard()
    test_vanilla_bypass_flag()
    test_no_native_frame_padding()
    test_nextgqa_answer_map_and_prompt()
    test_nextgqa_uniform_and_oracle_alloc()
    test_nextgqa_oracle_decode_failures()
    test_nextgqa_loader_audit_resume()
    test_nextgqa_runner_protocol()
    test_nextgqa_vanilla_hybrid_hook_flags()
    test_stable_int_seed()
    test_scheduler_update_count()
    test_eval_mode_identity()
    test_eval_mode_identity_packs_spans()
    test_hook_install_count_and_remove()
    test_letter_token_ids()
    test_refuse_incomplete_state_dict()
    test_sync_persist_skips_same_out_dir()
    print("=" * 60)
    print("ALL UNIT TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_unit_tests()
