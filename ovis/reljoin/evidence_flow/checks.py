"""CPU checks for merge order and, when PyTorch is installed, directed ablations."""
from __future__ import annotations

import numpy as np

from reljoin.evidence_flow.geometry import run_merge_order_self_test
from reljoin.evidence_flow.mine import label_oracle_times
from reljoin.evidence_flow.select import select_keyframes
from reljoin.herbench_oracle import choose_oracle_times


def run_static_checks(paper=None) -> None:
    from pathlib import Path

    report = run_merge_order_self_test()
    if report["status"] != "pass":
        raise AssertionError(report)
    spans = [(15.3, 16.2), (68.9, 69.9), (108.1, 108.9), (138.6, 139.7)]
    labeled, _labels, _padded, _mode = label_oracle_times(spans, 152.03, 8)
    official, _official_padded, _official_mode = choose_oracle_times(spans, 152.03, num_frames=8)
    if len(labeled) != len(official) or any(abs(a - b) > 1e-6 for a, b in zip(labeled, official)):
        raise AssertionError(f"sampler labels diverged: {labeled} vs {official}")
    beta = np.zeros((4, 4), dtype=np.float64)
    for receiver in range(4):
        for source in range(4):
            if source != receiver:
                beta[receiver, source] = 0.05
    beta[2, 0] = 0.80
    beta[2, 1] = 0.15
    beta[2] = beta[2] / beta[2].sum()
    for receiver in range(4):
        if receiver == 2:
            continue
        row = beta[receiver]
        beta[receiver] = row / row.sum()
    chosen = select_keyframes(beta)
    edge = max(chosen["edges"], key=lambda item: item["beta"])
    if (edge["receiver"], edge["source"]) != (2, 0):
        raise AssertionError(
            f"arrow j -> i must use beta[i, j]; dominant edge was {edge}, expected receiver 2 source 0"
        )
    if abs(float(beta[2].sum()) - 1.0) > 1e-6:
        raise AssertionError("test beta row does not sum to 1")
    if paper is not None:
        from reljoin.evidence_flow.install import install_anchors_present

        install_anchors_present(Path(paper))
    from reljoin.evidence_flow.render import render_layout_smoke

    render_layout_smoke(Path("/tmp/evidence_flow_layout_smoke"))
    print("PASS: merge-order self-test, sampler labels, beta[i, j] direction, and layout smoke")


def run_intervention_checks() -> None:
    import torch
    import torch.nn as nn

    from reljoin.reljoin_modules import RelJoinResidual, assert_beta_receiver_rows

    torch.manual_seed(0)
    model = RelJoinResidual(32, rel_dim=16, variant="Hybrid")
    nn.init.normal_(model.head.W_o.weight, std=0.05)
    nn.init.constant_(model.head.gate.bias, 2.0)
    model.eval()
    frames, mask, frame_mask, timestamps, question = _batch()
    with torch.no_grad():
        baseline = model(frames, mask, frame_mask, timestamps, question, check_beta=True)
    beta = model.last_beta
    assert beta is not None
    assert_beta_receiver_rows(beta, frame_mask[:, :, None] & frame_mask[:, None, :] & ~torch.eye(4, dtype=torch.bool))
    if float(beta[0, 0, 1]) <= 1e-6:
        raise AssertionError("expected a nonzero unablated beta[0, 1] for the directed-edge test")
    suppress = torch.zeros(4, 6, dtype=torch.bool)
    suppress[0, 0] = True
    with torch.no_grad():
        ablated = model(
            frames, mask, frame_mask, timestamps, question,
            suppress_tokens=suppress, check_beta=True,
        )
    if float(model.last_alpha[0, 0, 0]) != 0.0:
        raise AssertionError("suppressed token still receives pooling mass")
    if abs(float(model.last_alpha[0, 0].sum()) - 1.0) > 1e-4:
        raise AssertionError("alpha no longer sums to 1 after suppression")
    if not torch.allclose(ablated[0, 0, 0], frames[0, 0, 0], atol=1e-5):
        raise AssertionError("suppressed token did not revert to the vanilla token")
    if torch.allclose(baseline[0, 0, 1], frames[0, 0, 1], atol=1e-5):
        raise AssertionError("the unsuppressed residual is unexpectedly zero")
    with torch.no_grad():
        model(
            frames, mask, frame_mask, timestamps, question,
            blocked_edges=[(0, 1)], edge_ablation="pre_softmax", check_beta=True,
        )
    if float(model.last_beta[0, 0, 1]) != 0.0:
        raise AssertionError("pre-softmax edge ablation did not close beta[receiver=0, source=1]")
    if abs(float(model.last_beta[0, 0].sum()) - 1.0) > 1e-4:
        raise AssertionError("pre-softmax ablation did not renormalize the receiver row")
    kept = float(model.last_beta[0, 1, 0])
    with torch.no_grad():
        reference = RelJoinResidual(32, rel_dim=16, variant="Hybrid")
        reference.load_state_dict(model.state_dict())
        reference.eval()
        reference(frames, mask, frame_mask, timestamps, question, check_beta=True)
    if abs(float(reference.last_beta[0, 1, 0]) - kept) > 1e-5:
        raise AssertionError("closing edge 1 -> 0 changed the opposite edge 0 -> 1")
    before = float(reference.last_beta[0, 0, 1])
    with torch.no_grad():
        model(
            frames, mask, frame_mask, timestamps, question,
            blocked_edges=[(0, 1)], edge_ablation="post_softmax", check_beta=False,
        )
    if float(model.last_beta[0, 0, 1]) != 0.0:
        raise AssertionError("post-softmax removal did not zero beta[0, 1]")
    if abs(float(model.last_beta[0, 0].sum()) - (1.0 - before)) > 1e-4:
        raise AssertionError("post-softmax removal renormalized the receiver row")
    print("PASS: suppress_tokens and directed pre/post edge ablation")


def _batch():
    import torch

    frames = torch.randn(1, 4, 6, 32)
    mask = torch.ones(1, 4, 6, dtype=torch.bool)
    frame_mask = torch.ones(1, 4, dtype=torch.bool)
    timestamps = torch.linspace(0, 1, 4).view(1, -1)
    question = torch.randn(1, 32)
    return frames, mask, frame_mask, timestamps, question
