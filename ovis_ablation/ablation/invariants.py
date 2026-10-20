"""CPU invariant tests. They do not load Ovis or score HERBench."""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .common import ensure_v6_on_path
from .config import TASKS, TASK_FAMILIES, TSO_TASK, AblationConfig, load_config
from .modules import (
    ContentGlobalRelation,
    PureLocalRelation,
    build_adapter,
    count_parameters,
    gated_rms_delta,
    select_token_mlp_hidden,
    token_mlp_param_count,
)
from .paths import resolve_paths
from .schedule import build_schedules, build_train_rows, build_val_rows, load_split, uniform_plan

ensure_v6_on_path()

from reljoin.data_perception import deterministic_permute, permute_choices, stable_int_seed  # noqa: E402
from reljoin.losses import batch_reljoin_v5_loss, letter_token_ids, reljoin_v5_loss  # noqa: E402
from reljoin.ovis_hooks import (  # noqa: E402
    VISUAL_ATOM_ID,
    pack_variable_frame_tokens,
    scatter_variable_frame_tokens,
    visual_atom_spans,
)
from reljoin.reljoin_modules import (  # noqa: E402
    GlobalRelation,
    LocalRelation,
    RelJoinResidual,
    masked_softmax,
    pair_mask_from_frames,
    rms_fp32,
)
from reljoin.train import count_scheduler_updates, sample_K  # noqa: E402

from .aggregate import paired_comparison, summarize_predictions  # noqa: E402
from .train import LOSS_FN  # noqa: E402


class InvariantError(AssertionError):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise InvariantError(message)


def _batch(batch: int, frames: int, tokens: int, dim: int, *, valid_frames: int | None = None):
    tokens_x = torch.randn(batch, frames, tokens, dim)
    token_mask = torch.ones(batch, frames, tokens, dtype=torch.bool)
    frame_mask = torch.ones(batch, frames, dtype=torch.bool)
    if valid_frames is not None:
        frame_mask[:, valid_frames:] = False
        token_mask[:, valid_frames:] = False
    timestamps = torch.linspace(0, 1, frames).unsqueeze(0).expand(batch, frames).contiguous()
    question = torch.randn(batch, dim)
    return tokens_x, token_mask, frame_mask, timestamps, question


def _call(module, batch):
    tokens_x, token_mask, frame_mask, timestamps, question = batch
    return module(
        frame_tokens=tokens_x,
        token_mask=token_mask,
        frame_mask=frame_mask,
        timestamps=timestamps,
        question_hidden=question,
    )


def _randomize_output(module) -> None:
    with torch.no_grad():
        module.W_o.weight.normal_(0, 0.2)
        if module.W_o.bias is not None:
            module.W_o.bias.normal_(0, 0.2)


def test_identity_and_single_frame() -> None:
    dim, width = 32, 8
    specs = [
        ("hybrid", {}),
        ("global_only", {}),
        ("local_only", {}),
        ("hybrid_no_time", {}),
        ("token_mlp", {"hidden": 6}),
        ("frame_only", {}),
    ]
    for name, kwargs in specs:
        module = build_adapter(name, dim, width, rms_eps=1e-8, time_eps=1e-8, **kwargs)
        module.eval()
        batch = _batch(2, 3, 4, dim)
        with torch.no_grad():
            output = _call(module, batch)
        _check(torch.equal(output, batch[0]), f"{name} init is not identity")
        _randomize_output(module)
        single = _batch(1, 1, 4, dim)
        with torch.no_grad():
            alone = _call(module, single)
        _check(torch.equal(alone, single[0]), f"{name} K=1 changed after nonzero head")
        padded = _batch(1, 3, 4, dim, valid_frames=1)
        with torch.no_grad():
            padded_out = _call(module, padded)
        _check(torch.equal(padded_out[:, 0], padded[0][:, 0]), f"{name} single valid frame changed")
    _check(build_adapter("vanilla", dim, width) is None, "vanilla should not build an adapter")
    print("PASS: identity at init and K=1 after nonzero head")


def test_text_positions_unchanged() -> None:
    dim = 16
    module = build_adapter("token_mlp", dim, 8, hidden=4)
    _randomize_output(module)
    atom = VISUAL_ATOM_ID
    ids = torch.tensor([[5, 6, atom, atom, atom, 7, atom, atom, 8]])
    embeds = torch.randn(1, ids.shape[1], dim)
    spans = visual_atom_spans(ids, atom_id=atom)
    groups = [embeds[:, start:end, :] for start, end in spans]
    packed, token_mask = pack_variable_frame_tokens(groups)
    frames = packed.shape[1]
    updated = module(
        frame_tokens=packed,
        token_mask=token_mask,
        frame_mask=torch.ones(1, frames, dtype=torch.bool),
        timestamps=torch.linspace(0, 1, frames).view(1, frames),
        question_hidden=torch.randn(1, dim),
    )
    restored = scatter_variable_frame_tokens(embeds.clone(), spans, updated)
    for index, token_id in enumerate(ids[0].tolist()):
        if int(token_id) != int(atom):
            _check(torch.equal(restored[0, index], embeds[0, index]), f"text/special position {index} changed")
    _check(restored.shape == embeds.shape, "sequence length changed")
    print("PASS: text and special positions unchanged")


def test_token_independence() -> None:
    dim = 24
    module = build_adapter("token_mlp", dim, 8, hidden=5)
    module.eval()
    with torch.no_grad():
        module.B_x.weight.normal_(0, 0.3)
        module.B_o.weight.normal_(0, 0.3)
        module.gate.weight.normal_(0, 0.3)
    tokens, token_mask, frame_mask, timestamps, question = _batch(1, 3, 5, dim)
    token_mask = token_mask.clone()
    token_mask[:, :, -1] = False
    with torch.no_grad():
        base = module(tokens, token_mask, frame_mask, timestamps, question)
        other = tokens.clone()
        other[:, 0, 1:] += 1.5
        other[:, 1:] += 2.0
        other[:, 0, 0] = tokens[:, 0, 0]
        changed_others = module(other, token_mask, frame_mask, timestamps, question)
        own = tokens.clone()
        own[:, 0, 0] += 0.7
        changed_own = module(own, token_mask, frame_mask, timestamps, question)
    _check(torch.equal(changed_others[:, 0, 0], base[:, 0, 0]), "token_mlp changed when other tokens changed")
    _check(not torch.allclose(changed_own[:, 0, 0], base[:, 0, 0]), "token_mlp ignored its own token")
    _check(torch.equal(base[:, 0, -1], tokens[:, 0, -1]), "masked token was updated")
    print("PASS: token_mlp independence")


def test_call_graph_and_summaries() -> None:
    dim, width = 32, 8
    global_only = build_adapter("global_only", dim, width)
    local_only = build_adapter("local_only", dim, width)
    no_time = build_adapter("hybrid_no_time", dim, width)
    hybrid = build_adapter("hybrid", dim, width)
    _check(not any(isinstance(mod, LocalRelation) for mod in global_only.modules()), "global_only owns LocalRelation")
    _check(not any("local" in name for name, _ in global_only.named_parameters()), "global_only has local parameters")
    _check(not any(isinstance(mod, GlobalRelation) for mod in local_only.modules()), "local_only owns GlobalRelation")
    _check(not any(isinstance(mod, LocalRelation) for mod in local_only.modules()), "local_only uses canonical LocalRelation")
    banned = ("global_relation", "A_s", "A_r", "local_global", "head_global", "A_K", "A_V", "a_b")
    local_names = [name for name, _ in local_only.named_parameters()]
    for piece in banned:
        _check(not any(piece in name for name in local_names), f"local_only still has {piece}")
    _check(any(isinstance(mod, LocalRelation) for mod in hybrid.modules()), "hybrid lost LocalRelation")
    _check(any(isinstance(mod, GlobalRelation) for mod in hybrid.modules()), "hybrid lost GlobalRelation")
    _check(type(hybrid) is RelJoinResidual and hybrid.variant == "Hybrid", "hybrid is not the canonical class")
    no_time_names = [name for name, _ in no_time.named_parameters()]
    for piece in ("abs_position", "A_K", "A_V", "a_b"):
        _check(not any(piece in name for name in no_time_names), f"hybrid_no_time still has {piece}")
    removed = count_parameters(hybrid) - count_parameters(no_time)
    _check(removed == 10 * width + 4, f"time-parameter removal was {removed}, expected {10 * width + 4}")

    def perturb(module, frame_index: int):
        original = module.frame_pooler.forward

        def wrapped(*args, **kwargs):
            summary, alpha, values = original(*args, **kwargs)
            summary = summary.clone()
            summary[:, frame_index] = summary[:, frame_index] + 1.7
            return summary, alpha, values

        module.frame_pooler.forward = wrapped
        return original

    batch = _batch(1, 3, 4, dim)
    for module, expect_frame0_change in ((local_only, False), (global_only, True)):
        module.eval()
        with torch.no_grad():
            for param in module.parameters():
                if param.ndim >= 1:
                    param.normal_(0, 0.15)
            module.W_o.weight.normal_(0, 0.3)
        with torch.no_grad():
            before = _call(module, batch)
        restore = perturb(module, 1)
        try:
            with torch.no_grad():
                after = _call(module, batch)
        finally:
            module.frame_pooler.forward = restore
        frame0_same = torch.allclose(before[:, 0], after[:, 0], atol=1e-5, rtol=1e-5)
        frame1_same = torch.allclose(before[:, 1], after[:, 1], atol=1e-5, rtol=1e-5)
        _check(not frame1_same, f"{module.variant} ignored a source-summary perturbation")
        if expect_frame0_change:
            _check(not frame0_same, "global_only did not read the other frame summary")
        else:
            _check(frame0_same, "local_only read the target frame summary")
        _check("local" not in global_only.forward_trace or module is not global_only, "trace")
    fresh_global = build_adapter("global_only", dim, width)
    fresh_local = build_adapter("local_only", dim, width)
    _call(fresh_global, batch)
    _call(fresh_local, batch)
    _check("local" not in fresh_global.forward_trace, f"global_only trace {fresh_global.forward_trace}")
    _check("global" not in fresh_local.forward_trace, f"local_only trace {fresh_local.forward_trace}")
    _check("global" in fresh_global.forward_trace and "local" in fresh_local.forward_trace, "missing expected branch")
    print("PASS: call graph and summary isolation")


def test_no_time() -> None:
    dim, width = 28, 8
    module = build_adapter("hybrid_no_time", dim, width)
    module.eval()
    with torch.no_grad():
        for param in module.parameters():
            param.normal_(0, 0.2)
    tokens, token_mask, frame_mask, timestamps, question = _batch(1, 4, 3, dim)
    other_time = timestamps + torch.tensor([[0.2, -0.4, 1.5, 3.0]])
    with torch.no_grad():
        first = module(tokens, token_mask, frame_mask, timestamps, question)
        second = module(tokens, token_mask, frame_mask, other_time, question)
    _check(torch.allclose(first, second, atol=1e-6, rtol=1e-6), "hybrid_no_time changed with timestamps")
    _check("time" not in module.forward_trace, f"no-time trace used time: {module.forward_trace}")
    stamped = timestamps.clone().requires_grad_(True)
    output = module(tokens, token_mask, frame_mask, stamped, question)
    output.sum().backward()
    _check(stamped.grad is None, "hybrid_no_time backpropped into timestamps")

    import ablation.modules as adapter_mod

    original = adapter_mod.normalize_time_and_rank

    def shifted(timestamps_in, frame_mask_in, eps=1e-8):
        tau, rho = original(timestamps_in, frame_mask_in, eps=eps)
        return tau + 0.4, rho + 0.25

    global_only = build_adapter("global_only", dim, width)
    global_only.eval()
    with torch.no_grad():
        global_only.abs_position.weight.normal_(0, 0.4)
        global_only.global_relation.A_K.weight.normal_(0, 0.4)
        global_only.W_o.weight.normal_(0, 0.4)
        before = _call(global_only, (tokens, token_mask, frame_mask, timestamps, question))
    adapter_mod.normalize_time_and_rank = shifted
    try:
        with torch.no_grad():
            after_global = _call(global_only, (tokens, token_mask, frame_mask, timestamps, question))
            after_notime = module(tokens, token_mask, frame_mask, timestamps, question)
    finally:
        adapter_mod.normalize_time_and_rank = original
    _check(not torch.allclose(before, after_global, atol=1e-5, rtol=1e-5), "global_only ignored time metadata")
    _check(torch.allclose(first, after_notime, atol=1e-5, rtol=1e-5), "no-time followed the patched time function")
    print("PASS: hybrid_no_time ignores explicit time metadata")


def test_frame_only_structure_and_chi() -> None:
    dim, width = 24, 8
    module = build_adapter("frame_only", dim, width)
    module.eval()
    relations = (GlobalRelation, LocalRelation, ContentGlobalRelation, PureLocalRelation)
    _check(not any(isinstance(mod, relations) for mod in module.modules()), "frame_only owns a relation module")
    names = [name for name, _ in module.named_parameters()]
    for piece in ("global", "local", "A_K", "A_V", "a_b"):
        _check(not any(piece in name for name in names), f"frame_only still has {piece}")
    expected = 4 * dim * width + 3 * width * width + 3 * dim + 9 * width + 1
    _check(count_parameters(module) == expected, f"frame_only has {count_parameters(module)} params, expected {expected}")
    _check(not bool(module.W_o.weight.any()) and not bool(module.W_o.bias.any()), "frame_only W_o is not zero at init")
    _check(not bool(module.head.gate.weight.any()) and not bool(module.head.gate.bias.any()), "gate does not start at zero")

    import ablation.modules as adapter_mod

    original = adapter_mod.relative_time_features

    def forbidden(*args, **kwargs):
        raise InvariantError("frame_only called relative_time_features")

    adapter_mod.relative_time_features = forbidden
    try:
        with torch.no_grad():
            _call(module, _batch(1, 3, 4, dim))
    finally:
        adapter_mod.relative_time_features = original
    _check(module.forward_trace == ["pool", "time", "head"], f"frame_only trace {module.forward_trace}")
    _check(module.last_stats["gate_mean"] == 0.5, f"init gate mean {module.last_stats['gate_mean']} != 0.5")

    # Sample 0: three valid frames, one padded frame, one padded token. Sample 1: one valid frame.
    _randomize_output(module)
    tokens, token_mask, frame_mask, timestamps, question = _batch(2, 4, 5, dim)
    frame_mask[0, 3] = False
    token_mask[0, 3] = False
    token_mask[0, 0, -1] = False
    frame_mask[1, 1:] = False
    token_mask[1, 1:] = False
    with torch.no_grad():
        out = module(tokens, token_mask, frame_mask, timestamps, question)
    structural = pair_mask_from_frames(frame_mask).any(dim=-1)
    _check(bool(structural[0, :3].all()) and not bool(structural[0, 3]) and not bool(structural[1].any()), "fixture lost its chi pattern")
    _check(torch.equal(module.last_has_neighbor, structural), "frame_only chi is not the valid-frame neighbor mask")
    moved = module.last_delta_x.abs().amax(dim=-1) > 0
    _check(torch.equal(moved, token_mask & structural.unsqueeze(-1)), "residual is not live exactly where m_ip chi_i = 1")
    _check(torch.equal(out[1], tokens[1]), "single valid frame changed")
    _check(torch.equal(out[0, 3], tokens[0, 3]), "padded frame changed")
    _check(torch.equal(out[0, 0, -1], tokens[0, 0, -1]), "padded token changed")
    print("PASS: frame_only has no relation branch, gate 0.5 at init, chi from the valid-frame count")


def test_frame_only_frame_isolation() -> None:
    dim, width = 24, 8
    torch.manual_seed(3)
    module = build_adapter("frame_only", dim, width)
    module.eval()
    tokens, token_mask, frame_mask, _timestamps, question = _batch(1, 4, 6, dim)
    token_mask[:, 2, -2:] = False
    timestamps = torch.tensor([[0.0, 1.5, 4.0, 9.0]])
    frames = tokens.shape[1]
    for stage in ("nonzero W_o", "all parameters random"):
        with torch.no_grad():
            if stage == "nonzero W_o":
                _randomize_output(module)
            else:
                for param in module.parameters():
                    param.normal_(0, 0.2)
            base = module(tokens, token_mask, frame_mask, timestamps, question)
            base_delta = module.last_delta_x.clone()
        _check(bool(base_delta.any()), f"frame_only residual is zero with {stage}")
        for j in range(frames):
            swapped = tokens.clone()
            swapped[:, j] = 3.0 * torch.randn_like(swapped[:, j])
            with torch.no_grad():
                out = module(swapped, token_mask, frame_mask, timestamps, question)
            for i in range(frames):
                if i != j:
                    _check(torch.equal(out[:, i], base[:, i]), f"frame {i} changed with frame {j} content ({stage})")
            _check(not torch.equal(module.last_delta_x[:, j], base_delta[:, j]), f"frame {j} residual ignored its own content ({stage})")
    # Moving frame 1 inside (t_min, t_max) keeps the other frames' tau and rho, so only frame 1 may move.
    shifted = timestamps.clone()
    shifted[0, 1] = 2.5
    with torch.no_grad():
        moved = module(tokens, token_mask, frame_mask, shifted, question)
        other_question = module(tokens, token_mask, frame_mask, timestamps, torch.randn_like(question))
    _check(not torch.equal(moved[:, 1], base[:, 1]), "frame_only ignored its own absolute time")
    for i in (0, 2, 3):
        _check(torch.equal(moved[:, i], base[:, i]), f"frame {i} changed with the timestamp of frame 1")
    _check(not torch.equal(other_question, base), "frame_only ignored the question")
    print("PASS: frame_only output for frame i is invariant to the content and time of frame j != i")


def test_softmax_and_rms() -> None:
    logits = torch.randn(2, 3, 4)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    probs = masked_softmax(logits, mask, dim=-1)
    _check(bool(torch.isfinite(probs).all()), "all-masked softmax produced non-finite values")
    _check(torch.equal(probs, torch.zeros_like(probs)), "all-masked softmax was not zero")
    frame_mask = torch.ones(1, 4, dtype=torch.bool)
    pair = pair_mask_from_frames(frame_mask)
    _check(not bool(torch.diagonal(pair[0]).any()), "pair mask kept self edges")
    values = torch.randn(2, 3, 4, 5)
    update = torch.randn(2, 3, 4, 5)
    gate = torch.sigmoid(torch.randn(2, 3, 4, 1))
    token_mask = torch.ones(2, 3, 4, dtype=torch.bool)
    neighbors = torch.ones(2, 3, dtype=torch.bool)
    delta, _scale, sx = gated_rms_delta(values, update, gate, token_mask, neighbors, 1e-8)
    delta_rms = rms_fp32(delta, dim=-1, keepdim=True)
    _check(bool(torch.all(delta_rms <= sx + 1e-4)), "RMS calibration exceeded the input RMS")
    tiny = torch.full_like(values, 1e-12)
    tiny_delta, _, _ = gated_rms_delta(tiny, update, gate, token_mask, neighbors, 1e-8)
    _check(torch.equal(tiny_delta, torch.zeros_like(tiny_delta)), "tiny input RMS still updated")
    print("PASS: masked softmax and RMS bound")


def test_two_optimizer_steps() -> None:
    specs = [
        ("hybrid", "global_relation.W_Q.weight", {}),
        ("global_only", "global_relation.W_Q.weight", {}),
        ("local_only", "local.local_q.weight", {}),
        ("hybrid_no_time", "global_relation.W_Q.weight", {}),
        ("token_mlp", "B_x.weight", {"hidden": 5}),
        ("frame_only", "frame_pooler.token_key.weight", {}),
    ]
    dim = 20
    batch = _batch(1, 3, 4, dim)
    for name, upstream_name, kwargs in specs:
        torch.manual_seed(0)
        module = build_adapter(name, dim, 8, **kwargs)
        module.train()
        upstream = dict(module.named_parameters())[upstream_name]
        head = module.W_o.weight
        optimizer = AdamW(module.parameters(), lr=2e-4, weight_decay=0.01)

        def step() -> None:
            optimizer.zero_grad(set_to_none=True)
            output = _call(module, batch)
            output.float().pow(2).mean().backward()

        step()
        _check(head.grad is not None and bool(torch.isfinite(head.grad).all()), f"{name} head grad missing")
        _check(float(head.grad.abs().sum()) > 0, f"{name} head grad is zero at step 1")
        upstream_first = 0.0 if upstream.grad is None else float(upstream.grad.abs().sum())
        _check(upstream_first == 0.0, f"{name} upstream grad {upstream_first} before W_o stepped")
        optimizer.step()
        step()
        _check(upstream.grad is not None and bool(torch.isfinite(upstream.grad).all()), f"{name} upstream non-finite")
        _check(float(upstream.grad.abs().sum()) > 0, f"{name} upstream grad stayed zero after step 2")
    print("PASS: two optimizer steps unblock upstream gradients")


def test_budget_schedule_and_loss(cfg: AblationConfig) -> None:
    small = build_adapter("token_mlp", 32, 8, hidden=5)
    _check(count_parameters(small) == token_mlp_param_count(32, 8, 5), "token_mlp formula mismatch")
    budget = select_token_mlp_hidden(cfg.hidden_size, cfg.rel_dim, cfg.rms_eps, cfg.time_eps)
    _check(budget["rel_diff"] <= 0.01, f"token_mlp budget gap {budget['rel_diff']}")
    print(
        f"PASS: params hybrid={budget['hybrid_params']} token_mlp={budget['token_mlp_params']} "
        f"H={budget['hidden']} rel_diff={budget['rel_diff']:.6f}"
    )
    resolved = resolve_paths(cfg)
    if resolved.train_manifest and resolved.val_manifest:
        train = load_split(resolved.train_manifest)
        val = load_split(resolved.val_manifest)
        bundle_a = build_schedules(cfg, resolved, probe_frames=False)
        torch.manual_seed(1)
        random.seed(2)
        _ = build_adapter("hybrid", 40, 8)
        _ = build_adapter("token_mlp", 48, 8, hidden=3)
        random.seed(99)
        bundle_b = build_schedules(cfg, resolved, probe_frames=False)
        _check(bundle_a["meta"]["schedule_sha256"] == bundle_b["meta"]["schedule_sha256"], "schedule depends on model RNG")
        _check(len(bundle_a["train_rows"]) == cfg.max_epochs * len(train), "train schedule length")
        _check(len(bundle_a["val_rows"]) == len(val), "val schedule length")
        expected = build_train_rows(train, cfg.seed, cfg.max_epochs)
        for got, ref in zip(bundle_a["train_rows"], expected):
            _check(got["K"] == ref["K"] and got["permutation"] == ref["permutation"], "train schedule drifted from sample_K")
            _check(got["answer_letter"] == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[got["answer_index"]], "remapped letter")
        direct = build_val_rows(val)
        for sample, row in zip(val, direct):
            choices, index, perm = deterministic_permute(
                list(sample["choices"]), int(sample["answer_index"]), f"val::{sample['sample_id']}"
            )
            k = min(8, int(sample_K(random.Random(stable_int_seed(str(sample["sample_id"]))))))
            _check(row["K"] == k and row["permutation"] == perm and row["choices"] == choices, "val policy drifted")
            _check(row["answer_index"] == index, "val label remap drifted")
            _check(choices[index] == list(sample["choices"])[int(sample["answer_index"])], "val permutation lost the answer")
        _check(bundle_a["train_rows"][0]["order_index"] == 0, "question order is not manifest order")
        updates, _total, _warm = count_scheduler_updates(
            len(train), cfg.gradient_accumulation_steps, cfg.max_epochs, cfg.warmup_ratio
        )
        _check(updates == 57, f"expected 57 updates/epoch for 454 samples, got {updates}")
        flushed = [
            index
            for index in range(len(train))
            if (index + 1) % cfg.gradient_accumulation_steps == 0 or (index + 1) == len(train)
        ]
        _check(len(flushed) == 57 and flushed[-1] == len(train) - 1, "tail accumulation drops samples")
    else:
        print("SKIP: V6 split is written at prepare after PerceptionTest is downloaded")

    _check(LOSS_FN is batch_reljoin_v5_loss, "training loss is not the canonical V6 function")
    scores = torch.tensor([[0.4, -0.1, 0.2, 0.0]])
    vanilla = torch.tensor([[0.2, 0.3, -0.4, 0.1]])
    target = torch.tensor([2])
    loss, cand, corr = reljoin_v5_loss(scores, vanilla, target)
    manual_c = F.cross_entropy(scores, target, reduction="none")
    manual_r = F.cross_entropy(scores - vanilla, target, reduction="none")
    manual = torch.logsumexp(torch.stack([manual_c, manual_r], dim=-1), dim=-1) - math.log(2)
    _check(torch.allclose(loss, manual) and torch.allclose(cand, manual_c), "loss combination drifted")
    _check(torch.allclose(LOSS_FN(scores, vanilla, target)[0], manual.mean()), "batch reduction drifted")

    class _Tok:
        unk_token_id = 0

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [ord(text[0])]

    ids = letter_token_ids(_Tok(), 3)
    _check(ids.tolist() == [ord("A"), ord("B"), ord("C")], "letter ids hardcoded the choice count")
    times, indices = uniform_plan(25.0, 100, 4)
    _check(len(times) == 4 and len(indices) == 4, "uniform plan length")
    _check(indices == [min(99, max(0, int(round(t * 25.0)))) for t in times], "index formula drifted")
    from reljoin.herbench_oracle import choose_oracle_times, extract_evidence_spans

    _spans, hint = extract_evidence_spans({"timestamps": "all"})
    uniform_times, _padded, uniform_mode = choose_oracle_times([], 12.0, 8)
    _check(hint == "uniform_fallback" and uniform_mode == "uniform_fallback", "RLPC did not fall back to uniform frames")
    _check(len(uniform_times) == 8, "uniform fallback did not return 8 frames")
    from .eval_herbench import eval_prediction_dir

    smoke_dir, smoke = eval_prediction_dir(cfg, 2)
    full_dir, full = eval_prediction_dir(cfg, None)
    _check(smoke and smoke_dir.parent.name == "eval_smoke", "eval smoke directory drifted")
    _check((not full) and full_dir.parent.name == "eval", "full eval directory drifted")
    _check(smoke_dir != full_dir, "smoke and full eval share a directory")
    print("PASS: shared schedule, loss, and parameter budget")


def test_aggregate_definitions() -> None:
    rows = []
    # TSO: 1/1. Task B (AC): 0/2. Other 10 tasks: 1/1.
    rows.append(_fake("q0", TSO_TASK, True))
    rows.append(_fake("q1", "Action Counting", False))
    rows.append(_fake("q2", "Action Counting", False))
    others = [task for task in TASKS if task not in {TSO_TASK, "Action Counting"}]
    for index, task in enumerate(others):
        rows.append(_fake(f"q{index+3}", task, True))
    summary = summarize_predictions(rows, variant="vanilla", seed=42, expected_questions=len(rows))
    non_tso_tasks = [task for task in TASKS if task != TSO_TASK]
    macro_non = sum(summary["per_task"][task]["accuracy"] for task in non_tso_tasks) / 11
    micro_non = 10 / 12
    _check(abs(summary["non_tso_avg"] - macro_non) < 1e-9, "Non-TSO is not a macro mean")
    _check(abs(macro_non - micro_non) > 1e-6, "fixture does not separate macro and micro")
    _check(abs(summary["overall_avg"] - (1.0 + 10 * 1.0) / 12) < 1e-9, "Overall is not the 12-task macro")
    _check(abs(summary["qa_accuracy"] - (11 / 13)) < 1e-9, "QA is not micro accuracy")
    _check(summary["families"]["TR&C"] is not None, "TR&C missing")
    family = TASK_FAMILIES["ME&N"]
    expected_family = sum(summary["per_task"][task]["accuracy"] for task in family) / 3
    _check(abs(summary["families"]["ME&N"] - expected_family) < 1e-9, "ME&N is not a task macro")
    print("PASS: aggregate macro definitions")


def test_paired_report() -> None:
    # One question per task over three videos. Hybrid misses Action Counting, frame_only misses TSO.
    rows = []
    for index, task in enumerate(TASKS):
        video = f"v{index % 3}"
        rows.append({**_fake(f"p{index}", task, task != "Action Counting"), "variant": "hybrid", "video_id": video})
        rows.append({**_fake(f"p{index}", task, task != TSO_TASK), "variant": "frame_only", "video_id": video})
        rows.append({**_fake(f"p{index}", task, False), "variant": "token_mlp", "video_id": video})
    rows.append({**_fake("p_err", TSO_TASK, True), "variant": "hybrid", "video_id": "v0"})
    rows.append({**_fake("p_err", TSO_TASK, False), "variant": "frame_only", "video_id": "v0", "status": "runtime_error"})
    report = paired_comparison(rows, variant="frame_only", reference="hybrid", n_resamples=500, seed=42)
    _check(report["n_paired"] == len(TASKS), "pairing kept a question that failed for one variant")
    qa, tso, non = (report["tests"][name] for name in ("QA", "TSO", "Non-TSO questions"))
    _check(
        (qa["variant_only_correct"], qa["reference_only_correct"], qa["ties"]) == (1, 1, len(TASKS) - 2),
        "QA discordant counts drifted",
    )
    _check((tso["n"], tso["variant_only_correct"], tso["reference_only_correct"]) == (1, 0, 1), "TSO subset drifted")
    _check(
        (non["n"], non["variant_only_correct"], non["reference_only_correct"]) == (len(TASKS) - 1, 1, 0),
        "Non-TSO question subset drifted",
    )
    _check(abs(qa["bootstrap"]["delta"]) < 1e-12, "QA delta drifted")
    _check(abs(report["macro"]["overall_avg"]["delta"]) < 1e-12, "Overall macro delta drifted")
    _check(abs(report["macro"]["non_tso_avg"]["delta"] - 1 / 11) < 1e-12, "Non-TSO macro delta drifted")
    again = paired_comparison(rows, variant="frame_only", reference="hybrid", n_resamples=500, seed=42)
    _check(again["tests"]["QA"]["bootstrap"] == qa["bootstrap"], "video bootstrap is not reproducible")
    boot = non["bootstrap"]
    _check(boot["n_videos"] == 3 and boot["ci_low"] <= boot["delta"] <= boot["ci_high"], "video bootstrap CI drifted")
    clash = [dict(row) for row in rows]
    clash[1]["gold_letter"] = "B"
    try:
        paired_comparison(clash, variant="frame_only", reference="hybrid", n_resamples=10)
    except ValueError:
        pass
    else:
        raise InvariantError("paired comparison accepted rows from different eval manifests")
    print("PASS: paired comparison pairs by question_id and resamples whole videos")


def _fake(question_id: str, task: str, correct: bool) -> dict:
    return {
        "question_id": question_id,
        "video_id": "v",
        "task": task,
        "gold_letter": "A",
        "predicted_letter": "A" if correct else "B",
        "raw_response": "A" if correct else "B",
        "correct": correct,
        "valid": True,
        "variant": "vanilla",
        "seed": 42,
        "checkpoint_id": "vanilla",
        "status": "ok",
    }


def run_invariants(cfg: AblationConfig | None = None) -> dict:
    cfg = cfg or load_config()
    test_identity_and_single_frame()
    test_text_positions_unchanged()
    test_token_independence()
    test_call_graph_and_summaries()
    test_no_time()
    test_frame_only_structure_and_chi()
    test_frame_only_frame_isolation()
    test_softmax_and_rms()
    test_two_optimizer_steps()
    test_budget_schedule_and_loss(cfg)
    test_aggregate_definitions()
    test_paired_report()
    budget = select_token_mlp_hidden(cfg.hidden_size, cfg.rel_dim, cfg.rms_eps, cfg.time_eps)
    return {"ok": True, "token_mlp": budget}
