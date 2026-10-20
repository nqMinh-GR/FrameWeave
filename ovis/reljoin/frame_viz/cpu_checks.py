"""CPU checks for the visualization collector, geometry, and offline renderer.

GPU smoke is recorded as not_run. A passing CPU report is not a Colab smoke pass.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from .collector import FrameVizSession, active_session
from .geometry import (
    build_geometry_from_views,
    paint_frame_map,
    resolve_runtime_geometry,
    routing_components,
    token_bbox_on_original,
)
from .queries import select_queries
from .render import render_example
from ..reljoin_modules import (
    RelJoinResidual,
    _PI_CHUNK_SINKS,
    block_read_sources,
    pair_mask_from_frames,
)


def run_cpu_checks(output_dir: Path | None = None) -> dict:
    tests = []

    def check(name, fn):
        fn()
        tests.append({"name": name, "status": "pass"})
        print(f"PASS: {name}", flush=True)

    check("collector parity and pi conservation", _test_collector_parity)
    check("chunked pi matches the full token loop", _test_chunk_pi)
    check("empty-neighbor rows stay zero", _test_empty_neighbor)
    check("read-source block clears only other receivers", _test_block_sources)
    check("collector lifecycle clears sinks", _test_lifecycle)
    check("eta uses the delivered correction", _test_effective_eta)
    check("hook debug on and off match", _test_hook_parity)
    check("rectangular grid, letterbox, and padding", _test_grid_geometry)
    check("multi-crop mass is conserved and thumbnails are separate", _test_multicrop_mass)
    check("tokenizer probe verifies only a confirmed resize", _test_runtime_probe)
    check("offline render from a dump", _test_offline_render)
    check("top-eta selection is deterministic", _test_query_selection)
    check("survey quotas ignore heatmap sharpness", _test_survey_selection)
    report = {
        "cpu_checks": {"status": "pass", "tests": tests},
        "gpu_smoke": {
            "status": "not_run",
            "reason": "CPU fixtures do not load Ovis, Hybrid.pt, or HERBench videos.",
        },
    }
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        dest = path / "validation_report.json"
        current = {}
        if dest.is_file():
            current = json.loads(dest.read_text(encoding="utf-8"))
        current.update(report)
        dest.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return report


def _inputs(k=3, p=6, d=32, seed=0):
    import torch

    torch.manual_seed(seed)
    frame_tokens = torch.randn(1, k, p, d)
    token_mask = torch.ones(1, k, p, dtype=torch.bool)
    token_mask[:, :, -1] = False
    frame_mask = torch.tensor([[True, True, False]])
    if k != 3:
        frame_mask = torch.ones(1, k, dtype=torch.bool)
    timestamps = torch.arange(k, dtype=torch.float32).view(1, k)
    question = torch.randn(1, d)
    return frame_tokens, token_mask, frame_mask, timestamps, question


def _forward(module, tensors):
    import torch

    frame_tokens, token_mask, frame_mask, timestamps, question = tensors
    with torch.no_grad():
        return module(
            frame_tokens, token_mask, frame_mask, timestamps, question, eval_mode="full"
        )


def _test_collector_parity():
    import torch

    tensors = _inputs()
    module = RelJoinResidual(32, rel_dim=8, variant="Hybrid", local_token_chunk=2)
    module.eval()
    with torch.no_grad():
        before = _forward(module, tensors)
        alpha = module.last_alpha.clone()
        beta = module.last_beta.clone()
        gamma = module.last_gamma.clone()
        delta = module.last_delta_x.clone()
        with FrameVizSession(query_indices=[(0, 1), (1, 0)], collect_pi=True) as session:
            during = _forward(module, tensors)
            session.record_delivered(
                frame_tokens=tensors[0],
                delivered=during,
                token_mask=tensors[1],
                frame_mask=tensors[2],
                module=module,
                timestamps=[0.0, 1.0, 2.0],
            )
            bundle = session.export()
        after = _forward(module, tensors)
    if not torch.equal(before, during) or not torch.equal(during, after):
        raise AssertionError("collector changed the Hybrid forward")
    if not torch.allclose(bundle["alpha"], alpha.float().cpu()):
        raise AssertionError("alpha copy drifted")
    if not torch.allclose(bundle["beta"], beta.float().cpu()):
        raise AssertionError("beta copy drifted")
    if not torch.allclose(bundle["gamma"], gamma.float().cpu()):
        raise AssertionError("gamma copy drifted")
    if float(bundle["dtype_gap_max_abs"]) > 1e-6:
        raise AssertionError("float32 delivery should match the module delta")
    if not torch.allclose(bundle["module_delta_rms"], _rms(delta.float().cpu())):
        raise AssertionError("module delta rms was not the pre-cast residual")
    _assert_routing(bundle, tensors[1], tensors[2])
    if not torch.isfinite(bundle["eta"]).all():
        raise AssertionError("eta has non-finite values")


def _rms(value):
    import torch

    return torch.sqrt(value.float().pow(2).mean(dim=-1).clamp_min(0))


def _assert_routing(bundle, token_mask, frame_mask):
    pi = bundle["pi_selected"][0]
    gamma = bundle["gamma_selected"][0]
    queries = bundle["query_indices"]
    mask = token_mask[0].cpu()
    frames = frame_mask[0].cpu()
    pair = pair_mask_from_frames(frame_mask.cpu())[0]
    for index, (frame_i, token_p) in enumerate(queries.tolist()):
        if not bool(mask[int(frame_i), int(token_p)]):
            continue
        gamma_row = gamma[index].numpy()
        pi_row = pi[index].numpy()
        if not bool(frames[int(frame_i)].item()) or not bool(pair[int(frame_i)].any().item()):
            if np.max(np.abs(gamma_row)) > 1e-5 or np.max(np.abs(pi_row)) > 1e-5:
                raise AssertionError("empty-neighbor query was renormalized")
            continue
        if abs(float(gamma_row.sum()) - 1.0) > 1e-4:
            raise AssertionError(f"gamma row sum {gamma_row.sum()}")
        for sender in range(pi_row.shape[0]):
            sender_mask = mask[sender].numpy()
            token_sum = float(pi_row[sender, sender_mask].sum())
            if bool(pair[int(frame_i), sender].item()):
                if abs(token_sum - 1.0) > 1e-4:
                    raise AssertionError(f"pi sum {token_sum} for sender {sender}")
            elif token_sum > 1e-4:
                raise AssertionError("pi leaked onto a masked sender")
            if float(np.abs(pi_row[sender, ~sender_mask]).sum()) > 1e-4:
                raise AssertionError("pi leaked onto padding")
        parts = routing_components(
            gamma_row, pi_row, mask.numpy(), np.ones_like(mask.numpy(), dtype=bool)
        )
        if np.max(np.abs(parts["per_frame_mass"] - gamma_row)) > 1e-4:
            raise AssertionError("R does not sum to gamma")
        if abs(parts["total_mass"] - 1.0) > 1e-4:
            raise AssertionError("valid routing mass is not 1")


def _test_chunk_pi():
    import torch

    tensors = _inputs()
    full = RelJoinResidual(32, rel_dim=8, variant="Hybrid", local_token_chunk=0)
    chunked = RelJoinResidual(32, rel_dim=8, variant="Hybrid", local_token_chunk=2)
    chunked.load_state_dict(full.state_dict())
    full.eval()
    chunked.eval()
    queries = [(0, 1), (1, 2)]
    with torch.no_grad():
        with FrameVizSession(query_indices=queries, collect_pi=True) as left:
            y_full = _forward(full, tensors)
            left.record_delivered(
                frame_tokens=tensors[0], delivered=y_full, token_mask=tensors[1],
                frame_mask=tensors[2], module=full, timestamps=None,
            )
            full_bundle = left.export()
        with FrameVizSession(query_indices=queries, collect_pi=True) as right:
            y_chunk = _forward(chunked, tensors)
            right.record_delivered(
                frame_tokens=tensors[0], delivered=y_chunk, token_mask=tensors[1],
                frame_mask=tensors[2], module=chunked, timestamps=None,
            )
            chunk_bundle = right.export()
    if not torch.allclose(y_full, y_chunk, atol=1e-5):
        raise AssertionError("chunked forward diverged")
    if not torch.allclose(full_bundle["pi_selected"], chunk_bundle["pi_selected"], atol=1e-5):
        raise AssertionError("chunked pi rows diverged from the full loop")
    if not all(chunk_bundle["query_captured"]):
        raise AssertionError("a query row was missed inside the chunk loop")


def _test_empty_neighbor():
    import torch

    torch.manual_seed(1)
    frame_tokens = torch.randn(1, 1, 4, 16)
    token_mask = torch.ones(1, 1, 4, dtype=torch.bool)
    frame_mask = torch.ones(1, 1, dtype=torch.bool)
    timestamps = torch.zeros(1, 1)
    question = torch.randn(1, 16)
    module = RelJoinResidual(16, rel_dim=8, variant="Hybrid")
    module.eval()
    with torch.no_grad():
        out = module(frame_tokens, token_mask, frame_mask, timestamps, question)
    if not torch.equal(out, frame_tokens):
        raise AssertionError("K=1 must stay at the original tokens")
    if float(module.last_beta.abs().max()) > 0 or float(module.last_gamma.abs().max()) > 0:
        raise AssertionError("K=1 relations must stay zero")
    if float(module.last_delta_x.abs().max()) > 0:
        raise AssertionError("K=1 residual must stay zero")


def _test_block_sources():
    import torch

    tensors = _inputs(k=3, p=4, d=16, seed=2)
    frame_tokens, token_mask, _frame_mask, timestamps, question = tensors
    frame_mask = torch.ones(1, 3, dtype=torch.bool)
    module = RelJoinResidual(16, rel_dim=8, variant="Hybrid")
    module.eval()
    with torch.no_grad():
        module(frame_tokens, token_mask, frame_mask, timestamps, question)
        open_beta = module.last_beta.clone()
        module(
            frame_tokens, token_mask, frame_mask, timestamps, question,
            blocked_sources=(1,),
        )
        blocked = module.last_beta.clone()
    pair = pair_mask_from_frames(frame_mask)
    again = block_read_sources(pair, (1,))
    if torch.equal(pair, again):
        raise AssertionError("blocking a sender did not change the pair mask")
    if float(blocked[0, 0, 1]) > 1e-5 or float(blocked[0, 2, 1]) > 1e-5:
        raise AssertionError("other receivers can still read the blocked sender")
    if float(blocked[0, 1].sum()) <= 0:
        raise AssertionError("the blocked sender must still be able to read other frames")
    if float(open_beta[0, 0, 1]) <= 0:
        raise AssertionError("fixture expected the unblocked receiver to read sender 1")


def _test_lifecycle():
    try:
        with FrameVizSession(query_indices=[(0, 0)], collect_pi=True):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    if active_session() is not None or _PI_CHUNK_SINKS:
        raise AssertionError("collector survived an exception")
    with FrameVizSession(query_indices=[(0, 0)], collect_pi=True) as first:
        first._pi_rows[(0, 0)] = "stale"
    with FrameVizSession(query_indices=[(1, 1)], collect_pi=False) as second:
        if second._pi_rows:
            raise AssertionError("new session kept a previous pi row")
    if _PI_CHUNK_SINKS:
        raise AssertionError("sink left registered")


def _test_effective_eta():
    import torch

    original = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    module_delta = torch.tensor([[[[0.0, 0.1, 0.0, 0.0]]]])
    delivered = torch.round((original + module_delta) * 4) / 4
    effective = delivered - original

    class _Mod:
        last_delta_x = module_delta
        last_alpha = torch.ones(1, 1, 1)
        last_beta = torch.zeros(1, 1, 1)
        last_gamma = torch.zeros(1, 1, 1, 1)
        last_stats = {"gate_mean": 0.5}

    with FrameVizSession(collect_pi=False) as session:
        session.record_delivered(
            frame_tokens=original,
            delivered=delivered,
            token_mask=torch.ones(1, 1, 1, dtype=torch.bool),
            frame_mask=torch.ones(1, 1, dtype=torch.bool),
            module=_Mod(),
            timestamps=[0.0],
        )
        bundle = session.export()
    expected = _rms(effective) / (_rms(original) + 1e-8)
    if not torch.allclose(bundle["eta"], expected):
        raise AssertionError("eta did not use the delivered correction")
    if float(bundle["dtype_gap_max_abs"]) <= 0:
        raise AssertionError("fixture should expose a dtype-cast gap")


def _test_hook_parity():
    import torch

    from ..ovis_hooks import (
        VISUAL_ATOM_ID,
        _apply_reljoin,
        configure_reljoin,
        set_reljoin_context,
    )

    class _Dummy:
        pass

    module = RelJoinResidual(8, rel_dim=4, variant="Hybrid")
    module.eval()
    lengths = [3, 4]
    ids = []
    for index, length in enumerate(lengths):
        ids.extend([VISUAL_ATOM_ID] * length)
        ids.append(10 + index)
    input_ids = torch.tensor([ids])
    embeds = torch.randn(1, len(ids), 8)
    configure_reljoin(module, enabled=True, bypass=False, eval_mode="full")
    set_reljoin_context(
        question_hidden=torch.randn(1, 8),
        timestamps=torch.tensor([[0.0, 1.0]]),
        frame_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    with torch.no_grad():
        off = _apply_reljoin(_Dummy(), input_ids, embeds)
        with FrameVizSession(query_indices=[(0, 1)], collect_pi=True) as session:
            on = _apply_reljoin(_Dummy(), input_ids, embeds)
            bundle = session.export()
        again = _apply_reljoin(_Dummy(), input_ids, embeds)
    if not torch.equal(off, on) or not torch.equal(on, again):
        raise AssertionError("hook output changed while the collector was attached")
    if not bundle["query_captured"][0]:
        raise AssertionError("hook forward did not capture the query pi row")
    if active_session() is not None:
        raise AssertionError("hook session was left active")
    configure_reljoin(module, enabled=True, bypass=False, blocked_sources=(1,))
    with torch.no_grad():
        _apply_reljoin(_Dummy(), input_ids, embeds)
        blocked = module.last_beta.clone()
    configure_reljoin(module, enabled=True, bypass=False)
    if float(blocked[0, 0, 1]) > 1e-5:
        raise AssertionError("hook did not apply the read-source block")
    if module is not None and get_blocked() is not None:
        raise AssertionError("blocked_sources leaked into the next configure")


def get_blocked():
    from ..ovis_hooks import get_hook_state

    return get_hook_state().blocked_sources


def _test_grid_geometry():
    bbox = token_bbox_on_original(
        row=0, col=0, grid_hw=(2, 2), resized_hw=(6, 6),
        crop_box=[0, 0, 4, 4], letterbox=(1, 1, 1, 1),
    )
    if [round(v, 5) for v in bbox] != [0.0, 0.0, 2.0, 2.0]:
        raise AssertionError(bbox)
    geometry = build_geometry_from_views(
        [{"height": 8, "width": 8}],
        [[{
            "view_id": "full",
            "view_kind": "full",
            "grid_hw": [2, 2],
            "crop_box": [0, 0, 8, 8],
            "resized_hw": [8, 8],
            "letterbox": [0, 0, 0, 0],
            "n_prefix_special": 1,
        }]],
        pmax=6,
        synthetic=True,
    )
    kinds = [token["kind"] for token in geometry["tokens"] if token["frame_index"] == 0]
    if kinds != ["special", "spatial", "spatial", "spatial", "spatial", "padding"]:
        raise AssertionError(kinds)
    spatial = np.asarray(geometry["spatial_token_mask"])
    if spatial.sum() != 4 or spatial[0, 0] or not spatial[0, 1]:
        raise AssertionError("spatial mask does not follow the special prefix")


def _test_multicrop_mass():
    geometry = build_geometry_from_views(
        [{"height": 4, "width": 4}],
        [[
            {
                "view_id": "tile",
                "view_kind": "tile",
                "grid_hw": [1, 1],
                "crop_box": [0, 0, 4, 4],
                "resized_hw": [4, 4],
                "letterbox": [0, 0, 0, 0],
            },
            {
                "view_id": "thumb",
                "view_kind": "thumbnail",
                "grid_hw": [1, 1],
                "crop_box": [0, 0, 4, 4],
                "resized_hw": [4, 4],
                "letterbox": [0, 0, 0, 0],
            },
        ]],
        pmax=2,
        synthetic=True,
    )
    values = np.array([0.25, 0.75], dtype=np.float64)
    painted = paint_frame_map(geometry, 0, values, mode="mass", separate_thumbnail=True)
    if abs(float(painted["canvas"].sum()) - 0.25) > 1e-6:
        raise AssertionError(painted["canvas"].sum())
    if abs(float(painted["thumbnail"].sum()) - 0.75) > 1e-6:
        raise AssertionError(painted["thumbnail"].sum())
    overlap = build_geometry_from_views(
        [{"height": 4, "width": 4}],
        [[
            {
                "view_id": "a",
                "view_kind": "tile",
                "grid_hw": [1, 1],
                "crop_box": [0, 0, 4, 4],
                "resized_hw": [4, 4],
            },
            {
                "view_id": "b",
                "view_kind": "tile",
                "grid_hw": [1, 1],
                "crop_box": [0, 0, 4, 4],
                "resized_hw": [4, 4],
            },
        ]],
        pmax=2,
        synthetic=True,
    )
    added = paint_frame_map(overlap, 0, values, mode="mass", separate_thumbnail=False)
    if abs(float(added["canvas"].sum()) - 1.0) > 1e-6:
        raise AssertionError("overlap mass was not added")
    averaged = paint_frame_map(overlap, 0, np.array([0.0, 1.0]), mode="mean", separate_thumbnail=False)
    if abs(float(averaged["canvas"].mean()) - 0.5) > 1e-6:
        raise AssertionError("coverage-weighted mean should average the two supports")


def _test_runtime_probe():
    from PIL import Image

    class _Cfg:
        patch_size = 2

    class _Tokenizer:
        config = _Cfg()

        def smart_resize(self, height, width, factor, min_pixels, max_pixels):
            del width, factor, min_pixels, max_pixels
            return 4, 4

    class _Model:
        visual_tokenizer = _Tokenizer()

    image = Image.new("RGB", (4, 4), "white")
    verified = resolve_runtime_geometry(
        _Model(), [image], [[1, 2, 2]], [4], min_pixels=1, max_pixels=64,
    )
    if verified["mapping_status"] != "verified_direct_resize":
        raise AssertionError(verified["mapping_status"])
    if verified["synthetic"]:
        raise AssertionError("a confirmed runtime mapping is not a synthetic fixture")

    class _NoResize:
        config = _Cfg()

    class _Unconfirmed:
        visual_tokenizer = _NoResize()

    unverified = resolve_runtime_geometry(
        _Unconfirmed(), [image], [[1, 2, 2]], [4], min_pixels=1, max_pixels=64,
    )
    if unverified["mapping_status"] != "spatial_mapping_unverified":
        raise AssertionError(unverified["mapping_status"])
    mismatch = resolve_runtime_geometry(
        _Model(), [image], [[1, 2, 2]], [3], min_pixels=1, max_pixels=64,
    )
    if mismatch["mapping_status"] != "spatial_mapping_unverified":
        raise AssertionError("token count mismatch must stay unverified")

    # Ovis2.5 keeps patch_size and hidden_stride on visual_tokenizer.vit.config.
    class _VitCfg:
        patch_size = 16
        hidden_stride = 2

    class _Vit:
        config = _VitCfg()

    class _OvisTokenizer:
        vit = _Vit()

        @staticmethod
        def smart_resize(height, width, factor=28, min_pixels=0, max_pixels=0):
            del height, width, factor, min_pixels, max_pixels
            return 320, 576

    class _OvisModel:
        visual_tokenizer = _OvisTokenizer()

    frame = Image.new("RGB", (1920, 1080), "white")
    ovis = resolve_runtime_geometry(
        _OvisModel(), [frame], [[1, 20, 36]], [180], min_pixels=200704, max_pixels=200704,
    )
    if ovis["mapping_status"] != "verified_direct_resize":
        raise AssertionError(ovis["mapping_status"])
    token = next(t for t in ovis["tokens"] if t["frame_index"] == 0 and t["token_index"] == 35)
    if list(token["patch_rc"]) != [1, 17]:
        raise AssertionError(f"token 35 should be merged cell (1, 17), got {token['patch_rc']}")


def _test_offline_render():
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        example = root / "ex"
        example.mkdir()
        frames = [{"height": 8, "width": 8}, {"height": 8, "width": 8}]
        view = {
            "view_id": "full",
            "view_kind": "full",
            "grid_hw": [2, 2],
            "crop_box": [0, 0, 8, 8],
            "resized_hw": [8, 8],
            "letterbox": [0, 0, 0, 0],
        }
        geometry = build_geometry_from_views(
            frames, [[view], [view]], pmax=4, mapping_status="verified_fixture", synthetic=True,
        )
        _write_dump(example, geometry, synthetic=True)
        stats = render_example(example)
        for name in (
            "figure_cross_frame.png",
            "figure_cross_frame.pdf",
            "figure_read_update.png",
            "figure_read_update.pdf",
            "figure_frame_matrix.png",
            "figure_frame_matrix.pdf",
            "geometry_debug.png",
            "captions_en.md",
        ):
            path = example / name
            if not path.is_file() or path.stat().st_size <= 0:
                raise AssertionError(f"missing {name}")
        caption = (example / "captions_en.md").read_text(encoding="utf-8")
        if "Synthetic fixture" not in caption or "question-guided pooling" not in caption.lower() and "Question-guided pooling" not in caption:
            raise AssertionError(caption)
        if "relative token correction" not in caption and "relative residual" not in caption:
            raise AssertionError(caption)
        unverified = root / "unverified"
        unverified.mkdir()
        bad = resolve_runtime_geometry(None, [Image.new("RGB", (8, 8))], [[1, 2, 2]], [4], min_pixels=1, max_pixels=8)
        # None model has no tokenizer. Build via the unresolved helper path already used.
        if bad["mapping_status"] != "spatial_mapping_unverified":
            raise AssertionError(bad.get("mapping_status"))
        # The None-model call uses n_frames from frames. tokens [4] and one frame.
        # pmax may be 4. Pad the dump tensors to that pmax.
        _write_dump(unverified, bad, synthetic=False)
        render_example(unverified)
        text = (unverified / "captions_en.md").read_text(encoding="utf-8")
        if "spatial_mapping_unverified" not in text:
            raise AssertionError("unverified figure was described as confirmed")
        del stats


def _write_dump(example: Path, geometry: dict, synthetic: bool) -> None:
    from PIL import Image

    k = int(geometry["n_frames"])
    pmax = int(geometry["pmax"])
    frame_dir = example / "frames"
    frame_dir.mkdir(exist_ok=True)
    for index in range(k):
        Image.new("RGB", (8, 8), (20 * index, 40, 80)).save(frame_dir / f"frame_{index:02d}.png")
    alpha = np.zeros((1, k, pmax), dtype=np.float32)
    alpha[0, :, 0] = 1
    beta = np.zeros((1, k, k), dtype=np.float32)
    if k > 1:
        beta[0, 0, 1] = 1
        beta[0, 1, 0] = 1
    pi = np.zeros((1, 1, k, pmax), dtype=np.float32)
    gamma = np.zeros((1, 1, k), dtype=np.float32)
    if k > 1:
        pi[0, 0, 1, 0] = 1
        gamma[0, 0, 1] = 1
    eta = np.linspace(0, 1, k * pmax, dtype=np.float32).reshape(1, k, pmax)
    token_mask = np.zeros((1, k, pmax), dtype=bool)
    token_mask[:, :, 0] = True
    frame_mask = np.ones((1, k), dtype=bool)
    np.savez(
        example / "tensors.npz",
        alpha=alpha,
        beta=beta,
        pi_selected=pi,
        gamma_selected=gamma,
        input_rms=np.ones_like(alpha),
        delta_rms=eta,
        module_delta_rms=eta,
        eta=eta,
        cosine_x_delta=np.zeros_like(alpha),
        token_mask=token_mask,
        frame_mask=frame_mask,
        spatial_token_mask=np.asarray(geometry["spatial_token_mask"], dtype=bool),
        query_indices=np.array([[0, 0]], dtype=np.int64),
    )
    (example / "metadata.json").write_text(json.dumps({
        "example_id": "synthetic" if synthetic else "unverified-case",
        "question": "Where is the change?",
        "gold": "A",
        "timestamps": [0.0, 1.5][:k],
        "frame_indices": list(range(k)),
        "mapping_status": geometry["mapping_status"],
    }), encoding="utf-8")
    (example / "queries.json").write_text(json.dumps({
        "method": "explicit",
        "queries": [{"frame_index": 0, "token_index": 0}],
    }), encoding="utf-8")
    (example / "predictions.json").write_text(json.dumps({
        "vanilla": {"parsed": "A"},
        "frameweave": {"parsed": "B"},
    }), encoding="utf-8")
    (example / "token_geometry.json").write_text(json.dumps(geometry), encoding="utf-8")


def _test_query_selection():
    eta = np.array([[0.1, 0.9, 0.2], [0.8, 0.05, 0.0]], dtype=np.float64)
    token_mask = np.ones_like(eta, dtype=bool)
    spatial = token_mask.copy()
    geometry = {"mapping_status": "verified_fixture", "tokens": []}
    first = select_queries(
        "top_eta", eta=eta, token_mask=token_mask, spatial_mask=spatial,
        geometry=geometry, max_queries=2, seed=0,
    )
    second = select_queries(
        "top_eta", eta=eta, token_mask=token_mask, spatial_mask=spatial,
        geometry=geometry, max_queries=2, seed=0,
    )
    if first["queries"] != second["queries"]:
        raise AssertionError("top eta was not deterministic")
    if first["queries"][0] != {"frame_index": 0, "token_index": 1}:
        raise AssertionError(first["queries"])
    if first["queries"][1]["frame_index"] == 0:
        raise AssertionError("the second query should prefer the other frame")


def _test_survey_selection():
    import csv
    import json
    import tempfile

    from .survey import (
        PAPER_SLOTS,
        QUOTA_CONTROL,
        QUOTA_FAILURE,
        QUOTA_SUCCESS,
        QUOTA_TSO,
        SURVEY_SORT_FIELDS,
        build_survey,
        contact_sort_key,
        write_paper_groups,
        write_survey,
    )

    joined = " ".join(SURVEY_SORT_FIELDS)
    for banned in ("heatmap", "eta", "pi", "gamma", "sharp"):
        if banned in joined:
            raise AssertionError(f"survey sort uses {banned}")
    entity = "Multi-Entities Grounding and Localization"
    state = "False Object Memory"
    other = "Scene Verification & Arrangement"
    tso = "Temporal Shot Ordering"
    abstain = "At least two descriptions do not accurately reflect any shot from the video"
    tso_choices = ["first then second", "second then first", "third order", abstain]
    rows = []
    predictions = {}

    def add(qid, task, spans, vanilla_ok, hybrid_ok, parsed="A", choices=None, video="clip.mp4", heatmap=0.0):
        rows.append({
            "question_id": qid,
            "task_type": task,
            "metadata": {"timestamps": spans},
            "resolved_video_path": video,
            "choices": choices or ["one", "two"],
            "answer": "A",
            "question": f"question {qid}",
            "heatmap_score": heatmap,
        })
        predictions[qid] = {
            "vanilla": {"correct": vanilla_ok, "parsed": parsed},
            "hybrid": {"correct": hybrid_ok, "parsed": "A"},
        }

    for index in range(8):
        add(f"E{index:02d}", entity, ["0 - 1", "4 - 5"], False, True, heatmap=0.0 if index == 0 else 999.0)
        add(f"S{index:02d}", state, ["0 - 1", "3 - 4", "8 - 9"], False, True)
    for index in range(5):
        add(f"CE{index:02d}", entity, ["1 - 2", "6 - 7"], True, True)
        add(f"CS{index:02d}", state, ["1 - 2", "6 - 7"], True, True)
    for index in range(4):
        add(f"SW{index:02d}", entity, ["0 - 1", "5 - 6"], False, False)
        add(f"RG{index:02d}", state, ["0 - 1", "5 - 6"], True, False)
    add("T_ABS", tso, ["0 - 1", "2 - 3"], False, True, parsed="D", choices=tso_choices)
    add("T_ABS2", tso, ["0 - 1", "2 - 3"], False, False, parsed="D", choices=tso_choices)
    add("T_WRONG", tso, ["0 - 1", "9 - 10"], False, False, parsed="B", choices=tso_choices)
    add("ONE", entity, ["0 - 1"], False, True)
    add("ALL", entity, "all", False, True)
    add("SCENE", other, ["0 - 1", "4 - 5"], False, True)
    add("NOVIDEO", entity, ["0 - 1", "4 - 5"], False, True, video="")

    survey = build_survey(rows, predictions)
    again = build_survey(rows, predictions)
    if survey["example_ids"] != again["example_ids"]:
        raise AssertionError("survey selection was not deterministic")
    if survey["heatmap_used_for_selection"] or survey["paper_figures_chosen"]:
        raise AssertionError("survey recorded a figure choice")
    ids = survey["example_ids"]
    if ids[0] != "E00":
        raise AssertionError(f"heatmap or id order changed the first success: {ids[:4]}")
    selected = {item["question_id"]: item for item in survey["selected"]}
    success = [item for item in survey["selected"] if item["quota"] == "success"]
    control = [item for item in survey["selected"] if item["quota"] == "control"]
    failure = [item for item in survey["selected"] if item["quota"] == "failure"]
    tso_rows = [item for item in survey["selected"] if item["quota"] == "tso"]
    if [len(success), len(control), len(failure), len(tso_rows)] != [
        QUOTA_SUCCESS, QUOTA_CONTROL, QUOTA_FAILURE, QUOTA_TSO,
    ]:
        raise AssertionError([len(success), len(control), len(failure), len(tso_rows)])
    if {item["family"] for item in success} != {"entity", "state"}:
        raise AssertionError("success quota did not keep two task families")
    if {item["prior_role"] for item in failure} != {"still_wrong", "regression"}:
        raise AssertionError("failure quota is missing a still-wrong or regression case")
    if {item["question_id"] for item in tso_rows} != {"T_ABS", "T_WRONG"}:
        raise AssertionError(tso_rows)
    if selected["T_ABS"]["prior_role"] != "tso_abstain" or selected["T_WRONG"]["prior_role"] != "tso_wrong_order":
        raise AssertionError("TSO tags were not split into abstain and wrong order")
    by_id = {item["question_id"]: item for item in survey["candidates"]}
    if "NOVIDEO" in by_id:
        raise AssertionError("a row without video entered the survey table")
    if by_id["ONE"]["selected_for_run"] or by_id["ALL"]["selected_for_run"]:
        raise AssertionError("single-span or uniform fallback was selected")
    if "ONE" not in by_id or "SCENE" not in by_id:
        raise AssertionError("the full surveyed list dropped an excluded or unselected row")
    if by_id["SCENE"]["selected_for_run"]:
        raise AssertionError("a non-priority task filled a quota while entity and state rows remained")
    success_ids = [item["question_id"] for item in success]
    if success_ids != sorted(success_ids, key=lambda qid: (0 if qid.startswith("E") else 1, qid)):
        raise AssertionError("success rows were not ordered by task priority then question id")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_survey(root, survey, predictions_csv="")
        paper = json.loads((root / "paper_selection.json").read_text(encoding="utf-8"))
        for slot in PAPER_SLOTS:
            if paper[slot]["question_id"] or paper[slot]["reason"]:
                raise AssertionError("paper selection was filled by the survey")
        paper["main_success_1"] = {"question_id": "E00", "reason": "readable object across frames"}
        (root / "paper_selection.json").write_text(json.dumps(paper), encoding="utf-8")
        write_survey(root, survey, predictions_csv="")
        kept = json.loads((root / "paper_selection.json").read_text(encoding="utf-8"))
        if kept["main_success_1"]["question_id"] != "E00":
            raise AssertionError("a later survey overwrote the manual paper choice")
        groups = root / "groups"
        groups.mkdir()
        _group_example(groups / "E00", task=entity, gold="A", vanilla="B", frameweave="A", prior="success")
        _group_example(groups / "SW00", task=entity, gold="A", vanilla="B", frameweave="C", prior="success")
        write_paper_groups(groups)
        grouped = list(csv.DictReader((groups / "paper_groups.csv").open(encoding="utf-8")))
        roles = {row["question_id"]: row for row in grouped}
        if roles["E00"]["role"] != "success" or roles["E00"]["role_disagrees_with_prior_eval"] != "False":
            raise AssertionError(roles["E00"])
        if roles["SW00"]["role"] != "still_wrong" or roles["SW00"]["role_disagrees_with_prior_eval"] != "True":
            raise AssertionError(roles["SW00"])
        if [row["question_id"] for row in grouped] != ["E00", "SW00"]:
            raise AssertionError("paper groups were not ordered by role")
    pending = build_survey(rows[:3], None)
    if any(item["quota"] == "success" for item in pending["selected"]):
        raise AssertionError("missing predictions were treated as a success quota")
    order = [
        {"role": "control", "example_id": "b"},
        {"role": "success", "example_id": "a", "heatmap": "9"},
        {"role": "success", "example_id": "c", "heatmap": "0"},
    ]
    if [row["example_id"] for row in sorted(order, key=contact_sort_key)] != ["a", "c", "b"]:
        raise AssertionError("contact order used something other than role then id")


def _group_example(path: Path, *, task: str, gold: str, vanilla: str, frameweave: str, prior: str) -> None:
    import json

    path.mkdir()
    (path / "metadata.json").write_text(json.dumps({
        "example_id": path.name,
        "task": task,
        "gold": gold,
        "choices": ["one", "two", "three"],
        "prior_role": prior,
        "n_evidence_spans": 2,
        "survey_reason": "fixture",
    }), encoding="utf-8")
    (path / "predictions.json").write_text(json.dumps({
        "vanilla": {"parsed": vanilla},
        "frameweave": {"parsed": frameweave},
    }), encoding="utf-8")


if __name__ == "__main__":
    run_cpu_checks()
