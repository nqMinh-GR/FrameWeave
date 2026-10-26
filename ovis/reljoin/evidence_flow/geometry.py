"""Pixel boxes for Ovis2.5 merged visual tokens.

The merge order is the one in ``modeling_ovis2_5.VisualTokenizer.preprocess``:
patches are grouped as ``(merged_row, merged_col, sub_row, sub_col)`` with
``hidden_stride`` 2, then every ``stride**2`` consecutive patches become one
token in row-major order over the merged grid. ``verified_token_boxes`` uses
that index. The impulse test checks this preprocess tensor. The softmax head
is not a spatial index, because attention moves a corner impulse to other tokens.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448


class AlignmentError(RuntimeError):
    """The live token of maximum deviation is not the predicted merged cell."""


class ModelRequired(RuntimeError):
    """The impulse test needs the frozen Ovis2.5 vision tower."""


def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Same arithmetic as ``VisualTokenizer.smart_resize`` in Ovis2.5-9B."""
    height = int(height)
    width = int(width)
    if height < factor or width < factor:
        if height < width:
            width = round(factor / height * width)
            height = factor
        else:
            height = round(factor / width * height)
            width = factor
    elif max(height, width) / min(height, width) > 200:
        if height > width:
            height = 200 * width
        else:
            width = 200 * height
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def merged_index_groups(grid_h: int, grid_w: int, stride: int) -> np.ndarray:
    """Return ``[P, stride*stride]`` patch-row-major labels in merge order.

    This follows the preprocess reshape/transpose, not a nominal 10×18 grid.
    """
    if stride < 1 or grid_h % stride or grid_w % stride:
        raise ValueError(f"grid {(grid_h, grid_w)} is not divisible by stride {stride}")
    token_h = grid_h // stride
    token_w = grid_w // stride
    labels = np.arange(grid_h * grid_w, dtype=np.int64).reshape(grid_h, grid_w)
    grouped = labels.reshape(token_h, stride, token_w, stride)
    ordered = grouped.transpose(0, 2, 1, 3).reshape(token_h * token_w, stride * stride)
    return ordered


def verified_token_boxes(
    raw_image_hw: tuple[int, int],
    processor_metadata: dict,
    grid_thw: Any,
) -> np.ndarray:
    """``[P, 4]`` boxes ``[x0, y0, x1, y1]`` in original RGB pixels, row-major merged tokens."""
    grid = _as_thw(grid_thw)
    temporal, grid_h, grid_w = grid
    if temporal != 1:
        raise ValueError(f"expected one image, grid_thw temporal={temporal}")
    patch = int(processor_metadata["patch_size"])
    stride = int(processor_metadata["hidden_stride"])
    resized_h, resized_w = [int(v) for v in processor_metadata["resized_hw"]]
    if (grid_h, grid_w) != (resized_h // patch, resized_w // patch):
        raise ValueError(
            f"grid_thw {(grid_h, grid_w)} != resized {(resized_h, resized_w)} / patch {patch}"
        )
    raw_h, raw_w = int(raw_image_hw[0]), int(raw_image_hw[1])
    letterbox = [int(v) for v in processor_metadata.get("letterbox", (0, 0, 0, 0))]
    if letterbox != [0, 0, 0, 0]:
        raise ValueError(
            f"Ovis native-aspect resize is not letterboxed, got padding {letterbox}"
        )
    token_h = grid_h // stride
    token_w = grid_w // stride
    cell_h = stride * patch
    cell_w = stride * patch
    scale_y = raw_h / float(resized_h)
    scale_x = raw_w / float(resized_w)
    boxes = np.zeros((token_h * token_w, 4), dtype=np.float64)
    for merged_row in range(token_h):
        for merged_col in range(token_w):
            y0 = merged_row * cell_h
            x0 = merged_col * cell_w
            y1 = y0 + cell_h
            x1 = x0 + cell_w
            index = merged_row * token_w + merged_col
            boxes[index] = [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]
    return boxes


def processor_metadata_for_image(
    raw_image_hw: tuple[int, int],
    *,
    patch_size: int,
    hidden_stride: int,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> dict:
    raw_h, raw_w = int(raw_image_hw[0]), int(raw_image_hw[1])
    factor = int(patch_size) * int(hidden_stride)
    resized_h, resized_w = smart_resize(
        raw_h, raw_w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
    )
    return {
        "patch_size": int(patch_size),
        "hidden_stride": int(hidden_stride),
        "resized_hw": [int(resized_h), int(resized_w)],
        "letterbox": [0, 0, 0, 0],
        "min_pixels": int(min_pixels),
        "max_pixels": int(max_pixels),
        "merge_order": "row_major_merged_cell_subrow_subcol",
        "factor": factor,
    }


def run_merge_order_self_test() -> dict:
    """Check the reshape groups without loading Ovis.

    A single merged cell painted in patch space must be exactly one group of
    ``stride**2`` consecutive flattened patches, at the predicted index.
    """
    stride = 2
    grid_h, grid_w = 20, 36
    groups = merged_index_groups(grid_h, grid_w, stride)
    token_h, token_w = grid_h // stride, grid_w // stride
    if groups.shape != (token_h * token_w, stride * stride):
        raise AlignmentError(f"unexpected group shape {groups.shape}")
    probes = [(0, 0), (token_h - 1, token_w - 1), (token_h // 3, token_w // 5), (4, 11)]
    for merged_row, merged_col in probes:
        expected = merged_row * token_w + merged_col
        painted = np.zeros((grid_h, grid_w), dtype=np.float64)
        painted[
            merged_row * stride : (merged_row + 1) * stride,
            merged_col * stride : (merged_col + 1) * stride,
        ] = 1.0
        flat = painted.reshape(token_h, stride, token_w, stride)
        flat = flat.transpose(0, 2, 1, 3).reshape(-1)
        mass = flat.reshape(-1, stride * stride).sum(axis=1)
        winner = int(np.argmax(mass))
        if winner != expected or int(np.count_nonzero(mass > 0)) != 1:
            raise AlignmentError(
                f"merge order off at cell {(merged_row, merged_col)}: "
                f"winner {winner}, expected {expected}"
            )
        if not np.array_equal(np.sort(groups[expected]), np.flatnonzero(painted.reshape(-1))):
            raise AlignmentError(f"group labels disagree at token {expected}")
    for raw_hw in ((720, 1280), (1080, 1920)):
        meta = processor_metadata_for_image(raw_hw, patch_size=16, hidden_stride=2)
        resized_h, resized_w = meta["resized_hw"]
        if (resized_h, resized_w) != (320, 576):
            raise AlignmentError(f"{raw_hw} resized to {(resized_h, resized_w)}, expected (320, 576)")
        grid_h_live, grid_w_live = resized_h // 16, resized_w // 16
        boxes = verified_token_boxes(raw_hw, meta, [1, grid_h_live, grid_w_live])
        if boxes.shape != (10 * 18, 4):
            raise AlignmentError(f"{raw_hw} boxes {boxes.shape}, expected 10x18 merged tokens")
        if boxes.min() < -1e-6 or boxes[0, 0] > 1e-6 or boxes[0, 1] > 1e-6:
            raise AlignmentError(f"origin box is not the top-left cell: {boxes[0].tolist()}")
        if abs(float(boxes[-1, 2]) - raw_hw[1]) > 1e-4 or abs(float(boxes[-1, 3]) - raw_hw[0]) > 1e-4:
            raise AlignmentError(f"last merged cell does not reach the raw frame edge: {boxes[-1].tolist()}")
    return {
        "status": "pass",
        "scope": "reshape_order_without_vision_tower",
        "grid_hw_patches": [grid_h, grid_w],
        "merged_hw": [token_h, token_w],
        "hidden_stride": stride,
        "probes": [list(item) for item in probes],
    }


def run_localization_test(model, *, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> dict:
    """Impulse images through the live Ovis preprocess and 2×2 patch grouping.

    The merged-token index is the order of ``VisualTokenizer.preprocess`` after
    grouping every ``hidden_stride**2`` patches. The softmax head is not a
    spatial index: attention mixes a corner impulse into other tokens.
    """
    if model is None:
        raise ModelRequired(
            "Ovis2.5-9B is not loaded. Refusing to draw a spatial heatmap."
        )
    spec = _live_spec(model)
    patch = int(spec["patch_size"])
    stride = int(spec["hidden_stride"])
    if patch < 1 or stride < 1:
        raise AlignmentError(f"visual tokenizer spec is incomplete: {spec}")
    live_resize = _live_smart_resize(model)
    raw_hw = (720, 1280)
    meta = processor_metadata_for_image(
        raw_hw, patch_size=patch, hidden_stride=stride,
        min_pixels=min_pixels, max_pixels=max_pixels,
    )
    resized_h, resized_w = meta["resized_hw"]
    if live_resize is not None:
        got = live_resize(raw_hw[0], raw_hw[1], patch * stride, min_pixels, max_pixels)
        if tuple(got) != (resized_h, resized_w):
            raise AlignmentError(
                f"live smart_resize {got} != traced {(resized_h, resized_w)}"
            )
    grid_h, grid_w = resized_h // patch, resized_w // patch
    boxes = verified_token_boxes(raw_hw, meta, [1, grid_h, grid_w])
    token_h, token_w = grid_h // stride, grid_w // stride
    probes = [(0, 0), (token_h - 1, token_w - 1), (max(token_h // 3, 1), max(token_w // 5, 1))]
    background = _neutral_image(raw_hw)
    base, live_grid = _merged_preprocess_features(model, background, min_pixels, max_pixels)
    if tuple(live_grid) != (1, grid_h, grid_w):
        raise AlignmentError(
            f"preprocess grid_thw {live_grid} != traced {(1, grid_h, grid_w)}"
        )
    if int(base.shape[0]) != token_h * token_w:
        raise AlignmentError(
            f"preprocess returned {int(base.shape[0])} merged tokens, expected {token_h * token_w}"
        )
    checked = []
    for merged_row, merged_col in probes:
        expected = merged_row * token_w + merged_col
        image = _paint_box(background, boxes[expected])
        feat, _grid = _merged_preprocess_features(model, image, min_pixels, max_pixels)
        delta = np.linalg.norm(feat - base, axis=-1)
        winner = int(np.argmax(delta))
        if winner != expected:
            top = [int(i) for i in np.argsort(-delta)[:5]]
            raise AlignmentError(
                f"impulse at merged cell {(merged_row, merged_col)} "
                f"peaked at token {winner}, expected {expected}, top={top}"
            )
        checked.append(
            {
                "merged_rc": [merged_row, merged_col],
                "token_index": expected,
                "box": boxes[expected].tolist(),
                "peak_delta": float(delta[winner]),
            }
        )
    return {
        "status": "pass",
        "scope": "preprocess_patch_groups",
        "patch_size": patch,
        "hidden_stride": stride,
        "resized_hw": [resized_h, resized_w],
        "merged_hw": [token_h, token_w],
        "tokenizer_class": spec.get("class_name", ""),
        "probes": checked,
    }


def _as_thw(grid_thw: Any) -> tuple[int, int, int]:
    if hasattr(grid_thw, "detach"):
        grid_thw = grid_thw.detach().cpu().tolist()
    array = np.asarray(grid_thw).reshape(-1)
    if array.size < 3:
        raise ValueError(f"grid_thw needs temporal, height, width; got {grid_thw}")
    return int(array[-3]), int(array[-2]), int(array[-1])


def _live_spec(model) -> dict:
    from reljoin.frame_viz.geometry import introspect_visual_tokenizer

    spec = introspect_visual_tokenizer(model)
    if not spec.get("patch_size") or not spec.get("hidden_stride"):
        raise AlignmentError(f"cannot read patch_size/hidden_stride from the loaded model: {spec}")
    return spec


def _live_smart_resize(model):
    vt = getattr(model, "visual_tokenizer", None)
    fn = getattr(vt, "smart_resize", None)
    if not callable(fn):
        return None

    def _call(height, width, factor, min_pixels, max_pixels):
        try:
            out = fn(height, width, factor, min_pixels, max_pixels)
        except TypeError:
            out = fn(height=height, width=width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
        return int(out[0]), int(out[1])

    return _call


def _neutral_image(raw_hw: tuple[int, int]):
    from PIL import Image

    height, width = raw_hw
    return Image.new("RGB", (width, height), (128, 128, 128))


def _paint_box(image, box: np.ndarray):
    """Paint the inner 60% of a token box so bilinear resize does not bleed a full cell."""
    from PIL import ImageDraw

    painted = image.copy()
    x0, y0, x1, y1 = [float(v) for v in box]
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    hx = 0.30 * (x1 - x0)
    hy = 0.30 * (y1 - y0)
    draw = ImageDraw.Draw(painted)
    draw.rectangle([cx - hx, cy - hy, cx + hx, cy + hy], fill=(255, 255, 255))
    return painted


def _merged_preprocess_features(model, image, min_pixels: int, max_pixels: int):
    """Group the live preprocess tensor the same way the 2×2 merger does."""
    vt = model.visual_tokenizer
    pixels, grid = vt.preprocess(image=image, min_pixels=min_pixels, max_pixels=max_pixels)
    stride = int(vt.vit.config.hidden_stride)
    flat = pixels.detach().float().cpu().numpy()
    group = stride * stride
    if flat.ndim != 2 or flat.shape[0] % group != 0:
        raise AlignmentError(f"preprocess patches have shape {flat.shape} for stride {stride}")
    grouped = flat.reshape(-1, group, flat.shape[-1]).reshape(flat.shape[0] // group, -1)
    grid_list = [int(v) for v in grid.detach().cpu().reshape(-1).tolist()[-3:]]
    return grouped, tuple(grid_list)
