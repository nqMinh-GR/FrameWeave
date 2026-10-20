"""Nominal token supports. Overlays are drawn only when mapping_status starts with verified."""
from __future__ import annotations

import inspect
from typing import Any, Optional

import numpy as np

VERIFIED_PREFIX = "verified"


def is_verified(status: str) -> bool:
    return str(status or "").startswith(VERIFIED_PREFIX)


def _as_box(box) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in box]
    return [x0, y0, x1, y1]


def token_bbox_on_original(
    *,
    row: int,
    col: int,
    grid_hw: tuple[int, int],
    resized_hw: tuple[int, int],
    crop_box: list[float],
    letterbox: tuple[int, int, int, int],
) -> list[float]:
    """Map one patch on the resized canvas back onto the original crop.

    letterbox is (top, left, bottom, right) padding inside the resized canvas.
    The crop box is the original-image rectangle covered by the content, not the padding.
    """
    grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
    resized_h, resized_w = int(resized_hw[0]), int(resized_hw[1])
    top, left, bottom, right = [int(v) for v in letterbox]
    content_h = resized_h - top - bottom
    content_w = resized_w - left - right
    if grid_h < 1 or grid_w < 1 or content_h < 1 or content_w < 1:
        raise ValueError("patch grid and content size must be positive")
    y0 = top + row * content_h / grid_h
    y1 = top + (row + 1) * content_h / grid_h
    x0 = left + col * content_w / grid_w
    x1 = left + (col + 1) * content_w / grid_w
    cx0, cy0, cx1, cy1 = crop_box
    sx = (cx1 - cx0) / content_w
    sy = (cy1 - cy0) / content_h
    return [
        cx0 + (x0 - left) * sx,
        cy0 + (y0 - top) * sy,
        cx0 + (x1 - left) * sx,
        cy0 + (y1 - top) * sy,
    ]


def build_geometry_from_views(
    frames: list[dict],
    views_per_frame: list[list[dict]],
    *,
    pmax: int,
    mapping_status: str = "verified_fixture",
    synthetic: bool = True,
    notes: Optional[list[str]] = None,
) -> dict:
    """Pack declared views into RelJoin token slots.

    Token order inside a frame is view order, then prefix specials, row-major
    spatial tokens, then suffix specials. Remaining slots are padding.
    """
    if len(frames) != len(views_per_frame):
        raise ValueError("frames and views_per_frame must have the same length")
    tokens: list[dict] = []
    spatial = np.zeros((len(frames), int(pmax)), dtype=bool)
    token_mask = np.zeros((len(frames), int(pmax)), dtype=bool)
    frame_summaries = []
    for frame_index, (frame, views) in enumerate(zip(frames, views_per_frame)):
        original_hw = [int(frame["height"]), int(frame["width"])]
        cursor = 0
        view_summaries = []
        for view in views:
            grid_h, grid_w = [int(v) for v in view["grid_hw"]]
            n_prefix = int(view.get("n_prefix_special") or 0)
            n_suffix = int(view.get("n_suffix_special") or 0)
            n_spatial = grid_h * grid_w
            view_id = str(view.get("view_id") or f"frame{frame_index}")
            view_kind = str(view.get("view_kind") or "full")
            crop_box = _as_box(view.get("crop_box") or [0, 0, frame["width"], frame["height"]])
            resized_hw = [int(v) for v in view.get("resized_hw") or original_hw]
            letterbox = tuple(int(v) for v in view.get("letterbox") or (0, 0, 0, 0))
            for _ in range(n_prefix):
                tokens.append(
                    _token_record(
                        frame_index, cursor, "special", view_id, view_kind,
                        crop_box, original_hw, resized_hw, [grid_h, grid_w], None, letterbox, None,
                    )
                )
                token_mask[frame_index, cursor] = True
                cursor += 1
            for row in range(grid_h):
                for col in range(grid_w):
                    bbox = token_bbox_on_original(
                        row=row,
                        col=col,
                        grid_hw=(grid_h, grid_w),
                        resized_hw=(resized_hw[0], resized_hw[1]),
                        crop_box=crop_box,
                        letterbox=letterbox,
                    )
                    tokens.append(
                        _token_record(
                            frame_index, cursor, "spatial", view_id, view_kind,
                            crop_box, original_hw, resized_hw, [grid_h, grid_w],
                            [row, col], letterbox, bbox,
                        )
                    )
                    token_mask[frame_index, cursor] = True
                    spatial[frame_index, cursor] = True
                    cursor += 1
            for _ in range(n_suffix):
                tokens.append(
                    _token_record(
                        frame_index, cursor, "special", view_id, view_kind,
                        crop_box, original_hw, resized_hw, [grid_h, grid_w], None, letterbox, None,
                    )
                )
                token_mask[frame_index, cursor] = True
                cursor += 1
            view_summaries.append(
                {
                    "view_id": view_id,
                    "view_kind": view_kind,
                    "grid_hw": [grid_h, grid_w],
                    "n_prefix_special": n_prefix,
                    "n_suffix_special": n_suffix,
                    "n_spatial": n_spatial,
                    "crop_box": crop_box,
                    "letterbox": list(letterbox),
                }
            )
        if cursor > int(pmax):
            raise ValueError(f"frame {frame_index} needs {cursor} tokens but pmax is {pmax}")
        for pad in range(cursor, int(pmax)):
            tokens.append(
                _token_record(
                    frame_index, pad, "padding", "", "padding",
                    None, original_hw, None, None, None, None, None,
                )
            )
        frame_summaries.append(
            {
                "frame_index": frame_index,
                "original_hw": original_hw,
                "n_real_tokens": cursor,
                "views": view_summaries,
            }
        )
    rules = {
        "routing_mass": (
            "Spread each spatial token's routing mass uniformly on its bbox and add "
            "overlaps so the pixel sum equals the spatial mass. Thumbnail views are "
            "kept on a separate canvas."
        ),
        "pooling_and_eta": (
            "Alpha and eta use a coverage-weighted mean on overlapping supports. "
            "That mean is an update or pooling magnitude, not a probability."
        ),
        "nonspatial": (
            "Special, separator, and unresolved tokens keep their routing mass in a "
            "separate total. They are not dropped and not folded back into spatial tokens."
        ),
    }
    return {
        "mapping_status": mapping_status,
        "synthetic": bool(synthetic),
        "pmax": int(pmax),
        "n_frames": len(frames),
        "tokens": tokens,
        "spatial_token_mask": spatial.tolist(),
        "geometry_token_mask": token_mask.tolist(),
        "frames": frame_summaries,
        "composition_rules": rules,
        "notes": list(notes or []),
    }


def _token_record(
    frame_index, token_index, kind, view_id, view_kind,
    crop_box, original_hw, resized_hw, patch_grid, patch_rc, letterbox, bbox,
) -> dict:
    return {
        "frame_index": int(frame_index),
        "token_index": int(token_index),
        "kind": kind,
        "view_id": view_id,
        "view_kind": view_kind,
        "crop_box": crop_box,
        "original_hw": original_hw,
        "resized_hw": resized_hw,
        "patch_grid": patch_grid,
        "patch_rc": patch_rc,
        "letterbox": None if letterbox is None else [int(v) for v in letterbox],
        "bbox": None if bbox is None else [float(v) for v in bbox],
        "spatial": kind == "spatial",
    }


def unresolved_geometry(
    *,
    n_frames: int,
    tokens_per_frame: list[int],
    frame_sizes: list[tuple[int, int]],
    grid_thws: list[list[int]],
    reason: str,
    spec: Optional[dict] = None,
) -> dict:
    pmax = max([1, *tokens_per_frame, 1])
    frames = []
    for index in range(n_frames):
        height, width = frame_sizes[index] if index < len(frame_sizes) else (0, 0)
        frames.append({"height": int(height), "width": int(width)})
    geometry = build_geometry_from_views(
        frames,
        [[] for _ in frames],
        pmax=pmax,
        mapping_status="spatial_mapping_unverified",
        synthetic=False,
        notes=[reason],
    )
    # Replace the empty real-token region with unresolved records.
    tokens = []
    spatial = np.zeros((n_frames, pmax), dtype=bool)
    token_mask = np.zeros((n_frames, pmax), dtype=bool)
    for frame_index, count in enumerate(tokens_per_frame):
        height, width = frames[frame_index]["height"], frames[frame_index]["width"]
        original_hw = [height, width]
        for token_index in range(int(count)):
            tokens.append(
                _token_record(
                    frame_index, token_index, "unresolved", "", "unknown",
                    None, original_hw, None, None, None, None, None,
                )
            )
            token_mask[frame_index, token_index] = True
        for token_index in range(int(count), pmax):
            tokens.append(
                _token_record(
                    frame_index, token_index, "padding", "", "padding",
                    None, original_hw, None, None, None, None, None,
                )
            )
        geometry["frames"][frame_index]["n_real_tokens"] = int(count)
        geometry["frames"][frame_index]["views"] = []
    geometry["tokens"] = tokens
    geometry["spatial_token_mask"] = spatial.tolist()
    geometry["geometry_token_mask"] = token_mask.tolist()
    geometry["raw_grid_thws"] = grid_thws
    geometry["tokenizer_spec"] = spec or {}
    geometry["notes"] = [reason]
    return geometry


def paint_frame_map(
    geometry: dict,
    frame_index: int,
    values: np.ndarray,
    *,
    mode: str,
    separate_thumbnail: bool = False,
) -> dict:
    """Paint one frame's token values.

    mode='mass' adds value/area so the canvas sums to the painted mass.
    mode='mean' is the coverage-weighted mean and is not a probability.
    """
    if mode not in ("mass", "mean"):
        raise ValueError(mode)
    frame = geometry["frames"][int(frame_index)]
    height, width = [int(v) for v in frame["original_hw"]]
    if height < 1 or width < 1:
        raise ValueError("original frame size is unknown")
    main = np.zeros((height, width), dtype=np.float64)
    weight = np.zeros((height, width), dtype=np.float64)
    thumb = np.zeros((height, width), dtype=np.float64)
    thumb_weight = np.zeros((height, width), dtype=np.float64)
    painted = 0.0
    thumb_mass = 0.0
    nonspatial = 0.0
    for token in geometry["tokens"]:
        if int(token["frame_index"]) != int(frame_index):
            continue
        token_index = int(token["token_index"])
        if token_index >= len(values):
            continue
        value = float(values[token_index])
        if token["kind"] == "padding":
            continue
        if not token["spatial"] or token["bbox"] is None:
            nonspatial += value
            continue
        if separate_thumbnail and token["view_kind"] == "thumbnail":
            _add_support(thumb, thumb_weight, token["bbox"], value, mode, width, height)
            thumb_mass += value
            continue
        _add_support(main, weight, token["bbox"], value, mode, width, height)
        painted += value
    if mode == "mean":
        shown = np.zeros_like(main)
        thumb_shown = np.zeros_like(thumb)
        np.divide(main, np.maximum(weight, 1e-12), out=shown, where=weight > 0)
        np.divide(thumb, np.maximum(thumb_weight, 1e-12), out=thumb_shown, where=thumb_weight > 0)
    else:
        shown = main
        thumb_shown = thumb
    return {
        "canvas": shown,
        "weight": weight,
        "thumbnail": thumb_shown,
        "painted_mass": float(painted),
        "thumbnail_mass": float(thumb_mass),
        "nonspatial_mass": float(nonspatial),
        "mode": mode,
    }


def _add_support(canvas, weight, bbox, value, mode, width, height) -> None:
    x0, y0, x1, y1 = bbox
    left = max(0, int(np.floor(x0)))
    top = max(0, int(np.floor(y0)))
    right = min(width, int(np.ceil(x1)))
    bottom = min(height, int(np.ceil(y1)))
    if right <= left or bottom <= top:
        return
    area = (right - left) * (bottom - top)
    if mode == "mass":
        canvas[top:bottom, left:right] += value / area
    else:
        canvas[top:bottom, left:right] += value
    weight[top:bottom, left:right] += 1.0


def routing_components(
    gamma_row: np.ndarray,
    pi_row: np.ndarray,
    token_mask_row: np.ndarray,
    spatial_mask_row: np.ndarray,
) -> dict:
    """R = gamma * pi for one receiver token. Masks are not renormalized."""
    gamma_row = np.asarray(gamma_row, dtype=np.float64)
    pi_row = np.asarray(pi_row, dtype=np.float64)
    token_mask_row = np.asarray(token_mask_row, dtype=bool)
    spatial_mask_row = np.asarray(spatial_mask_row, dtype=bool)
    routing = gamma_row[:, None] * pi_row
    real = token_mask_row
    spatial = real & spatial_mask_row
    nonspatial = real & ~spatial_mask_row
    return {
        "R": routing,
        "per_frame_mass": routing.sum(axis=-1),
        "spatial_mass": float(routing[spatial].sum()) if spatial.any() else 0.0,
        "nonspatial_mass": float(routing[nonspatial].sum()) if nonspatial.any() else 0.0,
        "total_mass": float(routing[real].sum()) if real.any() else 0.0,
    }


def _grid_rows(grid_thws) -> list[list[int]]:
    if grid_thws is None:
        return []
    if hasattr(grid_thws, "detach"):
        grid_thws = grid_thws.detach().cpu()
    array = np.asarray(grid_thws)
    if array.size == 0:
        return []
    if array.ndim == 1:
        array = array.reshape(1, -1)
    rows = []
    for row in array.reshape(-1, array.shape[-1]):
        rows.append([int(v) for v in row.tolist()])
    return rows


def introspect_visual_tokenizer(model) -> dict:
    spec: dict[str, Any] = {"found": False}
    vt = getattr(model, "visual_tokenizer", None)
    if vt is None:
        spec["reason"] = "model has no visual_tokenizer"
        return spec
    spec["found"] = True
    spec["class_name"] = type(vt).__name__
    cfg = getattr(vt, "config", None)
    spec["config_class"] = type(cfg).__name__ if cfg is not None else ""
    vit = getattr(vt, "vit", None)
    vit_cfg = getattr(vit, "config", None) if vit is not None else None
    for owner_name, owner in (("tokenizer", vt), ("config", cfg), ("vit_config", vit_cfg)):
        if owner is None:
            continue
        for name in (
            "patch_size",
            "temporal_patch_size",
            "merge_size",
            "spatial_merge_size",
            "hidden_stride",
        ):
            if hasattr(owner, name):
                value = getattr(owner, name)
                if isinstance(value, (int, float)):
                    spec[name] = int(value)
                    spec[f"{name}_from"] = owner_name
    return spec


def _find_smart_resize(model):
    vt = getattr(model, "visual_tokenizer", None)
    owners = [vt, getattr(vt, "image_processor", None) if vt is not None else None, model]
    for owner in owners:
        if owner is None:
            continue
        for name in ("smart_resize", "_smart_resize"):
            fn = getattr(owner, name, None)
            if callable(fn):
                return fn
    return None


def _call_smart_resize(fn, height: int, width: int, factor: int, min_pixels: int, max_pixels: int):
    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        params = []
    kwargs = {
        "height": int(height),
        "width": int(width),
        "factor": int(factor),
        "min_pixels": int(min_pixels),
        "max_pixels": int(max_pixels),
    }
    if params and not any(name in kwargs for name in params):
        return None
    if params:
        kwargs = {key: value for key, value in kwargs.items() if key in params}
    try:
        out = fn(**kwargs) if kwargs else fn(height, width, factor, min_pixels, max_pixels)
    except TypeError:
        return None
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        return int(out[0]), int(out[1])
    return None


def resolve_runtime_geometry(
    model,
    frames,
    grid_thws,
    tokens_per_frame: list[int],
    *,
    min_pixels: int,
    max_pixels: int,
) -> dict:
    """Bind evaluation grids to pixels only when the tokenizer confirms the resize."""
    n_frames = len(list(frames))
    frame_sizes = []
    for frame in frames:
        width, height = frame.size
        frame_sizes.append((int(height), int(width)))
    grids = _grid_rows(grid_thws)
    spec = {} if model is None else introspect_visual_tokenizer(model)
    reason = ""
    if len(tokens_per_frame) != n_frames:
        reason = (
            f"{len(tokens_per_frame)} visual-atom spans vs {n_frames} evaluation frames; "
            "span-to-image grouping is not in the packed tokens"
        )
    elif len(grids) != n_frames:
        reason = f"{len(grids)} grid_thws rows vs {n_frames} frames"
    elif any(len(row) < 3 for row in grids):
        reason = "grid_thws rows do not have temporal, height, and width"
    else:
        patch = spec.get("patch_size")
        merge = spec.get("merge_size") or spec.get("spatial_merge_size") or spec.get("hidden_stride")
        if not patch:
            reason = "visual tokenizer config has no patch_size; not inferring a grid from token count"
        else:
            resize_fn = None if model is None else _find_smart_resize(model)
            if resize_fn is None:
                reason = "tokenizer has patch_size but no smart_resize callable to confirm letterboxing"
            else:
                views_per_frame = []
                confirmed = True
                for index, (grid, count, (height, width)) in enumerate(
                    zip(grids, tokens_per_frame, frame_sizes)
                ):
                    temporal, grid_h, grid_w = int(grid[0]), int(grid[1]), int(grid[2])
                    if temporal != 1:
                        confirmed = False
                        reason = f"frame {index} temporal grid {temporal} is not a single image"
                        break
                    token_grid = None
                    if count == grid_h * grid_w:
                        token_h, token_w = grid_h, grid_w
                        factor = int(patch)
                    elif merge and count == (grid_h // int(merge)) * (grid_w // int(merge)):
                        token_h = grid_h // int(merge)
                        token_w = grid_w // int(merge)
                        factor = int(patch) * int(merge)
                    else:
                        confirmed = False
                        reason = (
                            f"frame {index} has {count} visual atoms but grid {[temporal, grid_h, grid_w]} "
                            "does not match the tokenizer patch/merge product"
                        )
                        break
                    resized = _call_smart_resize(
                        resize_fn, height, width, factor, min_pixels, max_pixels
                    )
                    expect = (grid_h * int(patch), grid_w * int(patch))
                    if resized != expect:
                        confirmed = False
                        reason = (
                            f"frame {index} smart_resize returned {resized}, "
                            f"expected resized canvas {expect} from grid and patch_size"
                        )
                        break
                    token_grid = (token_h, token_w)
                    views_per_frame.append(
                        [
                            {
                                "view_id": f"frame{index}/full",
                                "view_kind": "full",
                                "grid_hw": list(token_grid),
                                "crop_box": [0, 0, width, height],
                                "resized_hw": [int(resized[0]), int(resized[1])],
                                "letterbox": [0, 0, 0, 0],
                            }
                        ]
                    )
                if confirmed:
                    geometry = build_geometry_from_views(
                        [{"height": h, "width": w} for h, w in frame_sizes],
                        views_per_frame,
                        pmax=max(tokens_per_frame),
                        mapping_status="verified_direct_resize",
                        synthetic=False,
                        notes=[
                            "Direct resize confirmed by the visual tokenizer smart_resize. "
                            "Letterbox is zero because the resize canvas matches grid * patch_size."
                        ],
                    )
                    geometry["raw_grid_thws"] = grids
                    geometry["tokenizer_spec"] = spec
                    return geometry
    return unresolved_geometry(
        n_frames=n_frames,
        tokens_per_frame=list(tokens_per_frame) if len(tokens_per_frame) == n_frames else [0] * n_frames,
        frame_sizes=frame_sizes,
        grid_thws=grids,
        reason=reason or "spatial mapping could not be confirmed",
        spec=spec,
    )
