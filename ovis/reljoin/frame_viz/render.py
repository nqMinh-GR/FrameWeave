"""Offline figures. This module does not load Ovis or the adapter."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import is_verified, paint_frame_map, routing_components

MAX_SHOWN = 4


def load_example(example_dir: Path) -> dict:
    example_dir = Path(example_dir)
    meta = json.loads((example_dir / "metadata.json").read_text(encoding="utf-8"))
    geometry = json.loads((example_dir / "token_geometry.json").read_text(encoding="utf-8"))
    queries = json.loads((example_dir / "queries.json").read_text(encoding="utf-8"))
    predictions = json.loads((example_dir / "predictions.json").read_text(encoding="utf-8"))
    arrays = np.load(example_dir / "tensors.npz", allow_pickle=False)
    frames = []
    frame_dir = example_dir / "frames"
    if frame_dir.is_dir():
        for path in sorted(frame_dir.glob("frame_*.png")):
            frames.append(path)
    return {
        "dir": example_dir,
        "meta": meta,
        "geometry": geometry,
        "queries": queries,
        "predictions": predictions,
        "arrays": arrays,
        "frame_paths": frames,
    }


def render_example(example_dir: Path) -> dict:
    bundle = load_example(example_dir)
    stats = _figure_stats(bundle)
    _write_captions(bundle, stats)
    _figure_a(bundle, stats)
    _figure_b(bundle, stats)
    _figure_c(bundle, stats)
    _geometry_debug(bundle, stats)
    return stats


def _batch0(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim >= 1 and array.shape[0] == 1:
        return array[0]
    return array


def _figure_stats(bundle: dict) -> dict:
    arrays = bundle["arrays"]
    geometry = bundle["geometry"]
    queries = bundle["queries"].get("queries") or []
    beta = _batch0(arrays["beta"])
    frame_mask = np.asarray(_batch0(arrays["frame_mask"])).astype(bool)
    token_mask = np.asarray(_batch0(arrays["token_mask"])).astype(bool)
    spatial = np.asarray(geometry["spatial_token_mask"], dtype=bool)
    k = int(beta.shape[0])
    timestamps = list(bundle["meta"].get("timestamps") or [])
    primary = queries[0] if queries else None
    gamma_row = np.zeros(k, dtype=np.float64)
    pi_row = np.zeros((k, token_mask.shape[-1]), dtype=np.float64)
    if primary is not None and "gamma_selected" in arrays and arrays["gamma_selected"].shape[1] > 0:
        gamma_row = np.asarray(_batch0(arrays["gamma_selected"])[0], dtype=np.float64)
        pi_row = np.asarray(_batch0(arrays["pi_selected"])[0], dtype=np.float64)
    receiver = int(primary["frame_index"]) if primary else 0
    shown, omitted_mass = _shown_frames(receiver, gamma_row, frame_mask)
    parts = routing_components(gamma_row, pi_row, token_mask, spatial)
    return {
        "k": k,
        "timestamps": timestamps,
        "frame_mask": frame_mask,
        "token_mask": token_mask,
        "spatial": spatial,
        "primary": primary,
        "gamma_row": gamma_row,
        "pi_row": pi_row,
        "routing": parts,
        "shown": shown,
        "omitted_mass": float(omitted_mass),
        "receiver": receiver,
        "verified": is_verified(str(geometry.get("mapping_status"))),
        "synthetic": bool(geometry.get("synthetic")),
    }


def _shown_frames(receiver: int, gamma_row: np.ndarray, frame_mask: np.ndarray) -> tuple[list[int], float]:
    k = int(gamma_row.shape[0])
    valid = [j for j in range(k) if j < len(frame_mask) and frame_mask[j] and j != receiver]
    ranked = sorted(valid, key=lambda j: (-float(gamma_row[j]), j))
    targets = []
    for frame_j in ranked:
        if 1 + len(targets) >= MAX_SHOWN:
            break
        targets.append(frame_j)
    shown = sorted(set([receiver] + targets))
    shown = [j for j in shown if j < len(frame_mask) and frame_mask[j]]
    omitted = [j for j in valid if j not in shown]
    omitted_mass = float(sum(float(gamma_row[j]) for j in omitted))
    return shown, omitted_mass


def _write_captions(bundle: dict, stats: dict) -> None:
    meta = bundle["meta"]
    example_id = meta.get("example_id", "")
    status = meta.get("mapping_status") or bundle["geometry"].get("mapping_status")
    n_shown = len(stats["shown"])
    n_frames = int(stats["k"])
    omitted = stats["omitted_mass"]
    nonspatial = stats["routing"]["nonspatial_mass"]
    synthetic = " Synthetic fixture; not a HERBench result." if stats["synthetic"] else ""
    figure_a = (
        f"Figure A. Cross-frame local routing for example {example_id}. "
        f"The highlighted token is the receiver query. Target heatmaps are pi, the local "
        f"token match, and R = gamma * pi, the routing mass. gamma under each target is "
        f"that query's frame-read weight. Arrows run from the frame being read to the "
        f"receiver; line width follows gamma and is not an effect on the answer logit. "
        f"Shown {n_shown}/{n_frames} frames. Omitted routing mass is {omitted:.4f} and "
        f"nonspatial routing mass is {nonspatial:.4f}; neither was renormalized onto the "
        f"drawn frames. Mapping status: {status}.{synthetic} "
        f"This example is a selected case, not a statistical summary of the benchmark."
    )
    figure_b = (
        f"Figure B. Question-guided pooling and relative token correction for example {example_id}. "
        f"Alpha is the question-conditioned pooling weight inside each frame. "
        f"Eta is the relative residual correction, RMS of the effective delivered update "
        f"divided by RMS of the original projected token. Alpha and eta use separate color "
        f"scales. A gate, when present in the module statistics, is a modulation factor "
        f"and is not drawn as attention.{synthetic}"
    )
    text = f"## Figure A\n\n{figure_a}\n\n## Figure B\n\n{figure_b}\n"
    (bundle["dir"] / "captions_en.md").write_text(text, encoding="utf-8")
    stats["caption_a"] = figure_a
    stats["caption_b"] = figure_b


def _figure_a(bundle: dict, stats: dict) -> None:
    plt = _plt()
    shown = stats["shown"]
    n = max(1, len(shown))
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 7.2), squeeze=False)
    _header(fig, bundle, stats, "Cross-frame local routing")
    gamma_row = stats["gamma_row"]
    pi_row = stats["pi_row"]
    routing = stats["routing"]["R"]
    vmax_pi = float(np.max(pi_row)) if pi_row.size else 1.0
    vmax_r = float(np.max(routing)) if routing.size else 1.0
    images_for_bar = []
    for col, frame_j in enumerate(shown):
        _draw_map(
            axes[0, col],
            bundle,
            frame_j,
            pi_row[frame_j],
            vmin=0.0,
            vmax=max(vmax_pi, 1e-8),
            title=_frame_title(bundle, frame_j, stats),
            cmap="inferno",
            mode="mass",
            separate_thumbnail=True,
        )
        axes[0, col].set_xlabel(f"gamma={float(gamma_row[frame_j]):.3f}")
        artist = _draw_map(
            axes[1, col],
            bundle,
            frame_j,
            routing[frame_j],
            vmin=0.0,
            vmax=max(vmax_r, 1e-8),
            title="R = gamma * pi",
            cmap="inferno",
            mode="mass",
            separate_thumbnail=True,
        )
        images_for_bar.append(artist)
        if frame_j == stats["receiver"] and stats["primary"] is not None:
            _mark_query(axes[0, col], bundle, stats)
    _maybe_colorbar(fig, axes[0, -1], images_for_bar[-1] if images_for_bar else None, "pi / R")
    _routing_footer(fig, stats)
    _save(fig, bundle["dir"] / "figure_cross_frame")
    plt.close(fig)


def _figure_b(bundle: dict, stats: dict) -> None:
    plt = _plt()
    arrays = bundle["arrays"]
    alpha = np.asarray(_batch0(arrays["alpha"]), dtype=np.float64)
    eta = np.asarray(_batch0(arrays["eta"]), dtype=np.float64)
    frame_mask = stats["frame_mask"]
    columns = [j for j in range(stats["k"]) if frame_mask[j]]
    n = max(1, len(columns))
    fig, axes = plt.subplots(3, n, figsize=(2.6 * n, 8.4), squeeze=False)
    _header(fig, bundle, stats, "Where to read and where to update")
    alpha_max = float(alpha[frame_mask].max()) if frame_mask.any() else 1.0
    eta_max = float(eta[frame_mask].max()) if frame_mask.any() else 1.0
    alpha_artist = None
    eta_artist = None
    for col, frame_j in enumerate(columns):
        _show_rgb(axes[0, col], bundle, frame_j)
        axes[0, col].set_title(_frame_title(bundle, frame_j, stats), fontsize=8)
        alpha_artist = _draw_map(
            axes[1, col], bundle, frame_j, alpha[frame_j],
            vmin=0.0, vmax=max(alpha_max, 1e-8), title="alpha",
            cmap="viridis", mode="mean", separate_thumbnail=False,
        )
        eta_artist = _draw_map(
            axes[2, col], bundle, frame_j, eta[frame_j],
            vmin=0.0, vmax=max(eta_max, 1e-8), title="eta",
            cmap="magma", mode="mean", separate_thumbnail=False,
        )
    _maybe_colorbar(fig, axes[1, -1], alpha_artist, "alpha")
    _maybe_colorbar(fig, axes[2, -1], eta_artist, "eta")
    fig.text(
        0.01, 0.01,
        "alpha: question-guided pooling (coverage-weighted mean).  "
        "eta: relative residual magnitude, not a probability.  "
        "Color scales are not shared.",
        fontsize=8,
    )
    _save(fig, bundle["dir"] / "figure_read_update")
    plt.close(fig)


def _figure_c(bundle: dict, stats: dict) -> None:
    plt = _plt()
    beta = np.asarray(_batch0(bundle["arrays"]["beta"]), dtype=np.float64)
    frame_mask = stats["frame_mask"]
    shown = np.array(beta, copy=True)
    invalid = ~frame_mask
    shown[invalid, :] = np.nan
    shown[:, invalid] = np.nan
    eye = np.eye(shown.shape[0], dtype=bool)
    shown[eye] = np.nan
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    finite = shown[np.isfinite(shown)]
    vmax = float(finite.max()) if finite.size else 1.0
    image = axes[0].imshow(shown, cmap="cividis", vmin=0.0, vmax=max(vmax, 1e-8))
    labels = [_axis_label(bundle, j, stats) for j in range(stats["k"])]
    axes[0].set_xticks(range(stats["k"]), labels, rotation=45, ha="right", fontsize=7)
    axes[0].set_yticks(range(stats["k"]), labels, fontsize=7)
    axes[0].set_xlabel("frame read j")
    axes[0].set_ylabel("receiver frame i")
    axes[0].set_title("beta")
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    row_sums = np.nansum(np.where(np.isnan(shown), 0.0, shown), axis=1)
    axes[0].text(
        0.0, -0.35, "row sums: " + ", ".join(f"{v:.3f}" for v in row_sums),
        transform=axes[0].transAxes, fontsize=8,
    )
    gamma_row = stats["gamma_row"]
    axes[1].plot(gamma_row, label="gamma of primary query")
    if stats["primary"] is not None:
        receiver = stats["receiver"]
        axes[1].plot(beta[receiver], label=f"beta row {receiver}")
    axes[1].set_xticks(range(stats["k"]), labels, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].set_title("local gamma beside global beta")
    _header(fig, bundle, stats, "Frame read matrix")
    _save(fig, bundle["dir"] / "figure_frame_matrix")
    plt.close(fig)


def _geometry_debug(bundle: dict, stats: dict) -> None:
    plt = _plt()
    from matplotlib.patches import Rectangle

    geometry = bundle["geometry"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    status = geometry.get("mapping_status")
    if stats["synthetic"]:
        ax.set_title(f"SYNTHETIC geometry debug — {status}")
    else:
        ax.set_title(f"geometry debug — {status}")
    if not stats["verified"]:
        ax.imshow(stats["token_mask"], aspect="auto", cmap="gray_r")
        ax.set_xlabel("token index")
        ax.set_ylabel("frame index")
        ax.text(
            0.5, -0.18, "spatial_mapping_unverified — token index only, not an image overlay",
            transform=ax.transAxes, ha="center", fontsize=8,
        )
    else:
        frame_j = stats["receiver"]
        _show_rgb(ax, bundle, frame_j)
        for token in geometry["tokens"]:
            if int(token["frame_index"]) != frame_j or not token["spatial"] or not token["bbox"]:
                continue
            x0, y0, x1, y1 = token["bbox"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=0.4))
        _mark_query(ax, bundle, stats)
        for view in geometry["frames"][frame_j]["views"]:
            box = view.get("crop_box")
            if not box:
                continue
            ax.add_patch(
                Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    fill=False, edgecolor="cyan", linewidth=1.2,
                )
            )
    _save(fig, bundle["dir"] / "geometry_debug")
    plt.close(fig)


def contact_sheet(output_dir: Path) -> Path:
    from .survey import contact_sort_key

    plt = _plt()
    output_dir = Path(output_dir)
    index_path = output_dir / "examples_index.csv"
    rows = []
    if index_path.is_file():
        import csv
        with index_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows.sort(key=contact_sort_key)
    n = max(1, len(rows))
    cols = 4 if n > 1 else 1
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.4 * cols, 3.2 * rows_n), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    if not rows:
        axes[0, 0].text(0.5, 0.5, "no examples", ha="center")
    for index, row in enumerate(rows):
        ax = axes[index // cols, index % cols]
        ax.axis("on")
        frame = Path(row.get("frame_path") or "")
        if frame.is_file():
            ax.imshow(_read_rgb(frame))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{row.get('role', '')}  {row.get('example_id', '')}\n"
            f"gold {row.get('gold', '')}  V {row.get('vanilla', '')}  F {row.get('frameweave', '')}\n"
            f"{row.get('mapping_status', '')}",
            fontsize=7,
        )
    fig.suptitle("FrameWeave V6 visualization contact sheet")
    path = output_dir / "contact_sheet.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw_map(ax, bundle, frame_j, values, *, vmin, vmax, title, cmap, mode, separate_thumbnail):
    values = np.asarray(values, dtype=np.float64)
    verified = is_verified(str(bundle["geometry"].get("mapping_status")))
    image = _frame_array(bundle, frame_j)
    artist = None
    if verified and image is not None:
        ax.imshow(image)
        painted = paint_frame_map(
            bundle["geometry"], frame_j, values, mode=mode, separate_thumbnail=separate_thumbnail,
        )
        canvas = painted["canvas"]
        mask = painted["weight"] > 0
        show = np.ma.array(canvas, mask=~mask)
        artist = ax.imshow(show, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.72)
    else:
        artist = ax.imshow(values.reshape(1, -1), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_ylabel("spatial_mapping_unverified", fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return artist


def _mark_query(ax, bundle, stats) -> None:
    from matplotlib.patches import Rectangle

    primary = stats["primary"]
    if primary is None:
        return
    if not stats["verified"]:
        ax.scatter([int(primary["token_index"])], [0], c="cyan", s=30)
        return
    for token in bundle["geometry"]["tokens"]:
        if int(token["frame_index"]) != int(primary["frame_index"]):
            continue
        if int(token["token_index"]) != int(primary["token_index"]):
            continue
        box = token.get("bbox")
        if not box:
            return
        ax.add_patch(
            Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                fill=False, edgecolor="cyan", linewidth=1.8,
            )
        )
        return


def _show_rgb(ax, bundle, frame_j) -> None:
    image = _frame_array(bundle, frame_j)
    if image is None:
        ax.imshow(np.zeros((8, 8, 3), dtype=np.uint8))
        ax.set_ylabel("no frame", fontsize=7)
    else:
        ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])


def _frame_array(bundle, frame_j):
    paths = bundle["frame_paths"]
    if frame_j >= len(paths):
        return None
    return _read_rgb(paths[frame_j])


def _read_rgb(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _frame_title(bundle, frame_j, stats) -> str:
    stamp = ""
    times = stats["timestamps"]
    if frame_j < len(times):
        stamp = f" t={float(times[frame_j]):.2f}s"
    indices = bundle["meta"].get("frame_indices") or []
    index = indices[frame_j] if frame_j < len(indices) else frame_j
    return f"frame {frame_j} idx {index}{stamp}"


def _axis_label(bundle, frame_j, stats) -> str:
    times = stats["timestamps"]
    if frame_j < len(times):
        return f"{frame_j}\n{float(times[frame_j]):.2f}s"
    return str(frame_j)


def _header(fig, bundle, stats, title: str) -> None:
    meta = bundle["meta"]
    pred = bundle["predictions"]
    question = str(meta.get("question") or "").replace("\n", " ")
    if len(question) > 180:
        question = question[:177] + "..."
    vanilla = (pred.get("vanilla") or {}).get("parsed", "")
    frameweave = (pred.get("frameweave") or {}).get("parsed", "")
    banner = "SYNTHETIC  " if stats["synthetic"] else ""
    if not stats["verified"]:
        banner += "spatial_mapping_unverified  "
    fig.suptitle(
        f"{banner}{title}\n{question}\n"
        f"gold={meta.get('gold', '')}  vanilla={vanilla}  frameweave={frameweave}",
        fontsize=9,
    )


def _routing_footer(fig, stats) -> None:
    shown = stats["shown"]
    fig.text(
        0.01, 0.01,
        f"shown_frames={shown} / total_frames={stats['k']}; "
        f"omitted routing mass={stats['omitted_mass']:.4f}; "
        f"nonspatial mass={stats['routing']['nonspatial_mass']:.4f}. "
        "Subset was not re-softmaxed. Arrows j -> i use gamma as line width only.",
        fontsize=8,
    )


def _maybe_colorbar(fig, ax, artist, label: str) -> None:
    if artist is None:
        return
    bar = fig.colorbar(artist, ax=ax, fraction=0.046)
    bar.set_label(label, fontsize=8)


def _save(fig, stem: Path) -> None:
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt
