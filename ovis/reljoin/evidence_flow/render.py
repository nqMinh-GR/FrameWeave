"""One TSO question: heatmap panels plus a standalone TikZ figure."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CVPR_BLUE = (54, 125, 189)
AMBER = "E65100"
AMBER_SOFT = "FB8C00"
GREEN = "1B8A5A"
CRIMSON = "C62828"
BLUE = "367DBD"
FIGURE_WIDTH_CM = 17.4


def render_case(result: dict, tex_path: Path, pdf_path: Path, *, allow_candidate: bool = False) -> dict:
    bundle = result["bundle"]
    if not bundle.get("render_allowed") and not allow_candidate:
        raise RuntimeError(bundle.get("reject_reason") or "This question is not eligible to render.")
    if "chosen_trial" not in bundle:
        raise RuntimeError(bundle.get("reject_reason") or "This question has no measured trial to draw.")
    frames = result["frames"]
    trial = bundle["chosen_trial"]
    keyframes = bundle["selection"]["keyframes"]
    panel_dir = pdf_path.parent / f"{bundle['question_id']}_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    for frame_index, tokens in zip(keyframes, trial["token_sets"]):
        image = frames[frame_index]
        alpha = np.asarray(bundle["alpha"][frame_index], dtype=np.float64)
        boxes = np.asarray(bundle["boxes"][frame_index], dtype=np.float64)
        panel_path = panel_dir / f"frame_{frame_index:02d}.png"
        _write_panel(image, alpha, boxes, tokens, panel_path)
        panels.append(panel_path)
    tex = _tikz(bundle, panels)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex, encoding="utf-8")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    _compile(tex_path, pdf_path)
    preview = pdf_path.with_suffix(".png")
    _preview(pdf_path, preview)
    return {"tex": str(tex_path), "pdf": str(pdf_path), "preview": str(preview)}


def _write_panel(image: Image.Image, alpha: np.ndarray, boxes: np.ndarray, tokens: list[int], path: Path) -> None:
    base = image.convert("RGBA")
    valid = alpha > 0
    if not bool(valid.any()):
        raise RuntimeError("A displayed frame has no pooling mass.")
    median = float(np.median(alpha[valid]))
    peak = float(alpha[valid].max())
    span = max(peak - median, 1e-8)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = base.size
    for index, weight in enumerate(alpha):
        if index >= len(boxes) or weight <= median:
            continue
        x0, y0, x1, y1 = boxes[index]
        strength = min(0.42, 0.42 * ((weight - median) / span))
        draw.rectangle(
            [x0, y0, x1, y1],
            fill=(CVPR_BLUE[0], CVPR_BLUE[1], CVPR_BLUE[2], int(255 * strength)),
        )
    composed = Image.alpha_composite(base, overlay).convert("RGB")
    scale = max(1, int(round(1400 / max(width, 1))))
    if scale > 1:
        composed = composed.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(path)


def _tikz(bundle: dict, panels: list[Path]) -> str:
    trial = bundle["chosen_trial"]
    keyframes = bundle["selection"]["keyframes"]
    stamps = bundle["timestamps"]
    sample = Image.open(panels[0])
    aspect = sample.size[0] / sample.size[1]
    sample.close()
    n_frames = len(panels)
    frames_width = 11.7
    gap = 0.28
    frame_w = (frames_width - gap * (n_frames - 1)) / n_frames
    frame_h = frame_w / aspect
    callout_h = 1.05
    timeline_h = 0.62
    card_block = 7.15
    question = _escape(bundle.get("question") or "")
    question_lines = max(1, int(np.ceil(len(question) / 105)))
    question_h = 0.34 * question_lines + 0.45
    frame_top = 0.2 + max(card_block, callout_h + frame_h)
    frame_bottom = frame_top - frame_h
    n_edges = max(1, len(bundle.get("displayed_edges") or []))
    arrow_band = 0.46 * n_edges + 0.12
    timeline_bottom = frame_top + arrow_band + 0.22
    question_bottom = timeline_bottom + timeline_h + 0.16
    figure_h = question_bottom + question_h + 0.12
    cards_x = 12.15
    lines = [
        r"\documentclass[tikz,border=1pt]{standalone}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage{graphicx}",
        r"\usetikzlibrary{arrows.meta}",
        rf"\definecolor{{cvprblue}}{{HTML}}{{{BLUE}}}",
        rf"\definecolor{{amber}}{{HTML}}{{{AMBER}}}",
        rf"\definecolor{{ambersoft}}{{HTML}}{{{AMBER_SOFT}}}",
        rf"\definecolor{{goldok}}{{HTML}}{{{GREEN}}}",
        rf"\definecolor{{vanillabad}}{{HTML}}{{{CRIMSON}}}",
        r"\begin{document}",
        rf"\begin{{tikzpicture}}[x=1cm,y=1cm,>=Stealth]",
        rf"\node[anchor=north west,text width=17.1cm,font=\sffamily\footnotesize,align=left] at (0.1,{figure_h - 0.08:.3f}) {{\textbf{{{_escape(bundle['question_id'])} --- Temporal Shot Ordering{_candidate_tag(bundle)}}}\\{question}}};",
    ]
    _timeline(lines, stamps, keyframes, y=timeline_bottom + 0.08, width=frames_width)
    rects = []
    for slot, (frame_index, panel) in enumerate(zip(keyframes, panels)):
        x0 = slot * (frame_w + gap)
        rects.append((x0, frame_bottom, frame_w, frame_h, frame_index))
        lines.append(
            rf"\node[anchor=south west,inner sep=0] at ({x0:.3f},{frame_bottom:.3f}) "
            rf"{{\includegraphics[width={frame_w:.3f}cm]{{{panel.resolve()}}}}};"
        )
        lines.append(
            rf"\node[anchor=north west,fill=white,fill opacity=0.92,text opacity=1,inner sep=1.2pt,"
            rf"font=\sffamily\scriptsize] at ({x0 + 0.04:.3f},{frame_top - 0.04:.3f}) "
            rf"{{$t={stamps[frame_index]:.1f}\,\mathrm{{s}}$}};"
        )
        _boxes(lines, bundle, frame_index, trial["token_sets"][slot], x0, frame_bottom, frame_w, frame_h)
        impact = trial["token_impacts"][slot]
        lines.append(
            rf"\node[anchor=north,text width={frame_w:.3f}cm,align=center,font=\sffamily\scriptsize] "
            rf"at ({x0 + frame_w / 2:.3f},{frame_bottom - 0.04:.3f}) "
            rf"{{\(\uparrow\) Top-\(\alpha\) contribution\\ \(\Delta z_y={_signed(impact['delta_z'])}\)"
            rf"\\ (margin {_signed(impact['delta_margin'])})}};"
        )
    _arrows(lines, bundle, rects, frame_top)
    _cards(lines, bundle, cards_x, frame_top)
    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def _timeline(lines: list[str], stamps: list[float], keyframes: list[int], y: float, width: float) -> None:
    lines.append(rf"\draw[cvprblue!40,line width=0.6pt] (0.15,{y:.3f}) -- ({width - 0.15:.3f},{y:.3f});")
    n = len(stamps)
    for index, stamp in enumerate(stamps):
        x = 0.15 + (width - 0.3) * (index / max(n - 1, 1))
        if index in keyframes:
            lines.append(rf"\fill[cvprblue] ({x:.3f},{y:.3f}) circle (1.6pt);")
            lines.append(
                rf"\node[above,font=\sffamily\scriptsize,text=cvprblue] at ({x:.3f},{y + 0.03:.3f}) {{$t_{{{index + 1}}}$}};"
            )
        else:
            lines.append(rf"\fill[black!45] ({x:.3f},{y:.3f}) circle (0.9pt);")


def _boxes(lines, bundle, frame_index, tokens, x0, y0, frame_w, frame_h) -> None:
    image_w = None
    boxes = np.asarray(bundle["boxes"][frame_index], dtype=np.float64)
    if boxes.size == 0:
        return
    # Original pixel width is the right edge of the last geometry, but use the
    # maximum x1/y1 as the image extent implied by the verified boxes.
    image_w = float(np.max(boxes[:, 2]))
    image_h = float(np.max(boxes[:, 3]))
    chosen = [boxes[index] for index in tokens if index < len(boxes)]
    for box in chosen:
        fx0, fy0, fx1, fy1 = _map_box(box, image_w, image_h, x0, y0, frame_w, frame_h)
        lines.append(
            rf"\draw[cvprblue,line width=0.7pt] ({fx0:.3f},{fy0:.3f}) rectangle ({fx1:.3f},{fy1:.3f});"
        )


def _map_box(box, image_w, image_h, x0, y0, frame_w, frame_h):
    px0, py0, px1, py1 = box
    fx0 = x0 + (px0 / image_w) * frame_w
    fx1 = x0 + (px1 / image_w) * frame_w
    fy0 = y0 + (1.0 - py1 / image_h) * frame_h
    fy1 = y0 + (1.0 - py0 / image_h) * frame_h
    return fx0, fy0, fx1, fy1


def _candidate_tag(bundle: dict) -> str:
    if bundle.get("render_allowed", True):
        return ""
    return " (candidate)"


def _arrows(lines, bundle, rects, frame_top: float) -> None:
    lookup = {item[4]: item for item in rects}
    edges = bundle.get("displayed_edges") or []
    if not edges:
        return
    peak = max(edge["beta"] for edge in edges)
    for ordinal, edge in enumerate(edges):
        source = lookup.get(edge["source"])
        receiver = lookup.get(edge["receiver"])
        if source is None or receiver is None:
            continue
        color = "amber" if ordinal == 0 else "ambersoft"
        width = 0.7 + 1.8 * (edge["beta"] / max(peak, 1e-6))
        x_start, y_start = _anchor(source, toward=receiver[0])
        x_end, y_end = _anchor(receiver, toward=source[0])
        control_y = frame_top + 0.28 + 0.46 * ordinal
        pre = edge["pre_softmax"]
        label = rf"\(\beta={edge['beta']:.2f},\ \Delta z_y={_signed(pre['delta_z'])}\)"
        mid_x = (x_start + x_end) / 2
        lines.append(
            rf"\draw[white,line width={width + 1.3:.2f}pt] ({x_start:.3f},{y_start:.3f}) "
            rf".. controls ({mid_x:.3f},{control_y:.3f}) .. ({x_end:.3f},{y_end:.3f});"
        )
        lines.append(
            rf"\draw[{color},line width={width:.2f}pt,->] ({x_start:.3f},{y_start:.3f}) "
            rf".. controls ({mid_x:.3f},{control_y:.3f}) .. ({x_end:.3f},{y_end:.3f});"
        )
        lines.append(
            rf"\node[fill=white,inner sep=1.2pt,font=\sffamily\scriptsize] at "
            rf"({mid_x:.3f},{control_y:.3f}) {{{label}}};"
        )


def _anchor(rect, toward: float) -> tuple[float, float]:
    x0, bottom, width, height, _frame = rect
    y = bottom + 0.62 * height
    if toward >= x0:
        return x0 + width, y
    return x0, y


def _cards(lines, bundle, x, top) -> None:
    vanilla = bundle["vanilla"]
    learned = bundle["frameweave"]
    gold = str(bundle.get("gold") or learned.get("letter") or "")
    vanilla_letter = str(vanilla.get("predicted_letter") or vanilla["letter"])
    learned_letter = str(learned.get("predicted_letter") or learned["letter"])
    interpretation = bundle["interpretation"]["lines"]
    body = r"\\".join(_escape(line) for line in interpretation)
    lines.append(
        rf"\node[anchor=north west,draw=cvprblue!70,rounded corners=1.5pt,inner sep=4pt,"
        rf"text width=4.9cm,font=\sffamily\scriptsize,align=left] at ({x:.3f},{top:.3f}) "
        rf"{{\textbf{{Visual reading of high-impact regions}}\\{body}}};"
    )
    answer = (
        rf"\textbf{{Answer and gold-logit shift}}\\"
        rf"Vanilla: {_letter_mark(vanilla_letter, gold)} "
        rf"{_escape(_short(vanilla['text']))}\\"
        rf"\(z^0_y={vanilla['z_gold']:.2f}\), prob \({vanilla['prob_gold']:.2f}\)\\"
        rf"FrameWeave: {_letter_mark(learned_letter, gold)} "
        rf"{_escape(_short(learned['text']))}\\"
        rf"\(z^\theta_y={learned['z_gold']:.2f}\), "
        rf"\(\Delta z_y={_signed(bundle['overall_delta_z'])}\)\\"
        rf"\(\Delta\mathrm{{margin}}={_signed(bundle['overall_delta_margin'])}\)"
    )
    lines.append(
        rf"\node[anchor=north west,draw=black!30,rounded corners=1.5pt,inner sep=4pt,"
        rf"text width=4.9cm,font=\sffamily\scriptsize,align=left] at ({x:.3f},{top - 3.55:.3f}) {{{answer}}};"
    )


def _letter_mark(letter: str, gold: str) -> str:
    shown = _escape(letter or "?")
    if gold and letter == gold:
        return rf"\textcolor{{goldok}}{{\(\checkmark\) {shown}}}"
    return rf"\textcolor{{vanillabad}}{{\(\times\) {shown}}}"


def _signed(value: float) -> str:
    number = float(value)
    return f"+{number:.2f}" if number >= 0 else f"{number:.2f}"


def _short(text: str, limit: int = 72) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _escape(text: str) -> str:
    mapping = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(mapping.get(char, char) for char in str(text))


def _compile(tex_path: Path, pdf_path: Path) -> None:
    import tempfile

    if shutil.which("pdflatex") is None:
        raise RuntimeError("pdflatex is not available, so the TikZ figure was not compiled.")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="evidence_flow_tex_") as tmp:
        work = Path(tmp)
        local_tex = work / tex_path.name
        shutil.copyfile(tex_path, local_tex)
        completed = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", local_tex.name],
            cwd=work,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            log = (completed.stdout or "") + (completed.stderr or "")
            raise RuntimeError(log[-4000:])
        shutil.copyfile(work / f"{tex_path.stem}.pdf", pdf_path)


def render_layout_smoke(out_dir: Path) -> None:
    """Compile a labeled synthetic figure outside the paper tree.

    This checks TikZ only. It is not an evidence-flow result and must not be
    copied into CVPR/figures.
    """
    out_dir = Path(out_dir)
    if out_dir.name != "evidence_flow_layout_smoke":
        raise RuntimeError("layout smoke refuses to write outside its temporary directory")
    if any(part == "CVPR" for part in out_dir.parts):
        raise RuntimeError("layout smoke must not write into the paper tree")
    width, height = 640, 360
    frames = []
    for index in range(3):
        image = Image.new("RGB", (width, height), (226, 226, 226))
        draw = ImageDraw.Draw(image)
        draw.rectangle([20 + 40 * index, 20, 120 + 40 * index, 120], fill=(80, 80, 80))
        draw.text((12, 12), "SYNTHETIC", fill=(20, 20, 20))
        frames.append(image)
    cell_w, cell_h = width / 6, height / 4
    boxes = []
    for _frame in range(8):
        frame_boxes = []
        for row in range(4):
            for col in range(6):
                frame_boxes.append(
                    [col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h]
                )
        boxes.append(frame_boxes)
    alpha = [[0.0] * 24 for _frame in range(8)]
    for frame in (1, 3, 6):
        alpha[frame][0] = 0.42
        alpha[frame][1] = 0.31
        alpha[frame][7] = 0.18
    bundle = {
        "question_id": "TSO_LAYOUT_SMOKE",
        "question": (
            "SYNTHETIC LAYOUT CHECK. This page is not a model result, not a HERBench frame, "
            "and must not be installed in the paper."
        ),
        "timestamps": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "alpha": alpha,
        "boxes": boxes,
        "selection": {"keyframes": [1, 3, 6]},
        "chosen_trial": {
            "token_sets": [[0, 1, 7], [0, 1, 7], [0, 1, 7]],
            "token_impacts": [
                {"delta_z": 0.21, "delta_margin": 0.14},
                {"delta_z": 0.18, "delta_margin": 0.11},
                {"delta_z": 0.16, "delta_margin": 0.09},
            ],
        },
        "displayed_edges": [
            {
                "receiver": 6,
                "source": 1,
                "beta": 0.61,
                "pre_softmax": {"delta_z": 0.27},
            },
            {
                "receiver": 3,
                "source": 1,
                "beta": 0.22,
                "pre_softmax": {"delta_z": 0.08},
            },
        ],
        "vanilla": {
            "letter": "C",
            "text": "synthetic wrong order",
            "z_gold": 1.10,
            "prob_gold": 0.21,
        },
        "frameweave": {
            "letter": "D",
            "text": "synthetic gold order",
            "z_gold": 1.55,
            "prob_gold": 0.44,
        },
        "overall_delta_z": 0.45,
        "overall_delta_margin": 0.33,
        "interpretation": {
            "lines": [
                "Focus patches: synthetic cells, not a visual reading",
                "Observed order: layout only",
                "Relation direction: source frame to receiver frame",
            ]
        },
        "render_allowed": True,
    }
    render_case(
        {"bundle": bundle, "frames": [frames[0]] * 8},
        out_dir / "qual_evidence_flow_LAYOUT_SMOKE.tex",
        out_dir / "LAYOUT_SMOKE.pdf",
    )


def _preview(pdf_path: Path, png_path: Path) -> None:
    tool = shutil.which("pdftoppm")
    if tool is None:
        return
    stem = png_path.with_suffix("")
    subprocess.run(
        [tool, "-png", "-r", "300", "-singlefile", str(pdf_path), str(stem)],
        check=True,
        capture_output=True,
    )
