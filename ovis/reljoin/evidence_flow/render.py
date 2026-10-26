"""Compact example figure: question, five choices, two answers, frames, and arrows."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

AMBER = "E65100"
AMBER_SOFT = "FB8C00"
GREEN = "1B8A5A"
CRIMSON = "C62828"
BLUE = "367DBD"
FIGURE_WIDTH_CM = 17.4
CONTENT_WIDTH_CM = 17.0
LEFT_CM = 0.2
LINE_CM = 0.36


def render_case(result: dict, tex_path: Path, pdf_path: Path, *, allow_candidate: bool = False) -> dict:
    bundle = result["bundle"]
    if not bundle.get("render_allowed") and not allow_candidate:
        raise RuntimeError(bundle.get("reject_reason") or "This question is not eligible to render.")
    frames = result["frames"]
    keyframes = [int(index) for index in bundle["selection"]["keyframes"]]
    if not keyframes:
        raise RuntimeError(bundle.get("reject_reason") or "This question has no frames to draw.")
    panel_dir = pdf_path.parent / f"{bundle['question_id']}_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    for frame_index in keyframes:
        panel_path = panel_dir / f"frame_{frame_index:02d}.png"
        _write_frame(frames[frame_index], panel_path)
        panels.append(panel_path)
    tex = _tikz(bundle, panels)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex, encoding="utf-8")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    _compile(tex_path, pdf_path)
    preview = pdf_path.with_suffix(".png")
    _preview(pdf_path, preview)
    return {"tex": str(tex_path), "pdf": str(pdf_path), "preview": str(preview)}


def _write_frame(image: Image.Image, path: Path) -> None:
    frame = image.convert("RGB")
    if frame.width > 960:
        height = max(1, int(round(frame.height * 960 / frame.width)))
        frame = frame.resize((960, height), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.save(path)


def _tikz(bundle: dict, panels: list[Path]) -> str:
    keyframes = [int(index) for index in bundle["selection"]["keyframes"]]
    stamps = [float(value) for value in bundle["timestamps"]]
    sample = Image.open(panels[0])
    aspect = sample.size[0] / max(sample.size[1], 1)
    sample.close()
    frame_w, frame_h, gap = _frame_geometry(len(panels), aspect)
    question = _plain(bundle.get("question") or "")
    choices = _choice_lines(bundle)
    edges = [
        edge for edge in (bundle.get("displayed_edges") or [])
        if int(edge["source"]) in keyframes and int(edge["receiver"]) in keyframes
    ][:2]
    question_lines = 1 + _estimate_lines(question, 92)
    question_h = LINE_CM * question_lines + 0.08
    option_heights = [LINE_CM * count + 0.05 for _text, count in choices]
    option_h = sum(option_heights)
    arrow_band = 0.18 if not edges else 0.72 + 0.34 * (len(edges) - 1)
    bottom = 0.12
    frame_bottom = bottom
    frame_top = frame_bottom + frame_h
    figure_h = 0.10 + question_h + option_h + 0.10 + arrow_band + frame_h + bottom
    question_y = figure_h - 0.08
    option_y = question_y - question_h
    lines = [
        r"\documentclass[tikz,border=1pt]{standalone}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{graphicx}",
        r"\usetikzlibrary{arrows.meta}",
        rf"\definecolor{{cvprblue}}{{HTML}}{{{BLUE}}}",
        rf"\definecolor{{amber}}{{HTML}}{{{AMBER}}}",
        rf"\definecolor{{ambersoft}}{{HTML}}{{{AMBER_SOFT}}}",
        rf"\definecolor{{goldok}}{{HTML}}{{{GREEN}}}",
        rf"\definecolor{{vanillabad}}{{HTML}}{{{CRIMSON}}}",
        r"\begin{document}",
        r"\begin{tikzpicture}[x=1cm,y=1cm,>=Stealth]",
        rf"\node[anchor=north west,text width={CONTENT_WIDTH_CM:.2f}cm,font=\sffamily\footnotesize,align=left] "
        rf"at ({LEFT_CM:.2f},{question_y:.3f}) {{\textbf{{{_escape(bundle['question_id'])}}}\\{_escape(question)}}};",
    ]
    for (text, _count), height in zip(choices, option_heights):
        lines.append(
            rf"\node[anchor=north west,text width={CONTENT_WIDTH_CM:.2f}cm,font=\sffamily\footnotesize,align=left] "
            rf"at ({LEFT_CM:.2f},{option_y:.3f}) {{{text}}};"
        )
        option_y -= height
    rects = []
    row_width = len(panels) * frame_w + max(len(panels) - 1, 0) * gap
    x_origin = LEFT_CM + max(0.0, (CONTENT_WIDTH_CM - row_width) / 2)
    for slot, (frame_index, panel) in enumerate(zip(keyframes, panels)):
        x0 = x_origin + slot * (frame_w + gap)
        rects.append((x0, frame_bottom, frame_w, frame_h, frame_index))
        stamp = stamps[frame_index]
        lines.append(
            rf"\node[anchor=south west,inner sep=0] at ({x0:.3f},{frame_bottom:.3f}) "
            rf"{{\includegraphics[width={frame_w:.3f}cm]{{{panel.resolve()}}}}};"
        )
        lines.append(
            rf"\node[anchor=south west,fill=white,inner sep=1pt,font=\sffamily\scriptsize] "
            rf"at ({x0 + 0.04:.3f},{frame_bottom + 0.04:.3f}) {{${_stamp(stamp)}\,\mathrm{{s}}$}};"
        )
    _arrows(lines, edges, rects, frame_top)
    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def _choice_lines(bundle: dict) -> list[tuple[str, int]]:
    vanilla = _letter(bundle.get("vanilla"))
    learned = _letter(bundle.get("frameweave"))
    rows = []
    for index, raw in enumerate(bundle.get("choices") or []):
        letter = chr(ord("A") + index)
        shown = _plain(raw)
        prefix = f"{letter}. "
        if not (shown.upper().startswith(prefix.upper()) or shown.upper().startswith(f"{letter} ")):
            shown = prefix + shown
        marks = []
        if letter == vanilla:
            marks.append(r"\textcolor{vanillabad}{Vanilla}")
        if letter == learned:
            marks.append(r"\textcolor{goldok}{FrameWeave}")
        body = _escape(shown)
        if marks:
            body = " ".join(marks) + r"\quad " + body
        width = 90 if marks else 98
        rows.append((body, _estimate_lines(shown, width)))
    return rows


def _letter(block) -> str:
    if not isinstance(block, dict):
        return ""
    return str(block.get("letter") or "").strip().upper()[:1]


def _frame_geometry(n_frames: int, aspect: float) -> tuple[float, float, float]:
    gap = 0.18
    frame_w = (CONTENT_WIDTH_CM - gap * (n_frames - 1)) / max(n_frames, 1)
    frame_h = frame_w / max(aspect, 0.2)
    if frame_h > 3.2:
        frame_h = 3.2
        frame_w = frame_h * aspect
        if n_frames > 1:
            gap = (CONTENT_WIDTH_CM - n_frames * frame_w) / (n_frames - 1)
            gap = max(gap, 0.12)
    return frame_w, frame_h, gap


def _arrows(lines: list[str], edges: list[dict], rects: list[tuple], frame_top: float) -> None:
    lookup = {item[4]: item for item in rects}
    for ordinal, edge in enumerate(edges[:2]):
        source = lookup.get(int(edge["source"]))
        receiver = lookup.get(int(edge["receiver"]))
        if source is None or receiver is None or int(edge["source"]) == int(edge["receiver"]):
            continue
        color = "amber" if ordinal == 0 else "ambersoft"
        x_start, y_start = _anchor(source, toward=receiver[0], ordinal=ordinal, incoming=False)
        x_end, y_end = _anchor(receiver, toward=source[0], ordinal=ordinal, incoming=True)
        span = abs(x_end - x_start)
        control_y = frame_top + max(0.72, 0.30 + 0.07 * span) + 0.28 * ordinal
        mid_x = (x_start + x_end) / 2
        lines.append(
            rf"\draw[{color},line width=1.15pt,->] ({x_start:.3f},{y_start:.3f}) "
            rf".. controls ({mid_x:.3f},{control_y:.3f}) .. ({x_end:.3f},{y_end:.3f});"
        )


def _anchor(rect, toward: float, ordinal: int = 0, incoming: bool = False) -> tuple[float, float]:
    x0, bottom, width, height, _frame = rect
    y = bottom + height
    inset = min(max(width * 0.28, 0.45), 1.35)
    if toward >= x0 + width / 2:
        x = x0 + width - inset
        if incoming:
            x -= 0.28 * ordinal
    else:
        x = x0 + inset
        if incoming:
            x += 0.28 * ordinal
    x = min(max(x, x0 + 0.25), x0 + width - 0.25)
    return x, y


def _estimate_lines(text: str, width_chars: int) -> int:
    words = str(text).split()
    if not words:
        return 1
    lines = 1
    column = 0
    for word in words:
        needed = len(word) if column == 0 else column + 1 + len(word)
        if column and needed > width_chars:
            lines += 1
            column = len(word)
        else:
            column = needed
    return lines


def _stamp(value: float) -> str:
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def _plain(text: str) -> str:
    table = {
        "→": "->",
        "←": "<-",
        "↔": "<->",
        "–": "-",
        "—": "-",
        "−": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "…": "...",
        "\u00a0": " ",
    }
    out = str(text)
    for source, target in table.items():
        out = out.replace(source, target)
    return " ".join(out.split())


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
    """Compile the compact example layout outside the paper tree."""
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
    bundle = {
        "example": True,
        "question_id": "TSO_LAYOUT_SMOKE",
        "question": (
            "SYNTHETIC LAYOUT CHECK for a 100% compact example. "
            "This page is not a model result and must not be installed in the paper."
        ),
        "choices": [
            "A. 1->2->3->4",
            "B. 4->3->2->1",
            "C. synthetic wrong order",
            "D. synthetic gold order",
            "E. At least two descriptions do not match the video.",
        ],
        "timestamps": [12.0, 37.5, 90.0],
        "selection": {"keyframes": [0, 1, 2]},
        "displayed_edges": [
            {"receiver": 2, "source": 0},
            {"receiver": 2, "source": 1},
        ],
        "vanilla": {"letter": "C", "text": "synthetic wrong order"},
        "frameweave": {"letter": "D", "text": "synthetic gold order"},
        "render_allowed": True,
    }
    tex_path = out_dir / "qual_evidence_flow_LAYOUT_SMOKE.tex"
    render_case(
        {"bundle": bundle, "frames": frames},
        tex_path,
        out_dir / "LAYOUT_SMOKE.pdf",
    )
    tex = tex_path.read_text(encoding="utf-8")
    for required in ("TSO_LAYOUT_SMOKE", "Vanilla", "FrameWeave", "1->2->3->4", "At least two descriptions"):
        if required not in tex:
            raise AssertionError(f"compact layout is missing {required}")
    for forbidden in (r"\beta", r"\Delta", "candidate", "Visual reading", "Top-", "margin", "t_1"):
        if forbidden in tex:
            raise AssertionError(f"compact layout still contains {forbidden}")
    if tex.count("->") < 2:
        raise AssertionError("compact layout did not draw both arrows")


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
