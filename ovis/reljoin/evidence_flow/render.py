"""Paper row: small question, four choices, two answers, eight token frames."""
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
LINE_CM = 0.282


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
    header, header_lines = _header(bundle, question)
    edges = [
        edge for edge in (bundle.get("displayed_edges") or [])
        if int(edge["source"]) in keyframes and int(edge["receiver"]) in keyframes
    ]
    header_h = LINE_CM * header_lines
    arrow_band = 0.16 if not edges else 0.70
    bottom = 0.04
    frame_bottom = bottom
    frame_top = frame_bottom + frame_h
    figure_h = 0.02 + header_h + 0.04 + arrow_band + frame_h + bottom
    header_y = figure_h - 0.01
    text = r"font=\fontsize{7}{8}\selectfont\sffamily,align=left,inner sep=0pt"
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
        rf"\node[anchor=north west,text width={CONTENT_WIDTH_CM:.2f}cm,{text}] "
        rf"at ({LEFT_CM:.2f},{header_y:.3f}) {{{header}}};",
    ]
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
        _token_boxes(lines, bundle, frame_index, x0, frame_bottom, frame_w, frame_h)
        lines.append(
            rf"\node[anchor=south west,fill=white,inner sep=0.4pt,font=\sffamily\tiny] "
            rf"at ({x0 + 0.02:.3f},{frame_bottom + 0.02:.3f}) {{${_stamp(stamp)}\,\mathrm{{s}}$}};"
        )
    _arrows(lines, edges, rects, frame_top)
    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def _header(bundle: dict, question: str) -> tuple[str, int]:
    """Question, then two tight lines of two choices, then the two model answers."""
    question_lines = _estimate_lines(f"{bundle['question_id']} {question}", 128)
    choices = _shown_choices(bundle)
    rows = []
    row_lines = 0
    for start in (0, 2):
        pair = [item for item in choices[start:start + 2] if item]
        rows.append(r"\quad ".join(_escape(item) for item in pair))
        row_lines += _estimate_lines("  ".join(pair), 140) if pair else 1
    answer = _answer_line(bundle)
    body = (
        rf"\textbf{{{_escape(bundle['question_id'])}}} {_escape(question)}\\[0pt]"
        rf"{rows[0]}\\[0pt]{rows[1]}\\[0pt]{answer}"
    )
    return body, question_lines + row_lines + 1


def _shown_choices(bundle: dict) -> list[str]:
    shown = []
    for index, raw in enumerate(list(bundle.get("choices") or [])[:4]):
        letter = chr(ord("A") + index)
        text = _plain(raw)
        prefix = f"{letter}. "
        if not (text.upper().startswith(prefix.upper()) or text.upper().startswith(f"{letter} ")):
            text = prefix + text
        shown.append(text)
    return shown


def _answer_line(bundle: dict) -> str:
    learned = _letter(bundle.get("frameweave")) or "?"
    vanilla = _letter(bundle.get("vanilla")) or "?"
    gold = str(bundle.get("gold") or "").strip().upper()[:1]
    true = " (true)" if gold and learned == gold else ""
    return (
        rf"\textcolor{{goldok}}{{FrameWeave: {_escape(learned)}{_escape(true)}}}"
        rf" \quad \textcolor{{vanillabad}}{{Vanilla: {_escape(vanilla)}}}"
    )


def _letter(block) -> str:
    if not isinstance(block, dict):
        return ""
    return str(block.get("letter") or "").strip().upper()[:1]


def _frame_geometry(n_frames: int, aspect: float) -> tuple[float, float, float]:
    gap = 0.06 if n_frames >= 6 else 0.12
    frame_w = (CONTENT_WIDTH_CM - gap * max(n_frames - 1, 0)) / max(n_frames, 1)
    frame_h = frame_w / max(aspect, 0.2)
    return frame_w, frame_h, gap


def _token_boxes(lines, bundle, frame_index, x0, y0, frame_w, frame_h) -> None:
    boxes = bundle.get("boxes") or []
    token_sets = bundle.get("token_sets") or []
    sizes = bundle.get("image_sizes") or []
    if frame_index >= len(boxes) or frame_index >= len(token_sets):
        return
    frame_boxes = boxes[frame_index]
    if not frame_boxes:
        return
    if frame_index < len(sizes) and len(sizes[frame_index]) == 2:
        image_w, image_h = float(sizes[frame_index][0]), float(sizes[frame_index][1])
    else:
        image_w = max(float(box[2]) for box in frame_boxes)
        image_h = max(float(box[3]) for box in frame_boxes)
    if image_w <= 0 or image_h <= 0:
        return
    for index in token_sets[frame_index]:
        if not isinstance(index, int) or index < 0 or index >= len(frame_boxes):
            continue
        fx0, fy0, fx1, fy1 = _map_box(frame_boxes[index], image_w, image_h, x0, y0, frame_w, frame_h)
        lines.append(
            rf"\draw[cvprblue,line width=0.8pt] ({fx0:.3f},{fy0:.3f}) rectangle ({fx1:.3f},{fy1:.3f});"
        )


def _map_box(box, image_w, image_h, x0, y0, frame_w, frame_h):
    px0, py0, px1, py1 = [float(value) for value in box]
    fx0 = x0 + (px0 / image_w) * frame_w
    fx1 = x0 + (px1 / image_w) * frame_w
    fy0 = y0 + (1.0 - py1 / image_h) * frame_h
    fy1 = y0 + (1.0 - py0 / image_h) * frame_h
    return fx0, fy0, fx1, fy1


def _arrows(lines: list[str], edges: list[dict], rects: list[tuple], frame_top: float) -> None:
    lookup = {item[4]: item for item in rects}
    for ordinal, edge in enumerate(edges):
        source = lookup.get(int(edge["source"]))
        receiver = lookup.get(int(edge["receiver"]))
        if source is None or receiver is None or int(edge["source"]) == int(edge["receiver"]):
            continue
        color = "amber" if ordinal % 2 == 0 else "ambersoft"
        x_start, y_start = _anchor(source, toward=receiver[0], ordinal=ordinal, incoming=False)
        x_end, y_end = _anchor(receiver, toward=source[0], ordinal=ordinal, incoming=True)
        span = abs(x_end - x_start)
        control_y = frame_top + 0.18 + 0.16 * (ordinal % 3) + min(0.16, 0.012 * span)
        mid_x = (x_start + x_end) / 2
        lines.append(
            rf"\draw[{color},line width=0.9pt,->] ({x_start:.3f},{y_start:.3f}) "
            rf".. controls ({mid_x:.3f},{control_y:.3f}) .. ({x_end:.3f},{y_end:.3f});"
        )


def _anchor(rect, toward: float, ordinal: int = 0, incoming: bool = False) -> tuple[float, float]:
    x0, bottom, width, height, _frame = rect
    y = bottom + height
    inset = min(max(width * 0.22, 0.18), 0.55)
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
    for index in range(8):
        image = Image.new("RGB", (width, height), (226, 226, 226))
        draw = ImageDraw.Draw(image)
        draw.rectangle([20 + 30 * index, 20, 90 + 30 * index, 90], fill=(80, 80, 80))
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
    bundle = {
        "example": True,
        "layout": "token_row8_pairs",
        "question_id": "TSO_LAYOUT_SMOKE",
        "question": (
            "SYNTHETIC LAYOUT CHECK for a 100% compact example. "
            "This page is not a model result and must not be installed in the paper."
        ),
        "choices": [
            "A. 1->2->3->4",
            "B. 4->3->2->1",
            "C. 4->3->1->2",
            "D. 2->1->4->3",
            "E. At least two descriptions do not match the video.",
        ],
        "gold": "D",
        "timestamps": [12.0, 23.3, 37.5, 48.0, 62.0, 80.0, 90.0, 105.7],
        "selection": {"keyframes": list(range(8))},
        "displayed_edges": [
            {"receiver": 7, "source": 0},
            {"receiver": 7, "source": 3},
            {"receiver": 4, "source": 1},
        ],
        "boxes": boxes,
        "token_sets": [[0, 7], [1, 8], [2], [3, 9], [4], [5, 10], [6], [0, 11]],
        "image_sizes": [[width, height] for _frame in range(8)],
        "vanilla": {"letter": "E", "text": "abstention"},
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
    for required in ("TSO_LAYOUT_SMOKE", "Vanilla: E", "FrameWeave: D (true)", "1->2->3->4", "cvprblue"):
        if required not in tex:
            raise AssertionError(f"compact layout is missing {required}")
    if "At least two descriptions" in tex:
        raise AssertionError("the fifth choice should stay off the paper row")
    for forbidden in (r"\beta", r"\Delta", "candidate", "Visual reading", "Top-", "margin", "t_1"):
        if forbidden in tex:
            raise AssertionError(f"compact layout still contains {forbidden}")
    pair_break = tex.find(r"A. 1->2->3->4")
    second_row = tex.find(r"C. 4->3->1->2")
    if pair_break < 0 or second_row < pair_break or r"\\[0pt]" not in tex[pair_break:second_row]:
        raise AssertionError("choices must be two tight lines of two answers")
    if tex.count(r"\includegraphics") != 8:
        raise AssertionError("the paper row must show all 8 frames")
    if tex.count("rectangle") < 8:
        raise AssertionError("token boxes were not drawn")


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
