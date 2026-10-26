"""Install one chosen TSO figure into the paper. Does nothing unless asked."""
from __future__ import annotations

import subprocess
from pathlib import Path

EXPERIMENTS_INCLUDE = r"""\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/qual_evidence_flow.pdf}
\vspace{-0.8em}
\caption{\textbf{Qualitative evidence flow on one HERBench lite\_v2 temporal-shot-ordering question (Ovis2.5-9B).}
Inference uses the published eight chronological frames; the timeline marks the displayed keyframes.
$\alpha_{ip}$ is the intra-frame pooling weight (\cref{eq:pool}).
An arrow from frame $j$ to frame $i$ is weighted by $\beta_{ij}$ (\cref{eq:global}), the receiver-$i$ softmax over source frames $j\neq i$.
The contribution is $\Delta z_y=z_y(\mathrm{full})-z_y(\mathrm{intervened})$, so a positive number supports the gold letter.
Boxes are merged-token regions whose pixel alignment passed the impulse test.
The summary lines are a human reading of those regions, not a probed symbolic state.
\texttt{__QID__}: Vanilla predicts \texttt{__VANILLA__} and FrameWeave predicts gold \texttt{__GOLD__}, with gold-logit gain $\Delta z_y=__DZ__$ and margin gain $\Delta m_y=__DM__$.}
\label{fig:qual_main}
\vspace{-0.4em}
\end{figure*}
"""

OLD_INCLUDE_START = r"\includegraphics[width=\textwidth]{figures/qual_cross_frame_annotated.pdf}"

OLD_SUPPLEMENT = """The available geometry supports nominal-grid visualizations but does not establish a complete token-to-original-pixel mapping.
The collector records $\\mathrm{grid\\_thws}=(1,20,36)$ and 180 tokens per frame, without the complete resize and crop correspondence to the original images.
Accordingly, the overlays in \\cref{fig:qual_main} use a nominal 10-by-18 merged-patch grid: highlighted cells are approximate token locations, with no verified anatomical or object correspondence.
The supplementary view in \\cref{fig:qual} instead reports token indices and frame probabilities without spatial overlays.
Neither view establishes a causal explanation for an answer."""

NEW_SUPPLEMENT = """\\Cref{fig:qual_main} uses merged-token boxes checked by an impulse through the Ovis2.5 preprocessor, vision tower, and $2\\times 2$ merger, and it reports adapter-internal counterfactual contributions $\\Delta z_y=z_y(\\mathrm{full})-z_y(\\mathrm{intervened})$.
The summary sentences are a human reading of those verified regions, not a claim that the adapter stores symbolic object or state slots.
Inference still consumes all eight chronological frames.
The supplementary view in \\cref{fig:qual} is unchanged: it reports cached token indices and frame probabilities, without spatial overlays and without a causal contribution."""

OLD_BODY = r"""\Cref{fig:qual_main} illustrates the routing distributions for two selected successes.
In \texttt{TSO\_0179} (\cref{fig:qual_main}a), the receiver token at $60.6$\,s assigns its largest frame weights to woodland shots at $43.0$--$43.3$\,s ($\gamma=0.22$ each) and a silo shot at $93.2$\,s ($\gamma=0.16$); Vanilla abstains, whereas \ours selects the correct ordering.
In \texttt{MEGL\_124} (\cref{fig:qual_main}b), the receiver at $5.2$\,s assigns its largest weights to frames at $118.0$\,s, $121.2$\,s, and $6.6$\,s ($\gamma=0.29,0.23,0.21$).
The nominal-grid overlays do not verify spatial or object correspondences, and these routing patterns do not establish why the answers change.
Additional routing diagnostics and the mapping limitation appear in \cref{sec:supp_qual}."""

NEW_BODY = r"""\Cref{fig:qual_main} shows one temporal-shot-ordering question. Inference uses all eight published frames; the figure draws the keyframes whose regions and directed reads have positive gold-logit contributions $\Delta z_y=z_y(\mathrm{full})-z_y(\mathrm{intervened})$.
The boxes are merged-token regions checked against the Ovis preprocessor, and each arrow from frame $j$ to frame $i$ uses $\beta_{ij}$.
The summary lines are a human reading of those regions.
Cached routing bars that do not make this counterfactual claim remain in \cref{fig:qual,sec:supp_qual}."""


def install_question(paper: Path, question_id: str) -> None:
    import json
    import shutil

    audit = paper / "audit" / f"qual_evidence_flow_{question_id}.json"
    pdf = paper / "figures" / "qual_evidence_flow" / f"{question_id}.pdf"
    if not audit.is_file() or not pdf.is_file():
        raise FileNotFoundError(
            f"Refusing to edit the paper. Missing {audit.name} or {pdf.name}. "
            "Render passes only after the localization test and positive contributions."
        )
    bundle = json.loads(audit.read_text(encoding="utf-8"))
    if not bundle.get("render_allowed"):
        raise RuntimeError("Refusing to install a question that failed the evidence-flow filters.")
    if not bundle.get("localization", {}).get("status") == "pass":
        raise RuntimeError("Refusing to install before the live pixel-to-token test has passed.")
    target = paper / "figures" / "qual_evidence_flow.pdf"
    shutil.copyfile(pdf, target)
    _replace_figure(paper / "sec" / "4_experiments.tex", bundle)
    _replace_body(paper / "sec" / "4_experiments.tex")
    _replace_supplement(paper / "sec" / "X_suppl.tex")
    subprocess.run(
        [
            "latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
            "-outdir=build", "main.tex",
        ],
        cwd=paper,
        check=True,
    )


def _replace_figure(path: Path, bundle: dict) -> None:
    text = path.read_text(encoding="utf-8")
    if OLD_INCLUDE_START not in text and "qual_evidence_flow.pdf" not in text:
        raise RuntimeError(f"{path} does not contain fig:qual_main.")
    vanilla = bundle["vanilla"]["letter"]
    gold = bundle["frameweave"]["letter"]
    caption = (
        EXPERIMENTS_INCLUDE
        .replace("__QID__", _tex_id(bundle["question_id"]))
        .replace("__VANILLA__", vanilla)
        .replace("__GOLD__", gold)
        .replace("__DZ__", f"{float(bundle['overall_delta_z']):+.2f}")
        .replace("__DM__", f"{float(bundle['overall_delta_margin']):+.2f}")
    )
    start = text.find(r"\begin{figure*}[t]")
    end = text.find(r"\end{figure*}", start)
    if start < 0 or end < 0:
        raise RuntimeError("Could not locate the qualitative figure environment.")
    # Keep the first figure* only when it is fig:qual_main.
    block = text[start : end + len(r"\end{figure*}")]
    if "fig:qual_main" not in block:
        raise RuntimeError("The first figure* is not fig:qual_main.")
    path.write_text(text[:start] + caption.strip() + "\n" + text[end + len(r"\end{figure*}") :], encoding="utf-8")


def _replace_body(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if NEW_BODY in text:
        return
    if OLD_BODY not in text:
        raise RuntimeError("The main-text qualitative paragraph was not found, so it was left unchanged.")
    path.write_text(text.replace(OLD_BODY, NEW_BODY, 1), encoding="utf-8")


def install_anchors_present(paper: Path) -> None:
    """Read-only check that --install can find the paragraphs it is allowed to replace."""
    experiments = (paper / "sec" / "4_experiments.tex").read_text(encoding="utf-8")
    supplement = (paper / "sec" / "X_suppl.tex").read_text(encoding="utf-8")
    if OLD_INCLUDE_START not in experiments and "qual_evidence_flow.pdf" not in experiments:
        raise AssertionError("fig:qual_main include was not found")
    if OLD_BODY not in experiments and NEW_BODY not in experiments:
        raise AssertionError("main-text qualitative paragraph was not found")
    if OLD_SUPPLEMENT not in supplement and NEW_SUPPLEMENT not in supplement:
        raise AssertionError("supplement geometry paragraph was not found")


def _replace_supplement(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if NEW_SUPPLEMENT in text:
        return
    if OLD_SUPPLEMENT not in text:
        raise RuntimeError("The supplement geometry paragraph was not found, so it was left unchanged.")
    path.write_text(text.replace(OLD_SUPPLEMENT, NEW_SUPPLEMENT, 1), encoding="utf-8")


def _tex_id(question_id: str) -> str:
    return question_id.replace("_", r"\_")
