#!/usr/bin/env python3
"""Build one HERBench TSO evidence-flow figure at a time.

The default command ranks the 152 TSO questions, checks the merge-order
arithmetic, then tries to extract the next concrete-wrong question. If Ovis
weights, CUDA, or the video are missing, it stops without drawing a figure.
``install`` is the only command that edits the paper.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path("/home/quangminh/Documents/DATN/Thí Nghiệm")
PAPER = Path("/home/quangminh/Documents/DATN/CVPR")


def main() -> None:
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--paper", type=Path, default=PAPER)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("rank")
    sub.add_parser("check")

    extract = sub.add_parser("extract")
    _add_runtime(extract)
    extract.add_argument("--q-id", default="")
    extract.add_argument("--allow-abstention", action="store_true")

    render = sub.add_parser("render")
    render.add_argument("--q-id", required=True)

    install = sub.add_parser("install")
    install.add_argument("--q-id", required=True)

    run = sub.add_parser("run")
    _add_runtime(run)
    run.add_argument("--q-id", default="")
    run.add_argument("--allow-abstention", action="store_true")

    args = parser.parse_args()
    command = args.command or "run"
    if command == "rank":
        _write_rank(args.workspace, args.paper)
    elif command == "check":
        _check(args.paper)
    elif command == "extract":
        _extract(args)
    elif command == "render":
        _render_saved(args.paper, args.q_id)
    elif command == "install":
        from reljoin.evidence_flow.install import install_question

        install_question(args.paper, args.q_id)
    elif command == "run":
        _write_rank(args.workspace, args.paper)
        _check(args.paper)
        _extract(args)
    else:
        parser.error(f"unknown command {command}")


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--videos-root", default="")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=Path("/tmp/evidence_flow_root"))


def _write_rank(workspace: Path, paper: Path) -> dict:
    from reljoin.evidence_flow.mine import EXPECTED_SHA256, default_paths, rank_tso_candidates
    from reljoin.evidence_flow.signals import write_json
    from reljoin.herbench_oracle import sha256_file

    report = rank_tso_candidates(workspace, paper)
    checkpoint = default_paths(workspace, paper)["checkpoint"]
    if checkpoint.is_file():
        digest = sha256_file(checkpoint)
        report["checkpoint_sha256_actual"] = digest
        report["checkpoint_sha256_matches"] = digest == EXPECTED_SHA256
    path = paper / "audit" / "qual_candidates_ranked.json"
    write_json(path, report)
    print(
        f"Wrote {path} concrete={report['n_concrete_wrong']} "
        f"queue={report['render_queue']}",
        flush=True,
    )
    return report


def _check(paper: Path) -> None:
    sys.path.insert(0, str(WORKSPACE / "v6_ovis"))
    from reljoin.evidence_flow.checks import run_intervention_checks, run_static_checks

    run_static_checks(paper)
    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP: intervention checks need PyTorch", flush=True)
        return
    run_intervention_checks()


def _extract(args) -> None:
    sys.path.insert(0, str(args.workspace / "v6_ovis"))
    from reljoin.evidence_flow.mine import default_paths
    from reljoin.evidence_flow.signals import RuntimeBlocked, extract_question, write_json

    report_path = args.paper / "audit" / "qual_candidates_ranked.json"
    if not report_path.is_file():
        _write_rank(args.workspace, args.paper)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    question_id = _select_id(report, getattr(args, "q_id", ""), getattr(args, "allow_abstention", False), args.paper)
    case = _find_case(report, question_id)
    if args.checkpoint is None:
        args.checkpoint = default_paths(args.workspace, args.paper)["checkpoint"]
    try:
        result = extract_question(case, args)
    except RuntimeBlocked as exc:
        status = {
            "question_id": question_id,
            "render_allowed": False,
            "blockers": exc.blockers,
            "checkpoint_sha256_expected": report.get("checkpoint_sha256_expected"),
            "checkpoint_sha256_actual": report.get("checkpoint_sha256_actual"),
            "checkpoint_sha256_matches": report.get("checkpoint_sha256_matches"),
            "vanilla_letter": case.get("vanilla_letter"),
            "gold": case.get("gold"),
            "hybrid_letter": case.get("hybrid_letter"),
            "frame_timestamps": case.get("frame_timestamps"),
            "evidence_spans": case.get("evidence_spans"),
            "frame_source_labels": case.get("frame_source_labels"),
            "seeds_hybrid_correct": case.get("seeds_hybrid_correct"),
            "timestamps_match_sampler": case.get("timestamps_match_sampler"),
            "localization": {
                "status": "not_run",
                "reason": "The impulse test requires the local Ovis2.5-9B weights, CUDA, and the HERBench video.",
            },
            "signals": None,
            "note": "Stopped before any heatmap. No paper PDF was written.",
        }
        path = args.paper / "audit" / f"qual_evidence_flow_{question_id}.status.json"
        write_json(path, status)
        print("Stopped before any heatmap.", flush=True)
        for blocker in exc.blockers:
            print(f"- {blocker}", flush=True)
        raise SystemExit(2) from exc
    bundle = result["bundle"]
    audit = args.paper / "audit" / f"qual_evidence_flow_{question_id}.json"
    write_json(audit, bundle)
    if not result["render"]:
        print(f"No figure for {question_id}: {bundle.get('reject_reason')}", flush=True)
        return
    from reljoin.evidence_flow.render import render_case

    tex = args.paper / "fig" / f"qual_evidence_flow_{question_id}.tex"
    pdf = args.paper / "figures" / "qual_evidence_flow" / f"{question_id}.pdf"
    outputs = render_case(result, tex, pdf)
    print(f"Wrote {outputs['pdf']}", flush=True)


def _render_saved(paper: Path, question_id: str) -> None:
    raise SystemExit(
        "Render runs from a live extraction in this process. "
        f"Use extract --q-id {question_id} so the panels and tensors come from the same forward."
    )


def _select_id(report: dict, requested: str, allow_abstention: bool, paper: Path) -> str:
    if requested:
        return requested
    queue = list(report.get("render_queue") or [])
    if allow_abstention:
        queue.extend(row["question_id"] for row in report.get("abstention_flips_top10") or [])
    for question_id in queue:
        pdf = paper / "figures" / "qual_evidence_flow" / f"{question_id}.pdf"
        if not pdf.is_file():
            return question_id
    if not queue:
        raise SystemExit("No eligible TSO question is in the render queue.")
    return queue[0]


def _find_case(report: dict, question_id: str) -> dict:
    pools = list(report.get("concrete_wrong") or []) + list(report.get("abstention_flips_top10") or [])
    for row in pools:
        if row["question_id"] == question_id:
            return row
    raise SystemExit(f"{question_id} is not in the ranked TSO tables.")


if __name__ == "__main__":
    main()
