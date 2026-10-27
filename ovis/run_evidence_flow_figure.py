#!/usr/bin/env python3
"""Build HERBench TSO evidence-flow figures outside the paper.

Figures, TeX, and JSON are written under ``--paper``, which is an output
directory and is not the paper tree. This command never edits the paper.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path("/home/quangminh/Documents/DATN/Thí Nghiệm")
PAPER = Path("/tmp/evidence_flow_out")


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
    extract.add_argument("--flips", action="store_true")
    extract.add_argument("--live", action="store_true")
    extract.add_argument("--example", action="store_true")
    extract.add_argument("--all-tso", action="store_true")
    extract.add_argument("--draw-all", action="store_true")
    extract.add_argument("--skip-existing", action="store_true")
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
        raise SystemExit(
            "Paper install is disabled. Figures stay in the output directory so they can be chosen by hand."
        )
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


def _write_lite(workspace: Path, paper: Path) -> dict:
    from reljoin.evidence_flow.mine import EXPECTED_SHA256, default_paths, load_seed42_lite
    from reljoin.evidence_flow.signals import write_json
    from reljoin.herbench_oracle import sha256_file

    report = load_seed42_lite(workspace, paper)
    checkpoint = default_paths(workspace, paper)["checkpoint"]
    if checkpoint.is_file():
        digest = sha256_file(checkpoint)
        report["checkpoint_sha256_actual"] = digest
        report["checkpoint_sha256_matches"] = digest == EXPECTED_SHA256
    path = paper / "audit" / "qual_lite_questions.json"
    write_json(path, report)
    print(f"Wrote {path} n_questions={report['n_questions']}", flush=True)
    return report


def _write_rank(workspace: Path, paper: Path) -> dict:
    from reljoin.evidence_flow.mine import EXPECTED_SHA256, default_paths, rank_seed42_flips
    from reljoin.evidence_flow.signals import write_json
    from reljoin.herbench_oracle import sha256_file

    report = rank_seed42_flips(workspace, paper)
    checkpoint = default_paths(workspace, paper)["checkpoint"]
    if checkpoint.is_file():
        digest = sha256_file(checkpoint)
        report["checkpoint_sha256_actual"] = digest
        report["checkpoint_sha256_matches"] = digest == EXPECTED_SHA256
    path = paper / "audit" / "qual_candidates_ranked.json"
    write_json(path, report)
    print(
        f"Wrote {path} n_questions={report['n_questions']} n_flips={report['n_flips']} "
        f"tso_flips={report['n_tso_flips']}",
        flush=True,
    )
    return report


def _check(paper: Path) -> None:
    sys.path.insert(0, str(WORKSPACE / "v6_ovis"))
    from reljoin.evidence_flow.checks import run_intervention_checks, run_static_checks

    del paper
    run_static_checks(None)
    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP: intervention checks need PyTorch", flush=True)
        return
    run_intervention_checks()


def _extract(args) -> None:
    sys.path.insert(0, str(args.workspace / "v6_ovis"))
    from reljoin.evidence_flow.mine import default_paths
    from reljoin.evidence_flow.signals import RuntimeBlocked, load_runtime, write_json

    if args.checkpoint is None:
        args.checkpoint = default_paths(args.workspace, args.paper)["checkpoint"]
    if getattr(args, "live", False):
        report = _write_lite(args.workspace, args.paper)
        cases = list(report.get("all_questions") or [])
        requested = str(getattr(args, "q_id", "") or "")
        if requested:
            cases = [row for row in cases if row["question_id"] == requested]
        if not cases:
            raise SystemExit("The seed-42 lite table has no question to score.")
        try:
            runtime = load_runtime(args)
        except RuntimeBlocked as exc:
            print("Stopped before any question.", flush=True)
            for blocker in exc.blockers:
                print(f"- {blocker}", flush=True)
            raise SystemExit(2) from exc
        _extract_many(args, report, cases, True, runtime, write_json)
        return
    report_path = args.paper / "audit" / "qual_candidates_ranked.json"
    report = {}
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("n_flips") or not report.get("all_questions"):
        report = _write_rank(args.workspace, args.paper)
    if getattr(args, "flips", False):
        args.example = True
    draw_all = bool(
        getattr(args, "draw_all", False)
        or getattr(args, "all_tso", False)
        or getattr(args, "flips", False)
    )
    if getattr(args, "flips", False) or getattr(args, "all_tso", False):
        cases = list(report.get("all_questions") or [])
        if not cases:
            raise SystemExit("The rank file has no Vanilla-wrong Hybrid-correct questions.")
        try:
            runtime = load_runtime(args)
        except RuntimeBlocked as exc:
            print("Stopped before any heatmap.", flush=True)
            for blocker in exc.blockers:
                print(f"- {blocker}", flush=True)
            raise SystemExit(2) from exc
        _extract_many(args, report, cases, draw_all, runtime, write_json)
        return
    question_id = _select_id(
        report, getattr(args, "q_id", ""), getattr(args, "allow_abstention", False), args.paper
    )
    case = _find_case(report, question_id)
    try:
        _extract_one(args, case, draw_all, None)
    except RuntimeBlocked as exc:
        _write_blocked(args.paper, report, case, exc.blockers, write_json)
        print("Stopped before any heatmap.", flush=True)
        for blocker in exc.blockers:
            print(f"- {blocker}", flush=True)
        raise SystemExit(2) from exc


def _extract_many(args, report, cases, draw_all, runtime, write_json) -> None:
    import traceback

    from reljoin.evidence_flow.signals import RuntimeBlocked

    summary = []
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        question_id = case["question_id"]
        pdf = args.paper / "figures" / "qual_evidence_flow" / f"{question_id}.pdf"
        audit = args.paper / "audit" / f"qual_evidence_flow_{question_id}.json"
        full_png = args.paper / "figures" / "full" / f"{question_id}.png"
        if _keep_existing(args, pdf, audit, full_png):
            print(f"[{index}/{total}] skip {question_id}", flush=True)
            saved = json.loads(audit.read_text(encoding="utf-8"))
            summary.append({
                "question_id": question_id,
                "status": "skipped",
                "render_allowed": saved.get("render_allowed"),
                "qualified": bool(saved.get("qualified")),
                "pdf": pdf.is_file(),
                "predicted_vanilla": (saved.get("vanilla") or {}).get("letter"),
                "predicted_frameweave": (saved.get("frameweave") or {}).get("letter"),
                "gold": saved.get("gold"),
            })
            continue
        print(f"[{index}/{total}] {question_id}", flush=True)
        try:
            bundle = _extract_one(args, case, draw_all, runtime)
            summary.append({
                "question_id": question_id,
                "status": "ok",
                "render_allowed": bundle.get("render_allowed"),
                "pdf": pdf.is_file(),
                "reject_reason": bundle.get("reject_reason") or "",
                "qualified": bool(bundle.get("qualified")),
                "predicted_vanilla": (bundle.get("vanilla") or {}).get("predicted_letter"),
                "predicted_frameweave": (bundle.get("frameweave") or {}).get("predicted_letter"),
                "gold": bundle.get("gold"),
            })
        except RuntimeBlocked as exc:
            _write_blocked(args.paper, report, case, exc.blockers, write_json)
            print(f"{question_id} blocked: {exc.blockers}", flush=True)
            summary.append({"question_id": question_id, "status": "blocked", "blockers": exc.blockers, "pdf": False})
        except Exception as exc:
            traceback.print_exc()
            _write_blocked(
                args.paper, report, case, [f"{type(exc).__name__}: {exc}"], write_json,
                note=traceback.format_exc(),
            )
            print(f"{question_id} failed: {type(exc).__name__}: {exc}", flush=True)
            summary.append({
                "question_id": question_id,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "pdf": False,
            })
    path = args.paper / "audit" / "qual_evidence_flow_summary.json"
    counts = {
        "ok": sum(row["status"] == "ok" for row in summary),
        "skipped": sum(row["status"] == "skipped" for row in summary),
        "blocked": sum(row["status"] == "blocked" for row in summary),
        "error": sum(row["status"] == "error" for row in summary),
        "pdf": sum(bool(row.get("pdf")) for row in summary),
        "qualified": sum(bool(row.get("qualified")) for row in summary),
    }
    full_dir = args.paper / "figures" / "full"
    n_full = len(list(full_dir.glob("*.png"))) if full_dir.is_dir() else 0
    write_json(path, {"n_questions": total, "n_full_images": n_full, "counts": counts, "results": summary})
    print(
        f"Finished {total} questions. Full images {n_full} in {full_dir}. Summary {path} counts={counts}",
        flush=True,
    )
    if counts["pdf"] == 0 and counts["error"] == total:
        raise SystemExit(1)


def _keep_existing(args, pdf: Path, audit: Path, full_png: Path | None = None) -> bool:
    """Skip a finished question. A live run only skips its own completed result."""
    if not (getattr(args, "skip_existing", False) and audit.is_file()):
        return False
    try:
        saved = json.loads(audit.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if getattr(args, "live", False):
        if not saved.get("live"):
            return False
        if not saved.get("qualified"):
            return True
        if saved.get("layout") != "token_row8_pairs":
            return False
        return pdf.is_file() and full_png is not None and full_png.is_file()
    if not pdf.is_file():
        return False
    if getattr(args, "example", False) and not saved.get("example"):
        return False
    return True


def _extract_one(args, case, draw_all: bool, runtime) -> dict:
    from reljoin.evidence_flow.render import render_case
    from reljoin.evidence_flow.signals import extract_question, write_json

    question_id = case["question_id"]
    result = extract_question(case, args, runtime=runtime)
    bundle = result["bundle"]
    audit = args.paper / "audit" / f"qual_evidence_flow_{question_id}.json"
    write_json(audit, bundle)
    if not result["render"]:
        print(f"No figure for {question_id}: {bundle.get('reject_reason')}", flush=True)
        return bundle
    if not (bundle.get("render_allowed") or draw_all):
        print(f"No figure for {question_id}: {bundle.get('reject_reason')}", flush=True)
        return bundle
    tex = args.paper / "fig" / f"qual_evidence_flow_{question_id}.tex"
    pdf = args.paper / "figures" / "qual_evidence_flow" / f"{question_id}.pdf"
    outputs = render_case(
        result, tex, pdf, allow_candidate=not bool(bundle.get("render_allowed"))
    )
    full_dir = _copy_full_image(pdf)
    print(f"Wrote {outputs['pdf']}", flush=True)
    print(f"Full image {full_dir / pdf.name}", flush=True)
    return bundle


def _copy_full_image(pdf: Path) -> Path:
    import shutil

    full_dir = pdf.parents[1] / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, full_dir / pdf.name)
    preview = pdf.with_suffix(".png")
    if preview.is_file():
        shutil.copy2(preview, full_dir / preview.name)
    return full_dir


def _write_blocked(paper, report, case, blockers, write_json, note: str = "") -> None:
    question_id = case["question_id"]
    status = {
        "question_id": question_id,
        "render_allowed": False,
        "blockers": blockers,
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
        "note": note or "Stopped before any heatmap. No PDF was written.",
    }
    write_json(paper / "audit" / f"qual_evidence_flow_{question_id}.status.json", status)


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
    pools = (
        list(report.get("all_questions") or [])
        + list(report.get("concrete_wrong") or [])
        + list(report.get("abstention_flips_top10") or [])
    )
    for row in pools:
        if row["question_id"] == question_id:
            return row
    raise SystemExit(f"{question_id} is not in the ranked TSO tables.")


if __name__ == "__main__":
    main()
