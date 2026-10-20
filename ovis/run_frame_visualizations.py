#!/usr/bin/env python3
"""FrameWeave Hybrid v6 qualitative figures on real HERBench frames.

Commands:
  audit          write implementation_audit.md
  cpu-checks     geometry, collector, and offline-render checks
  survey         choose the candidate list before any heatmap
  collect        run Hybrid.pt and write dumps
  render         draw figures from dumps, without loading Ovis
  contact-sheet  rewrite examples_index.csv and contact_sheet.png
  smoke          cpu-checks, the first survey example, then render
  intervene      optional read-source block; not part of smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FrameWeave V6 real-frame visualizations")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(command: argparse.ArgumentParser, *, output_required: bool = True) -> None:
        command.add_argument("--root", default="/content")
        command.add_argument("--output-dir", required=output_required)
        command.add_argument("--model-dir", default="")
        command.add_argument("--adapter-checkpoint", default="")
        command.add_argument("--herbench-root", default="")
        command.add_argument("--videos-root", default="")
        command.add_argument("--herbench-revision", default="30b42e97180f2c7c0fdc3f2aed412ae04f0366ca")
        command.add_argument("--model-revision", default="5a1f9b995c98ae5b85a28ec89d88d74295be0ea7")
        command.add_argument("--herbench-base-vlm", default="")
        command.add_argument("--herbench-base-sha256", default="")
        command.add_argument("--example-ids", default="")
        command.add_argument("--selection-manifest", default="")
        command.add_argument("--predictions-csv", default="")
        command.add_argument("--query-method", default="top_eta", choices=["top_eta", "explicit", "roi"])
        command.add_argument("--max-queries", type=int, default=8)
        command.add_argument("--dump-full-tokens", action="store_true")
        command.add_argument("--num-frames", type=int, default=0)
        command.add_argument("--min-pixels", type=int, default=0)
        command.add_argument("--max-pixels", type=int, default=0)
        command.add_argument("--limit", type=int, default=0)
        command.add_argument(
            "--herbench-config",
            default="lite_v2",
            help="Retained for the notebook. This pipeline reads lite_v2.",
        )

    for name in ("audit", "cpu-checks", "survey", "collect", "render", "contact-sheet", "smoke", "intervene"):
        add_common(sub.add_parser(name))
    return parser


def _use_survey_manifest(args) -> None:
    if args.example_ids or args.selection_manifest:
        return
    candidate = Path(args.output_dir) / "survey_manifest.json"
    if candidate.is_file():
        args.selection_manifest = str(candidate)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.herbench_config != "lite_v2":
        raise SystemExit("This visualization pipeline uses HERBench lite_v2.")
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from reljoin.frame_viz.cpu_checks import run_cpu_checks
    from reljoin.frame_viz.pipeline import (
        audit_command,
        collect_examples,
        intervene_examples,
        render_output,
        write_examples_index,
    )
    from reljoin.frame_viz.render import contact_sheet
    from reljoin.frame_viz.survey import survey_command, write_paper_groups

    output_dir = Path(args.output_dir)
    if args.command == "audit":
        audit_command(args)
        return 0
    if args.command == "cpu-checks":
        run_cpu_checks(output_dir)
        print("CPU CHECKS OK. GPU smoke was not run.", flush=True)
        return 0
    if args.command == "survey":
        survey_command(args)
        return 0
    if args.command in {"collect", "smoke"}:
        _use_survey_manifest(args)
    if args.command == "collect":
        if not args.limit and not args.example_ids and not args.selection_manifest:
            args.limit = 24
        collect_examples(args)
        return 0
    if args.command == "render":
        render_output(output_dir)
        return 0
    if args.command == "contact-sheet":
        write_examples_index(output_dir)
        write_paper_groups(output_dir)
        contact_sheet(output_dir)
        return 0
    if args.command == "smoke":
        run_cpu_checks(output_dir)
        args.limit = 1
        collect_examples(args)
        render_output(output_dir)
        print("SMOKE OK", flush=True)
        return 0
    if args.command == "intervene":
        intervene_examples(args)
        print("INTERVENE OK", flush=True)
        return 0
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
