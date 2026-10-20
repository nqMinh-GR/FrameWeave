#!/usr/bin/env python3
"""FrameWeave Ovis2.5-9B ablation runner.

Examples:
  python run_ablation.py prepare
  python run_ablation.py unit
  python run_ablation.py smoke --variant token_mlp
  python run_ablation.py train --variant all
  python run_ablation.py eval --variant all
  python run_ablation.py aggregate
  python run_ablation.py train --variant frame_only
  python run_ablation.py eval --variant frame_only
  python run_ablation.py paired --variant frame_only --reference hybrid
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation.aggregate import (  # noqa: E402
    PAIRED_RESAMPLES,
    paired_comparison,
    summarize_predictions,
    write_paired,
    write_tables,
)
from ablation.common import read_json, read_jsonl, write_json  # noqa: E402
from ablation.config import (  # noqa: E402
    TRAINABLE_VARIANTS,
    VARIANTS,
    VARIANT_MAPPING,
    V6_NOT_USED,
    load_config,
)
from ablation.eval_herbench import eval_prediction_dir, evaluate  # noqa: E402
from ablation.invariants import run_invariants  # noqa: E402
from ablation.modules import build_adapter, parameter_manifest, select_token_mlp_hidden  # noqa: E402
from ablation.paths import ensure_training_split, resolve_paths, write_resolved  # noqa: E402
from ablation.schedule import build_schedules, write_schedules  # noqa: E402
from ablation.train import environment_snapshot, smoke_variants, train_variants, variant_dir  # noqa: E402


def _variants(text: str, cfg) -> list[str]:
    if text.strip().lower() == "all":
        return list(cfg.variants)
    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variant {unknown}. Choose from {VARIANTS}.")
    return names


def prepare(cfg) -> None:
    resolved = ensure_training_split(cfg)
    write_resolved(cfg, resolved)
    write_json(cfg.output_dir() / "shared" / "variant_mapping.json", {"variants": VARIANT_MAPPING, "not_used": V6_NOT_USED})
    write_json(cfg.output_dir() / "shared" / "environment.json", environment_snapshot(cfg))
    budget = select_token_mlp_hidden(cfg.hidden_size, cfg.rel_dim, cfg.rms_eps, cfg.time_eps)
    manifests = {"token_mlp_match": budget}
    for name in TRAINABLE_VARIANTS:
        hidden = budget["hidden"] if name == "token_mlp" else None
        module = build_adapter(
            name,
            cfg.hidden_size,
            cfg.rel_dim,
            hidden=hidden,
            rms_eps=cfg.rms_eps,
            time_eps=cfg.time_eps,
        )
        info = parameter_manifest(module, name)
        info["relative_to_hybrid"] = (info["active"] - budget["hybrid_params"]) / budget["hybrid_params"]
        manifests[name] = info
        del module
    manifests["vanilla"] = parameter_manifest(None, "vanilla")
    write_json(cfg.output_dir() / "shared" / "parameter_budget.json", manifests)
    if resolved.train_manifest and resolved.val_manifest:
        bundle = build_schedules(cfg, resolved, probe_frames=True)
        write_schedules(cfg, bundle)
        print(
            f"Schedule {bundle['meta']['schedule_sha256'][:12]} "
            f"train_rows={bundle['meta']['train_rows']} val_rows={bundle['meta']['val_rows']} "
            f"frames_ready={bundle['meta']['frames']['rows_ready']}/{bundle['meta']['frames']['rows_total']}"
        )
    else:
        print(
            "PREPARE: PerceptionTest is not on disk yet, so the V6 split was not written. "
            "smoke/train download it first; prepare then builds 454/128 and 67/19."
        )
    print(f"token_mlp H={budget['hidden']} rel_diff={budget['rel_diff']:.6f}")
    interpreter = Path(sys.executable)
    commands = [
        f"{interpreter} run_ablation.py prepare",
        f"{interpreter} run_ablation.py unit",
        f"{interpreter} run_ablation.py smoke",
        f"{interpreter} run_ablation.py train --variant all",
        f"{interpreter} run_ablation.py eval --variant all",
        f"{interpreter} run_ablation.py aggregate",
    ]
    write_json(
        cfg.output_dir() / "shared" / "commands.json",
        {
            "cwd": str(ROOT),
            "interpreter": str(interpreter),
            "stages": commands,
            "note": "Smoke, train, and eval load Ovis only when CUDA, flash-attn, and local videos exist. They do not download weights or datasets and do not change resolution.",
        },
    )
    print(f"Output: {cfg.output_dir()}")


def _variant_summary(cfg, rows: list[dict], name: str) -> dict:
    selected_path = variant_dir(cfg, name) / "selected.json"
    budget_path = variant_dir(cfg, name) / "parameter_manifest.json"
    selected = read_json(selected_path) if selected_path.is_file() else {}
    budget = read_json(budget_path) if budget_path.is_file() else {}
    if not budget:
        budget_file = cfg.output_dir() / "shared" / "parameter_budget.json"
        shared_budget = read_json(budget_file) if budget_file.is_file() else {}
        budget = shared_budget.get(name, {}) if isinstance(shared_budget, dict) else {}
    if name == "vanilla":
        trainable = 0
        epoch = None
    else:
        trainable = budget.get("trainable")
        epoch = selected.get("epoch")
    checkpoint_ids = sorted({row.get("checkpoint_id", "") for row in rows if row.get("variant") == name})
    return summarize_predictions(
        rows,
        variant=name,
        seed=cfg.seed,
        expected_questions=cfg.herbench_expected_questions,
        checkpoint_id=checkpoint_ids[-1] if checkpoint_ids else "",
        selected_epoch=epoch,
        trainable_params=trainable,
    )


def aggregate(cfg) -> None:
    prediction_files = sorted((cfg.output_dir() / "eval").glob("seed_*/predictions.jsonl"))
    rows = []
    for path in prediction_files:
        rows.extend(read_jsonl(path))
    summaries = [_variant_summary(cfg, rows, name) for name in cfg.variants]
    written = write_tables(summaries, cfg.output_dir() / "aggregate")
    print(json_ready(written))
    incomplete = [item["variant"] for item in summaries if not item["complete"]]
    if incomplete:
        print("Aggregate is not a full lite_v2 result for: " + ", ".join(incomplete))


def paired(cfg, names: list[str], references: list[str], n_resamples: int) -> None:
    pred_dir, _smoke = eval_prediction_dir(cfg, None)
    path = pred_dir / "predictions.jsonl"
    if not path.is_file():
        raise SystemExit(f"No full predictions at {path}. Run eval first.")
    rows = read_jsonl(path)
    for name in names:
        compared = [ref for ref in references if ref != name]
        if not compared:
            continue
        out_dir = cfg.output_dir() / f"aggregate_{name}"
        summaries = [_variant_summary(cfg, rows, item) for item in [name, *compared]]
        written = write_tables(summaries, out_dir)
        reports = [
            paired_comparison(rows, variant=name, reference=ref, n_resamples=n_resamples)
            for ref in compared
        ]
        written.update(write_paired(reports, out_dir))
        print((out_dir / "results.md").read_text(encoding="utf-8"))
        print((out_dir / "paired.md").read_text(encoding="utf-8"))
        print(json_ready(written))
        incomplete = [item["variant"] for item in summaries if not item["complete"]]
        if incomplete:
            print("Not a full lite_v2 result for: " + ", ".join(incomplete))


def json_ready(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FrameWeave Ovis2.5-9B ablations",
        epilog=(
            "Stages: prepare, unit, smoke, train, eval, aggregate, paired. "
            "Example: python run_ablation.py unit && python run_ablation.py prepare "
            "&& python run_ablation.py smoke && python run_ablation.py train "
            "&& python run_ablation.py eval && python run_ablation.py aggregate. "
            "Smoke and full train/eval load Ovis only when CUDA, flash-attn, and local videos exist. "
            "They do not download weights or change resolution. "
            "eval --limit N writes eval_smoke and does not touch eval/seed_*/predictions.jsonl. "
            "paired --variant X --reference hybrid writes aggregate_X/ from eval/seed_<seed>/predictions.jsonl."
        ),
    )
    parser.add_argument("stage", choices=["prepare", "unit", "smoke", "train", "eval", "aggregate", "paired"])
    parser.add_argument("--config", default="")
    parser.add_argument("--variant", default="all")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Eval only: first N questions, written to eval_smoke.")
    parser.add_argument("--reference", default="hybrid", help="Paired only: comma-separated reference variants.")
    parser.add_argument("--bootstrap", type=int, default=PAIRED_RESAMPLES, help="Paired only: video-cluster draws.")
    args = parser.parse_args()
    if args.limit and args.stage != "eval":
        raise SystemExit("--limit is only valid for eval")
    cfg = load_config(Path(args.config) if args.config else None)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    names = _variants(args.variant, cfg)
    if args.stage == "prepare":
        prepare(cfg)
        return
    if args.stage == "unit":
        report = run_invariants(cfg)
        write_json(cfg.output_dir() / "unit" / "invariant_report.json", report)
        print("Unit invariants passed. Smoke accuracy is not a paper result.")
        return
    if args.stage == "paired":
        if int(args.bootstrap) < 1:
            raise SystemExit("--bootstrap must be >= 1")
        paired(cfg, names, _variants(args.reference, cfg), int(args.bootstrap))
        return
    if args.stage in {"smoke", "train"}:
        resolved = ensure_training_split(cfg)
    else:
        resolved = resolve_paths(cfg)
    write_resolved(cfg, resolved)
    if args.stage == "smoke":
        smoke_variants(cfg, resolved, names)
        return
    if args.stage == "train":
        train_variants(cfg, resolved, names, resume=args.resume)
        return
    if args.stage == "eval":
        evaluate(cfg, resolved, names, limit=args.limit or None)
        return
    aggregate(cfg)


if __name__ == "__main__":
    main()
