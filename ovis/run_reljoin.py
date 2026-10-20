#!/usr/bin/env python3
"""CLI entry for FrameWeave Hybrid v6.0 (V5.1) on Ovis2.5-9B (run inside ovis_env)."""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def print_environment(cfg) -> None:
    import torch

    from reljoin.config import METHOD_VERSION

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
        "vram_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
            if torch.cuda.is_available()
            else 0
        ),
        "method": "FrameWeave-Hybrid",
        "method_version": METHOD_VERSION,
        "backbone": "Ovis2.5-9B",
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = "missing"
    try:
        import flash_attn

        info["flash_attn"] = getattr(flash_attn, "__version__", "present")
    except ImportError:
        info["flash_attn"] = "missing"
    print("Environment:", info)
    if cfg.needs_model() and not torch.cuda.is_available():
        raise RuntimeError("CUDA required — refusing CPU fallback")
    if cfg.needs_model() and info["flash_attn"] == "missing":
        raise RuntimeError(
            "flash-attn missing. Install flash-attn==2.7.0.post2 --no-build-isolation"
        )
    env_path = cfg.out_dir / "environment.txt"
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(f"{k}={v}" for k, v in info.items()), encoding="utf-8")


def resolve_ckpt_dir(cfg) -> Path:
    from run_herbench_oracle_reljoin import find_extract_root, stage_ckpt_dir

    if cfg.reljoin_ckpt:
        ckpt_path = Path(cfg.reljoin_ckpt)
        return ckpt_path.parent if ckpt_path.suffix == ".pt" else ckpt_path
    extract = find_extract_root(cfg.root)
    return stage_ckpt_dir(extract)


def herbench_cli_argv(cfg, ckpt_dir: Path, persist: Path) -> list[str]:
    from reljoin.config import (
        HERBENCH_ACC_SHA256,
        HERBENCH_BASE_SHA256,
        HERBENCH_REVISION,
        MODEL_REVISION,
    )

    return [
        "--root",
        str(cfg.root),
        "--herbench-root",
        str(cfg.herbench_root),
        "--videos-root",
        str(cfg.herbench_videos),
        "--model-dir",
        str(cfg.model_dir),
        "--persist-dir",
        str(persist),
        "--herbench-revision",
        HERBENCH_REVISION,
        "--model-revision",
        MODEL_REVISION,
        "--herbench-base-vlm",
        str(cfg.eval_pins / "base_vlm.py"),
        "--herbench-accuracy",
        str(cfg.eval_pins / "calculate_accuracy.py"),
        "--herbench-base-sha256",
        HERBENCH_BASE_SHA256,
        "--herbench-acc-sha256",
        HERBENCH_ACC_SHA256,
        "--reljoin-ckpt",
        str(ckpt_dir),
        "--max-pixels",
        str(cfg.max_pixels),
    ]


def main() -> None:
    from reljoin.config import VALID_STAGES, VALID_VARIANTS, ExperimentConfig, make_output_tree

    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-mode",
        default="unit",
        choices=[
            "unit",
            "smoke",
            "overfit",
            "lite",
            "eval_herbench",
            "full",
            "diagnose_data",
        ],
    )
    p.add_argument("--root", default="/content")
    p.add_argument("--variant", default="Hybrid", choices=list(VALID_VARIANTS))
    p.add_argument("--stage", default="joint", choices=list(VALID_STAGES))
    p.add_argument("--init-ckpt", default="")
    p.add_argument(
        "--methods",
        default="vanilla,hybrid",
        help="eval_herbench: vanilla,hybrid,...",
    )
    p.add_argument("--smoke", action="store_true", help="eval_herbench smoke")
    p.add_argument("--tso-only", action="store_true", help="eval_herbench TSO subset")
    args = p.parse_args()
    cfg = ExperimentConfig(
        run_mode="unit" if args.run_mode == "diagnose_data" else args.run_mode,
        root=Path(args.root),
        variant=args.variant,
        stage=args.stage,
        init_ckpt=args.init_ckpt,
    )
    if args.run_mode != "unit":
        make_output_tree(cfg.out_dir)
        cfg.save()

    if args.run_mode == "diagnose_data":
        from reljoin.data_perception import diagnose_perception_data

        diagnose_perception_data(Path(args.root))
        return

    if cfg.run_mode == "unit":
        from reljoin.unit_tests import run_all_unit_tests

        run_all_unit_tests()
        return

    print_environment(cfg)

    if cfg.run_mode in {"smoke", "overfit", "lite", "full"}:
        from reljoin.train import main_train

        main_train(cfg)
        return

    if cfg.run_mode == "eval_herbench":
        ckpt_dir = resolve_ckpt_dir(cfg)
        from run_herbench_oracle_reljoin import build_parser, run as run_herbench

        persist = cfg.persist_dir if cfg.persist_dir.exists() else cfg.out_dir / "predictions"
        persist.mkdir(parents=True, exist_ok=True)
        argv = herbench_cli_argv(cfg, ckpt_dir, persist) + [
            "--out-dir",
            str(cfg.out_dir / "predictions"),
        ]
        if args.methods:
            argv += ["--methods", args.methods]
        if args.smoke:
            argv.append("--smoke")
        if args.tso_only:
            argv.append("--tso-only")
        run_herbench(build_parser().parse_args(argv))
        return

    raise ValueError(cfg.run_mode)


if __name__ == "__main__":
    main()
