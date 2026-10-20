#!/usr/bin/env python3
"""CLI entry for FrameWeave Hybrid v6.0 (V5.1) on Video-LLaVA-7B (run inside dtr_env)."""
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
        "backbone": "Video-LLaVA-7B",
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = "missing"
    print("Environment:", info)
    if cfg.needs_model() and not torch.cuda.is_available():
        raise RuntimeError("CUDA required — refusing CPU fallback")
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


def herbench_cli_argv(cfg, ckpt_dir: Path, persist: Path, site: Path) -> list[str]:
    from reljoin.config import (
        DTR_COMMIT,
        DTR_SOURCE_SHA256,
        HERBENCH_ACC_SHA256,
        HERBENCH_BASE_SHA256,
        HERBENCH_REVISION,
        MODEL_REVISION,
    )

    return [
        "--root",
        str(cfg.root),
        "--venv-site",
        str(site),
        "--dtr-repo",
        str(cfg.dtr_repo),
        "--dtr-commit",
        DTR_COMMIT,
        "--dtr-sha256",
        DTR_SOURCE_SHA256,
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
            "diagnose_decoder",
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
        default="",
        help="eval_herbench: default vanilla,dtr,hybrid,dtr_hybrid. diagnose_decoder: default vanilla,dtr,hybrid",
    )
    p.add_argument("--smoke", action="store_true", help="eval_herbench / diagnose_decoder smoke")
    p.add_argument("--smoke-n", type=int, default=8, help="diagnose_decoder smoke question count")
    p.add_argument(
        "--max-samples",
        type=int,
        default=128,
        help="diagnose_decoder stratified subset (0 = full lite_v2)",
    )
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

    if cfg.run_mode in {"eval_herbench", "diagnose_decoder"}:
        ckpt_dir = resolve_ckpt_dir(cfg)
        site = next((Path(sys.prefix) / "lib").glob("python3.*/site-packages"), Path(sys.prefix))
        if cfg.run_mode == "eval_herbench":
            from run_herbench_oracle_reljoin import build_parser, run as run_herbench

            persist = cfg.persist_dir if cfg.persist_dir.exists() else cfg.out_dir / "predictions"
            persist.mkdir(parents=True, exist_ok=True)
            argv = herbench_cli_argv(cfg, ckpt_dir, persist, site) + [
                "--out-dir",
                str(cfg.out_dir / "predictions"),
                "--methods",
                args.methods or "vanilla,dtr,hybrid,dtr_hybrid",
            ]
            if args.smoke:
                argv.append("--smoke")
            run_herbench(build_parser().parse_args(argv))
            return

        from run_herbench_decoder_diagnostics import (
            build_parser as build_diag_parser,
        )
        from run_herbench_decoder_diagnostics import run as run_decoder_diag

        persist = (
            cfg.persist_dir / "decoder_diagnostics"
            if cfg.persist_dir.exists()
            else cfg.out_dir / "decoder_diagnostics"
        )
        persist.mkdir(parents=True, exist_ok=True)
        argv = herbench_cli_argv(cfg, ckpt_dir, persist, site) + [
            "--out-dir",
            str(cfg.out_dir / "decoder_diagnostics"),
            "--methods",
            args.methods or "vanilla,dtr,hybrid",
            "--seed",
            "20260819",
        ]
        if args.smoke:
            argv += ["--smoke", "--smoke-n", str(args.smoke_n)]
        elif args.max_samples and args.max_samples > 0:
            argv += ["--max-samples", str(args.max_samples)]
        run_decoder_diag(build_diag_parser().parse_args(argv))
        return

    raise ValueError(cfg.run_mode)


if __name__ == "__main__":
    main()
