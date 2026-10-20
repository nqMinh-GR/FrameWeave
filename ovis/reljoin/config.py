"""Pinned resources and run-mode configuration for FrameWeave Hybrid v6 (V5.1) on Ovis2.5-9B."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

RunMode = Literal["unit", "smoke", "overfit", "lite", "eval_herbench", "full"]
TrainStage = Literal["joint", "A", "B1", "B2"]

# ---------------------------------------------------------------------------
# Pinned backbone / datasets
# ---------------------------------------------------------------------------
MODEL_ID = "ATH-MaaS/Ovis2.5-9B"
# Latest ATH-MaaS card commit known at implementation time. Weights match the
# Ovis2.5-9B family (NaViT + Qwen3-8B). from_pretrained records the resolved SHA.
MODEL_REVISION = "5a1f9b995c98ae5b85a28ec89d88d74295be0ea7"

HERBENCH_ID = "DanBenAmi/HERBench"
HERBENCH_CONFIG = "lite_v2"
HERBENCH_REVISION = "30b42e97180f2c7c0fdc3f2aed412ae04f0366ca"
HERBENCH_BASE_SHA256 = "6104884c366906d5c223cc9727c3bad778dcdf98c881c4b17b556c20e940af97"
HERBENCH_ACC_SHA256 = "eb67b618f3004b52c8290b2fd30e4012153a66f3f93d576c29760280caa3344d"
HERBENCH_BASE_URL = (
    "https://raw.githubusercontent.com/DanBenAmi/HERBench/main/"
    "evaluation/model_wrappers/base_vlm.py"
)
HERBENCH_ACC_URL = (
    "https://raw.githubusercontent.com/DanBenAmi/HERBench/main/"
    "evaluation/calculate_accuracy.py"
)

TRAIN_DATASET_ID = "lmms-lab/LLaVA-Video-178K"
TRAIN_DATASET_REVISION = "6d8c562dc26d70042a0d9704d1cae58c94b89098"
TRAIN_ANNOTATION = "0_30_s_perceptiontest/0_30_s_perceptiontest_mc_qa_processed.json"
TRAIN_VIDEO_ARCHIVE = "0_30_s_perceptiontest/0_30_s_perceptiontest_videos_1.tar.gz"

# FrameWeave Hybrid v6 / formulation V5.1 hyperparameters (not frame-count assumptions)
REL_DIM = 256
RESIDUAL_EPS = 1e-8
RMS_EPS = 1e-8
TIME_EPS = 1e-8
LOCAL_TOKEN_CHUNK = 0  # 0 = full einsum; >0 chunks source-token dim P
IGNORE_INDEX = -100
METHOD_VERSION = "6.0.0"
FORMULATION = "5.1.0"
TSO_TASK = "Temporal Shot Ordering"
EXPECTED_TSO_N = 152
LOCAL_ADAPTER_XAVIER_GAIN = 0.01

# Native-resolution budget. 9B + K=8 at Ovis default 1344x1792 OOMs on 40 GB.
MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
ENABLE_THINKING = False

# Official Ovis2.5-9B environment (model card)
TORCH_PIN = "2.4.0"
TRANSFORMERS_PIN = "4.51.3"
FLASH_ATTN_PIN = "2.7.0.post2"

# Training — micro-batch is 1 until a real collate exists.
SEED = 42
DTYPE = "bfloat16"
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
EFFECTIVE_BATCH_SIZE = MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0
LITE_EPOCHS = 8
EARLY_STOP_PATIENCE = 3
OVERFIT_MAX_EPOCHS = 30
NUM_WORKERS = 2
USE_CACHE = False
USE_SDPA = True
USE_GRADIENT_CHECKPOINTING = True

# Sampling sizes (experimental, not official splits)
SMOKE_TRAIN = 64
SMOKE_VAL = 16
OVERFIT_N = 32
LITE_TRAIN = 448
LITE_VAL = 64

K_TRAIN_CHOICES = (3, 4, 5, 6, 7, 8)
K_VALUES_UNIT = [1, 2, 3, 5, 8, 12]
TOKENS_PER_FRAME_VALUES = [32, 64, 256]
HERBENCH_ORACLE_FRAMES = 8

VALID_VARIANTS = (
    "GlobalTimeFix",
    "HybridLocalOff",
    "Hybrid",
    "HybridNoTime",
    "HybridNoGlobal",
)
VALID_STAGES = ("joint", "A", "B1", "B2")
CANONICAL_STAGES = ("joint", "B1", "B2")
HYBRID_FAMILY = ("Hybrid", "GlobalTimeFix", "HybridLocalOff")


def architecture_variant(name: str) -> str:
    """Map a checkpoint / run stem / alias to a joining-module variant."""
    stem = str(name or "Hybrid").strip()
    aliases = {
        "RelJoin": "Hybrid",
        "reljoin": "Hybrid",
        "hybrid": "Hybrid",
        "Global-TimeFix": "GlobalTimeFix",
        "global_timefix": "GlobalTimeFix",
        "Hybrid-LocalOff": "HybridLocalOff",
        "hybrid_localoff": "HybridLocalOff",
        "Hybrid-NoTime": "HybridNoTime",
        "hybrid_notime": "HybridNoTime",
        "Hybrid-NoGlobal": "HybridNoGlobal",
        "hybrid_noglobal": "HybridNoGlobal",
    }
    return aliases.get(stem, stem)


def normalize_stage(stage: str) -> str:
    """Map notebook alias A → joint. Main method is joint-from-scratch."""
    raw = str(stage or "joint").strip()
    if raw.lower() == "joint" or raw.upper() == "A":
        return "joint"
    st = raw.upper()
    if st in {"B1", "B2"}:
        return st
    raise ValueError(f"Unknown stage {stage!r}. Expected {VALID_STAGES}")


def run_name_for(variant: str, stage: str = "joint") -> str:
    """Checkpoint / log stem used by unit tests and eval loaders."""
    del stage
    arch = architecture_variant(variant)
    if arch in ("GlobalTimeFix", "HybridLocalOff"):
        return "GlobalTimeFix"
    if arch == "Hybrid":
        return "Hybrid"
    return arch


def variant_to_method(variant: str) -> str:
    arch = architecture_variant(variant)
    return {
        "GlobalTimeFix": "global_timefix",
        "HybridLocalOff": "hybrid_localoff",
        "Hybrid": "hybrid",
        "HybridNoTime": "hybrid_notime",
        "HybridNoGlobal": "hybrid_noglobal",
    }.get(arch, arch.lower())


METHOD_TO_VARIANT = {
    "vanilla": "vanilla",
    "reljoin": "Hybrid",
    "hybrid": "Hybrid",
    "global_timefix": "GlobalTimeFix",
    "hybrid_localoff": "HybridLocalOff",
    "hybrid_notime": "HybridNoTime",
    "hybrid_noglobal": "HybridNoGlobal",
}


@dataclass
class ExperimentConfig:
    run_mode: RunMode = "unit"
    root: Path = field(default_factory=lambda: Path(os.environ.get("RELJOIN_ROOT", "/content")))
    seed: int = SEED
    rel_dim: int = REL_DIM
    residual_eps: float = RESIDUAL_EPS
    rms_eps: float = RMS_EPS
    time_eps: float = TIME_EPS
    local_token_chunk: int = LOCAL_TOKEN_CHUNK
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    warmup_ratio: float = WARMUP_RATIO
    max_grad_norm: float = MAX_GRAD_NORM
    micro_batch_size: int = MICRO_BATCH_SIZE
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS
    lite_epochs: int = LITE_EPOCHS
    early_stop_patience: int = EARLY_STOP_PATIENCE
    overfit_max_epochs: int = OVERFIT_MAX_EPOCHS
    dtype: str = DTYPE
    max_pixels: int = MAX_PIXELS
    min_pixels: int = MIN_PIXELS
    enable_thinking: bool = ENABLE_THINKING
    use_4bit: bool = False
    use_sdpa: bool = USE_SDPA
    use_gradient_checkpointing: bool = USE_GRADIENT_CHECKPOINTING
    num_workers: int = NUM_WORKERS
    variant: str = "Hybrid"
    stage: TrainStage = "joint"
    init_ckpt: str = ""
    reljoin_ckpt: str = ""

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.variant = architecture_variant(self.variant)
        self.stage = normalize_stage(self.stage)  # type: ignore[assignment]
        if self.stage not in CANONICAL_STAGES:
            raise ValueError(f"Unknown stage {self.stage!r}. Expected {VALID_STAGES}")
        self.out_dir = self.root / "outputs_reljoin_v6_ovis"
        self.model_dir = self.root / "checkpoints" / "ovis2_5_9b"
        self.herbench_root = self.root / "HERBench_official"
        self.herbench_videos = self.herbench_root / "videos"
        self.eval_pins = self.root / "herbench_eval_pins"
        self.train_data_dir = self.root / "perceptiontest_lite"
        self.persist_dir = Path(
            os.environ.get(
                "RELJOIN_PERSIST",
                "/content/drive/MyDrive/reljoin_v6_ovis_outputs",
            )
        )

    def needs_herbench(self) -> bool:
        return self.run_mode in {"eval_herbench", "full"}

    def needs_perception(self) -> bool:
        return self.run_mode in {"smoke", "overfit", "lite", "full"}

    def needs_model(self) -> bool:
        return self.run_mode != "unit"

    @property
    def effective_batch_size(self) -> int:
        return int(self.micro_batch_size) * int(self.gradient_accumulation_steps)

    @property
    def run_name(self) -> str:
        return run_name_for(self.variant, self.stage)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        d.update(
            {
                "MODEL_ID": MODEL_ID,
                "MODEL_REVISION": MODEL_REVISION,
                "HERBENCH_ID": HERBENCH_ID,
                "HERBENCH_REVISION": HERBENCH_REVISION,
                "TRAIN_DATASET_ID": TRAIN_DATASET_ID,
                "TRAIN_DATASET_REVISION": TRAIN_DATASET_REVISION,
                "effective_batch_size": self.effective_batch_size,
                "run_name": self.run_name,
                "method": "FrameWeave-Hybrid",
                "method_version": METHOD_VERSION,
                "formulation": FORMULATION,
                "backbone": "Ovis2.5-9B",
            }
        )
        return d

    def save(self, path: Path | None = None) -> Path:
        path = path or (self.out_dir / "config.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def make_output_tree(out_dir: Path) -> dict[str, Path]:
    """Create the required outputs_reljoin_v6_ovis/ tree."""
    paths = {
        "root": out_dir,
        "data": out_dir / "data",
        "checkpoints": out_dir / "checkpoints",
        "checkpoints_last": out_dir / "checkpoints" / "last",
        "checkpoints_best": out_dir / "checkpoints" / "best_external_val",
        "logs": out_dir / "logs",
        "predictions": out_dir / "predictions",
        "diagnostics": out_dir / "diagnostics",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
