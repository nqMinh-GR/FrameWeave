"""Load the ablation YAML and expose pinned protocol constants."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import ablation_root

SUITE_VARIANTS = (
    "vanilla",
    "token_mlp",
    "global_only",
    "local_only",
    "hybrid",
    "hybrid_no_time",
)
# Trained and evaluated by name only. "--variant all" stays the six-variant suite.
EXTRA_VARIANTS = ("frame_only",)
VARIANTS = SUITE_VARIANTS + EXTRA_VARIANTS
TRAINABLE_VARIANTS = tuple(name for name in VARIANTS if name != "vanilla")

# Dataset task strings from the canonical Ovis lite_v2 summary.
TSO_TASK = "Temporal Shot Ordering"
TASKS = (
    "Temporal Shot Ordering",
    "Multi-Person Duration Reasoning",
    "Action Sequence Integrity Identification",
    "Appearance-Grounded Behavior & Interactions",
    "Appearance-Grounded Attribute Recognition",
    "Appearance-Grounded Localization & Trajectory",
    "False Action Memory",
    "Scene Verification & Arrangement",
    "False Object Memory",
    "Multi-Entities Grounding and Localization",
    "Action Counting",
    "Region Localized People Counting",
)
TASK_FAMILIES = {
    "TR&C": (
        "Temporal Shot Ordering",
        "Multi-Person Duration Reasoning",
        "Action Sequence Integrity Identification",
    ),
    "R&T": (
        "Appearance-Grounded Behavior & Interactions",
        "Appearance-Grounded Attribute Recognition",
        "Appearance-Grounded Localization & Trajectory",
    ),
    "GC&V": (
        "False Action Memory",
        "Scene Verification & Arrangement",
        "False Object Memory",
    ),
    "ME&N": (
        "Multi-Entities Grounding and Localization",
        "Action Counting",
        "Region Localized People Counting",
    ),
}

# What each ablation turns on. Hybrid is the untouched V6 module.
VARIANT_MAPPING = {
    "vanilla": {
        "adapter": False,
        "question_encoder": False,
        "frame_pooling": False,
        "global_relation": False,
        "local_matching": False,
        "explicit_time": False,
        "cross_frame_visual": False,
        "notes": "Frozen backbone, hook bypass. No adapter parameters.",
    },
    "token_mlp": {
        "adapter": True,
        "question_encoder": True,
        "frame_pooling": False,
        "global_relation": False,
        "local_matching": False,
        "explicit_time": "own-frame tau/rho only",
        "cross_frame_visual": False,
        "notes": (
            "Question-conditioned bottleneck MLP. After q, masks, and time metadata "
            "are fixed, output (i, p) depends only on x_ip. chi_i is the structural "
            "neighbor mask, so a single valid frame stays identity."
        ),
    },
    "global_only": {
        "adapter": True,
        "question_encoder": True,
        "frame_pooling": True,
        "global_relation": True,
        "local_matching": False,
        "explicit_time": True,
        "cross_frame_visual": "frame summaries",
        "notes": "No LocalRelation and no local residual parameters.",
    },
    "local_only": {
        "adapter": True,
        "question_encoder": True,
        "frame_pooling": "within-frame only",
        "global_relation": False,
        "local_matching": True,
        "explicit_time": "source-frame sbar_i only",
        "cross_frame_visual": "token correspondence, not target summaries",
        "notes": (
            "Keeps sbar_i of the source frame for the local query and residual head. "
            "Drops GlobalRelation, rG, eG, A_s(sbar_j-sbar_i), A_r, local_global, and head_global."
        ),
    },
    "hybrid": {
        "adapter": True,
        "question_encoder": True,
        "frame_pooling": True,
        "global_relation": True,
        "local_matching": True,
        "explicit_time": True,
        "cross_frame_visual": True,
        "notes": "Canonical v6 RelJoinResidual(variant='Hybrid'). Not reimplemented.",
    },
    "hybrid_no_time": {
        "adapter": True,
        "question_encoder": True,
        "frame_pooling": True,
        "global_relation": "content only",
        "local_matching": True,
        "explicit_time": False,
        "cross_frame_visual": True,
        "notes": (
            "Full global and local interaction with sbar_i = s_i. "
            "W_abs, phi_ij, A_K, A_V, and a_b are not registered. "
            "This removes explicit time/order features inside FrameWeave, not Ovis positional encodings."
        ),
    },
    "frame_only": {
        "adapter": True,
        "question_encoder": True,
        "frame_pooling": "within-frame only",
        "global_relation": False,
        "local_matching": False,
        "explicit_time": "own-frame tau/rho only",
        "cross_frame_visual": False,
        "notes": (
            "Eq. (1) pooling with sbar_i = s_i + f_i W_abs[tau_i; rho_i]. The residual head reads "
            "k_ip, sbar_i, and q only. No GlobalRelation, LocalRelation, phi_ij, head_global, "
            "or local_adapter. chi_i is the structural neighbor mask, so the residual is live "
            "whenever at least two frames are valid."
        ),
    },
}

V6_NOT_USED = {
    "HybridLocalOff": "Still constructs LocalRelation and P_L. Not global_only.",
    "GlobalTimeFix": "Alias of HybridLocalOff. Not global_only.",
    "HybridNoGlobal": "Local branch still receives rG, eG, and A_s(sbar_j-sbar_i). Not local_only.",
    "HybridNoTime": "Zeros tau/rho/phi but still owns W_abs, A_K, A_V, and a_b. Not hybrid_no_time.",
}


def default_config_path() -> Path:
    return ablation_root() / "configs" / "ovis_ablations.yaml"


def _parse_scalar(text: str) -> Any:
    value = text.strip()
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(text: str) -> dict:
    """Parse the project config. Nested mappings are one level deep."""
    root: dict[str, Any] = {}
    block_key: str | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if line.startswith("- "):
            if block_key is None:
                raise ValueError(f"List item without a key: {raw}")
            items = root.get(block_key)
            if not isinstance(items, list):
                items = []
                root[block_key] = items
            items.append(_parse_scalar(line[2:]))
            continue
        if ":" not in line:
            raise ValueError(f"Cannot parse YAML line: {raw}")
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()
        if indent == 0:
            block_key = key
            root[key] = {} if rest == "" else _parse_scalar(rest)
            continue
        parent = root.get(block_key)
        if not isinstance(parent, dict):
            parent = {}
            root[block_key] = parent
        if rest == "":
            raise ValueError(f"Only one nesting level is supported: {raw}")
        parent[key] = _parse_scalar(rest)
    return root


def load_yaml(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        return _load_simple_yaml(text)
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return loaded


@dataclass
class AblationConfig:
    seed: int = 42
    variants: list[str] = field(default_factory=lambda: list(SUITE_VARIANTS))
    model_id: str = "ATH-MaaS/Ovis2.5-9B"
    model_revision: str = "5a1f9b995c98ae5b85a28ec89d88d74295be0ea7"
    hidden_size: int = 4096
    rel_dim: int = 256
    herbench_id: str = "DanBenAmi/HERBench"
    herbench_config: str = "lite_v2"
    herbench_revision: str = "30b42e97180f2c7c0fdc3f2aed412ae04f0366ca"
    herbench_expected_questions: int = 1971
    herbench_expected_videos: int = 68
    herbench_base_sha256: str = ""
    herbench_acc_sha256: str = ""
    train_dataset_id: str = "lmms-lab/LLaVA-Video-178K"
    train_dataset_revision: str = "6d8c562dc26d70042a0d9704d1cae58c94b89098"
    train_annotation: str = "0_30_s_perceptiontest/0_30_s_perceptiontest_mc_qa_processed.json"
    expected_train_questions: int = 454
    expected_train_videos: int = 128
    expected_val_questions: int = 67
    expected_val_videos: int = 19
    min_pixels: int = 200704
    max_pixels: int = 200704
    enable_thinking: bool = False
    use_sdpa: bool = True
    use_gradient_checkpointing: bool = True
    dtype: str = "bfloat16"
    adapter_dtype: str = "float32"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_epochs: int = 8
    early_stop_patience: int = 3
    k_train: list[int] = field(default_factory=lambda: [3, 4, 5, 6, 7, 8])
    herbench_frames: int = 8
    time_eps: float = 1e-8
    rms_eps: float = 1e-8
    token_mlp_hidden: int | None = None
    val_policy: str = "canonical_v6_val_jobs"
    paths: dict[str, str] = field(default_factory=dict)
    output_root: str = "outputs/frameweave_ovis_ablation"
    config_path: str = ""

    def output_dir(self) -> Path:
        raw = Path(self.output_root)
        if raw.is_absolute():
            return raw
        return (ablation_root() / raw).resolve()

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "variants": list(self.variants),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "hidden_size": self.hidden_size,
            "rel_dim": self.rel_dim,
            "herbench_id": self.herbench_id,
            "herbench_config": self.herbench_config,
            "herbench_revision": self.herbench_revision,
            "herbench_expected_questions": self.herbench_expected_questions,
            "herbench_expected_videos": self.herbench_expected_videos,
            "train_dataset_id": self.train_dataset_id,
            "train_dataset_revision": self.train_dataset_revision,
            "train_annotation": self.train_annotation,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "enable_thinking": self.enable_thinking,
            "use_sdpa": self.use_sdpa,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "dtype": self.dtype,
            "adapter_dtype": self.adapter_dtype,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_epochs": self.max_epochs,
            "early_stop_patience": self.early_stop_patience,
            "k_train": list(self.k_train),
            "herbench_frames": self.herbench_frames,
            "time_eps": self.time_eps,
            "rms_eps": self.rms_eps,
            "token_mlp_hidden": self.token_mlp_hidden,
            "val_policy": self.val_policy,
            "paths": dict(self.paths),
            "output_root": str(self.output_dir()),
            "config_path": self.config_path,
        }


def load_config(path: Path | None = None) -> AblationConfig:
    path = Path(path) if path else default_config_path()
    raw = load_yaml(path)
    known = set(AblationConfig.__dataclass_fields__)
    payload = {key: value for key, value in raw.items() if key in known}
    cfg = AblationConfig(**payload)
    cfg.config_path = str(path)
    unknown = [name for name in cfg.variants if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}. Expected {VARIANTS}")
    if int(cfg.micro_batch_size) != 1:
        raise ValueError("micro_batch_size must stay 1; there is no multi-sample collate.")
    if cfg.val_policy != "canonical_v6_val_jobs":
        raise ValueError(
            "This suite materializes the canonical V6 validation policy. "
            f"Got {cfg.val_policy!r}."
        )
    return cfg
