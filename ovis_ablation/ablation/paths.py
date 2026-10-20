"""Resolve data and checkpoint paths that actually exist. Never invent one."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .common import ablation_root, v6_root, workspace_root, write_json
from .config import AblationConfig


def _candidates(*values: str | Path | None) -> list[Path]:
    found: list[Path] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        found.append(Path(text).expanduser())
    return found


def _first_file(candidates: list[Path]) -> tuple[str | None, list[str]]:
    searched = [str(path) for path in candidates]
    for path in candidates:
        if path.is_file():
            return str(path.resolve()), searched
    return None, searched


def _first_dir(candidates: list[Path], marker: str | None = None) -> tuple[str | None, list[str]]:
    searched = [str(path) for path in candidates]
    for path in candidates:
        if not path.is_dir():
            continue
        if marker is None or (path / marker).is_file():
            return str(path.resolve()), searched
    return None, searched


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


@dataclass
class ResolvedPaths:
    train_manifest: str | None = None
    val_manifest: str | None = None
    model_dir: str | None = None
    videos_dir: str | None = None
    herbench_root: str | None = None
    herbench_videos: str | None = None
    herbench_base_vlm: str | None = None
    herbench_accuracy: str | None = None
    herbench_parquet: str | None = None
    searched: dict[str, list[str]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "train_manifest": self.train_manifest,
            "val_manifest": self.val_manifest,
            "model_dir": self.model_dir,
            "videos_dir": self.videos_dir,
            "herbench_root": self.herbench_root,
            "herbench_videos": self.herbench_videos,
            "herbench_base_vlm": self.herbench_base_vlm,
            "herbench_accuracy": self.herbench_accuracy,
            "herbench_parquet": self.herbench_parquet,
            "searched": self.searched,
            "missing": self.missing,
        }


def _manifest_candidates(cfg: AblationConfig, name: str) -> list[Path]:
    pinned = cfg.paths or {}
    v6 = v6_root()
    return _candidates(
        pinned.get(name),
        _env("OVIS_TRAIN_MANIFEST" if name == "train_manifest" else "OVIS_VAL_MANIFEST"),
        v6 / "outputs_reljoin_v6_ovis" / "data" / f"{name}.json",
        cfg.output_dir() / "data" / f"{name}.json",
    )


def resolve_paths(cfg: AblationConfig) -> ResolvedPaths:
    pinned = cfg.paths or {}
    v6 = v6_root()
    reljoin_root = _env("RELJOIN_ROOT", "FRAMEWEAVE_ROOT", "OVIS_ABLATION_ROOT")
    resolved = ResolvedPaths()

    manifest_train, searched = _first_file(_manifest_candidates(cfg, "train_manifest"))
    resolved.train_manifest = manifest_train
    resolved.searched["train_manifest"] = searched
    if manifest_train is None:
        resolved.missing.append("train_manifest")

    manifest_val, searched = _first_file(_manifest_candidates(cfg, "val_manifest"))
    resolved.val_manifest = manifest_val
    resolved.searched["val_manifest"] = searched
    if manifest_val is None:
        resolved.missing.append("val_manifest")

    model_dir, searched = _first_dir(
        _candidates(
            pinned.get("model_dir"),
            _env("OVIS_MODEL_DIR"),
            Path(reljoin_root) / "checkpoints" / "ovis2_5_9b" if reljoin_root else None,
            Path("/content/checkpoints/ovis2_5_9b"),
            ablation_root() / "checkpoints" / "ovis2_5_9b",
        ),
        marker="config.json",
    )
    resolved.model_dir = model_dir
    resolved.searched["model_dir"] = searched
    if model_dir is None:
        resolved.missing.append("model_dir")

    videos_dir, searched = _first_dir(
        _candidates(
            pinned.get("videos_dir"),
            _env("PERCEPTION_VIDEOS", "OVIS_VIDEOS"),
            Path(reljoin_root) / "perceptiontest_lite" / "videos" if reljoin_root else None,
            Path("/content/perceptiontest_lite/videos"),
            v6 / "perceptiontest_lite" / "videos",
            workspace_root() / "perceptiontest_lite" / "videos",
        )
    )
    resolved.videos_dir = videos_dir
    resolved.searched["videos_dir"] = searched
    if videos_dir is None:
        resolved.missing.append("videos_dir")

    herbench_root, searched = _first_dir(
        _candidates(
            pinned.get("herbench_root"),
            _env("HERBENCH_ROOT"),
            Path(reljoin_root) / "HERBench_official" if reljoin_root else None,
            Path("/content/HERBench_official"),
            v6 / "HERBench_official",
            workspace_root() / "HERBench_official",
        )
    )
    resolved.herbench_root = herbench_root
    resolved.searched["herbench_root"] = searched
    if herbench_root is None:
        resolved.missing.append("herbench_root")

    video_candidates = _candidates(
        pinned.get("herbench_videos"),
        _env("HERBENCH_VIDEOS"),
        Path(herbench_root) / "videos" if herbench_root else None,
        Path(reljoin_root) / "HERBench_official" / "videos" if reljoin_root else None,
    )
    herbench_videos, searched = _first_dir(video_candidates)
    resolved.herbench_videos = herbench_videos
    resolved.searched["herbench_videos"] = searched
    if herbench_videos is None:
        resolved.missing.append("herbench_videos")

    parquet_name = f"herbench_{cfg.herbench_config}.parquet"
    parquet, searched = _first_file(
        _candidates(
            Path(herbench_root) / "data" / parquet_name if herbench_root else None,
            v6 / "HERBench_official" / "data" / parquet_name,
            Path("/content/HERBench_official/data") / parquet_name,
        )
    )
    resolved.herbench_parquet = parquet
    resolved.searched["herbench_parquet"] = searched
    if parquet is None:
        resolved.missing.append("herbench_parquet")

    base_vlm, searched = _first_file(
        _candidates(
            pinned.get("herbench_base_vlm"),
            Path(herbench_root) / "evaluation" / "model_wrappers" / "base_vlm.py" if herbench_root else None,
            v6 / "herbench_eval_pins" / "base_vlm.py",
            Path("/content/herbench_eval_pins/base_vlm.py"),
        )
    )
    resolved.herbench_base_vlm = base_vlm
    resolved.searched["herbench_base_vlm"] = searched
    if base_vlm is None:
        resolved.missing.append("herbench_base_vlm")

    accuracy, searched = _first_file(
        _candidates(
            pinned.get("herbench_accuracy"),
            Path(herbench_root) / "evaluation" / "calculate_accuracy.py" if herbench_root else None,
            v6 / "herbench_eval_pins" / "calculate_accuracy.py",
            Path("/content/herbench_eval_pins/calculate_accuracy.py"),
        )
    )
    resolved.herbench_accuracy = accuracy
    resolved.searched["herbench_accuracy"] = searched
    if accuracy is None:
        resolved.missing.append("herbench_accuracy")
    return resolved


def _perception_roots() -> list[Path]:
    roots = [
        Path("/content/perceptiontest_lite"),
        ablation_root() / "perceptiontest_lite",
        workspace_root() / "perceptiontest_lite",
        v6_root() / "perceptiontest_lite",
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _local_perception() -> tuple[Path, Path] | None:
    """Annotation and videos already on disk. Does not download."""
    filename = Path("0_30_s_perceptiontest/0_30_s_perceptiontest_mc_qa_processed.json").name
    for root in _perception_roots():
        videos = root / "videos"
        cache = root / "hf_cache"
        if not videos.is_dir() or not cache.is_dir():
            continue
        direct = cache / "0_30_s_perceptiontest" / filename
        ann = direct if direct.is_file() else next(cache.rglob(filename), None)
        if ann is not None and ann.is_file():
            return ann, videos
    return None


def ensure_training_split(cfg: AblationConfig) -> ResolvedPaths:
    """Use an existing split, or build V6's lite split once PerceptionTest is local.

    The notebook downloads PerceptionTest. This does not. select_for_mode(..., "lite")
    is the same call V6 training uses, then the 454/128 and 67/19 check rejects any other split.
    """
    resolved = resolve_paths(cfg)
    if resolved.train_manifest and resolved.val_manifest:
        return resolved
    located = _local_perception()
    if located is None:
        return resolved
    ann, videos = located
    from .common import ensure_v6_on_path

    ensure_v6_on_path()
    from reljoin.data_perception import (  # noqa: WPS433
        assert_no_herbench_in_training,
        load_and_filter_samples,
        save_manifests,
        select_for_mode,
    )

    samples, excluded = load_and_filter_samples(ann, videos)
    train_s, val_s = select_for_mode(samples, "lite", seed=int(cfg.seed))
    assert_no_herbench_in_training(list(train_s) + list(val_s))
    save_manifests(train_s, val_s, excluded, cfg.output_dir() / "data")
    print(
        f"Built V6 lite split from {ann.parent.name}: "
        f"train={len(train_s)} val={len(val_s)} → {cfg.output_dir() / 'data'}"
    )
    return resolve_paths(cfg)


def write_resolved(cfg: AblationConfig, resolved: ResolvedPaths) -> Path:
    path = cfg.output_dir() / "shared" / "resolved_paths.json"
    write_json(path, resolved.to_dict())
    return path
