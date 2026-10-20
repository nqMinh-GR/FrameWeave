"""Source audit for the FrameWeave V6 visualization pipeline."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    text = out.strip()
    return text or None


def write_audit(output_dir: Path, *, repo_root: Path, checkpoint: Path | None, dataset: dict, blockers: list[str]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    revision = source_revision(repo_root)
    ckpt_line = "checkpoint missing"
    if checkpoint is not None and checkpoint.is_file():
        ckpt_line = f"`{checkpoint}` sha256=`{sha256_file(checkpoint)}`"
    blocker_text = "\n".join(f"- {item}" for item in blockers) or "- none"
    revision_text = revision if revision else "unknown (git rev-parse failed; not invented)"
    text = f"""# FrameWeave V6 visualization audit

Source revision: `{revision_text}`

Checkpoint: {ckpt_line}

Dataset: config `{dataset.get("herbench_config", "lite_v2")}`, revision `{dataset.get("herbench_revision", "")}`.

## Where the tensors are computed

- `alpha`: `FramePooler.forward` in `reljoin/reljoin_modules.py`, stored on `RelJoinResidual.last_alpha`. Shape `[B, K, Pmax]`.
- `beta`: `GlobalRelation.forward` masked softmax over the sender axis, stored on `last_beta`. Shape `[B, K, K]`. Row is the receiver frame, column is the frame being read. The diagonal is masked.
- `pi`: `LocalRelation.forward` inside the existing source-token chunk loop, softmax over the target token axis. It is not kept on the module. A debug sink copies only the requested receiver rows to CPU. Shape after stacking: `[B, S, K, Pmax]`.
- `gamma`: `LocalRelation.forward` masked softmax of global logits plus the local frame score, stored on `last_gamma` with layout `[B, K, P, K]`. `gamma_selected` indexes that tensor at the query `(frame, token)`.
- Module residual: `HybridResidualHead.forward` after the token mask, neighbor mask, gate, and RMS calibration. Stored on `last_delta_x` in FP32, before the backbone dtype cast.
- Effective residual used for `eta`: in `ovis_hooks._apply_reljoin`, `delivered.to(embeds.dtype) - original_tokens`. `eta = RMS(effective) / (RMS(x) + eps)`. The dtype gap is `max |effective - module delta|`.

Name mapping is only in this audit and in each example's `metadata.json`. The module math is unchanged when no `FrameVizSession` is open.

## Masks

`pair_mask` is `M_ij = f_i f_j 1[i != j]`. Rows with no neighbor stay zero and are not renormalized. Read-source intervention, which is off unless the `intervene` command is used, clears other receivers' reads of a sender and then reruns this forward.

## Geometry

Token supports come from `grid_thws` plus the visual tokenizer's own `patch_size` and `smart_resize`. If that confirmation fails, `mapping_status` is `spatial_mapping_unverified` and image overlays are not written as verified figures.

## Blockers

{blocker_text}
"""
    path = output_dir / "implementation_audit.md"
    path.write_text(text, encoding="utf-8")
    return path
