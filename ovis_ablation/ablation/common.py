"""Paths, JSON helpers, and the V6 import root. Does not modify v6_ovis."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def ablation_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    return ablation_root().parent


def v6_root() -> Path:
    """Directory that contains the ``reljoin`` package.

    Colab uploads ``reljoin/`` straight into ``/content``, beside ``run_ablation.py``.
    A local checkout keeps the canonical package in the sibling ``v6_ovis`` tree.
    """
    flat = ablation_root() / "reljoin" / "reljoin_modules.py"
    if flat.is_file():
        return ablation_root()
    sibling = workspace_root() / "v6_ovis"
    if (sibling / "reljoin" / "reljoin_modules.py").is_file():
        return sibling
    raise FileNotFoundError(
        f"reljoin package not found at {ablation_root() / 'reljoin'} or {sibling / 'reljoin'}"
    )


def ensure_v6_on_path() -> Path:
    root = v6_root()
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return root


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(blob)


def git_revision(start: Path | None = None) -> str:
    cursor = Path(start or ablation_root()).resolve()
    for candidate in [cursor, *cursor.parents]:
        if (candidate / ".git").exists():
            try:
                proc = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=candidate,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except OSError:
                return "unknown"
            if proc.returncode == 0:
                return proc.stdout.strip() or "unknown"
            return "unknown"
    return "unknown"


def record_blocker(output_dir: Path, stage: str, reasons: list[str]) -> Path:
    path = Path(output_dir) / "shared" / "blockers.json"
    records = []
    if path.is_file():
        existing = read_json(path)
        records = list(existing.get("records", []))
    records.append({"stage": stage, "reasons": list(reasons)})
    write_json(path, {"records": records})
    return path
