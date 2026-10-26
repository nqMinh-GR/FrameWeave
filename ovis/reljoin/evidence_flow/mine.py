"""Rank HERBench lite_v2 Temporal Shot Ordering questions from saved logs.

Reads prediction logs and the upstream question metadata. Does not run the model
and does not write into evaluation outputs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from reljoin.herbench_oracle import choose_oracle_times, extract_evidence_spans
from reljoin.tso_abstention import is_abstention_choice

EXPECTED_SHA256 = "3a19caac96f32e7bc82d5c373bf77c6b3884489ee2fb35c6957c5b7d8858da03"
SEED_IDS = (1, 42, 100, 2027)


def default_paths(workspace: Path, paper: Path) -> dict[str, Path]:
    ablation = workspace / (
        "v6_ovis_ablation/frameweave_ovis_multiseed_outputs-20261018T091434Z-1-001/"
        "frameweave_ovis_multiseed_outputs"
    )
    return {
        "csv": workspace
        / "v6_ovis/reljoin_v6_ovis_outputs/eval_herbench_v2lite/predictions_reljoin_01024e98f2b7c407.csv",
        "upstream": paper / "audit/multiseed_upstream_lite_v2.json",
        "seed_jsonl": {
            1: ablation / "seed_1/eval/seed_1/predictions.jsonl",
            100: ablation / "seed_100/eval/seed_100/predictions.jsonl",
            2027: ablation / "seed_2027/eval/seed_2027/predictions.jsonl",
        },
        "checkpoint": workspace
        / "v6_ovis/outputs_reljoin_v6_ovis/checkpoints/best_external_val/Hybrid.pt",
    }


def rank_tso_candidates(workspace: Path, paper: Path) -> dict:
    paths = default_paths(workspace, paper)
    by_id = _paired_seed42(paths["csv"])
    upstream = _upstream_by_id(paths["upstream"])
    seed_hits = {
        seed: _hybrid_correct_ids(path)
        for seed, path in paths["seed_jsonl"].items()
    }
    rows = []
    for question_id, pair in by_id.items():
        rows.append(_score_question(question_id, pair, upstream.get(question_id), seed_hits))
    concrete = [row for row in rows if row["tier"] == "concrete_wrong"]
    abstention = [row for row in rows if row["tier"] == "abstention_flip"]
    concrete.sort(key=_sort_key)
    abstention.sort(key=_sort_key)
    for rank, row in enumerate(concrete, start=1):
        row["rank"] = rank
    for rank, row in enumerate(abstention, start=1):
        row["rank"] = rank
    eligible = [row["question_id"] for row in concrete if row["eligible_for_render"]]
    all_questions = sorted(rows, key=_batch_key)
    return {
        "task": "Temporal Shot Ordering",
        "herbench_config": "lite_v2",
        "n_tso": len(rows),
        "checkpoint_path": str(paths["checkpoint"]),
        "checkpoint_sha256_expected": EXPECTED_SHA256,
        "seed42_predictions": str(paths["csv"]),
        "beta_direction": "beta[i, j] is source j read by receiver i; arrow j -> i",
        "concrete_wrong": concrete,
        "abstention_flips_top10": abstention[:10],
        "n_concrete_wrong": len(concrete),
        "n_abstention_flips": len(abstention),
        "render_queue": eligible,
        "all_questions": all_questions,
        "notes": [
            "Concrete-wrong means Vanilla selected an ordering letter, not the abstention sentence, and Hybrid selected gold.",
            "Abstention flips are audited and are not rendered unless --allow-abstention is set.",
            "Frame times are the published seed-42 CSV timestamps. Sampler agreement is recorded beside them.",
        ],
    }


def label_oracle_times(spans: list[tuple[float, float]], duration: float, num_frames: int = 8):
    """Replay ``choose_oracle_times`` and tag each kept time as a span or a fill."""
    duration = max(float(duration or 1.0), 1e-3)
    labels: dict[float, tuple] = {}
    if not spans:
        times = [round(duration * i / max(num_frames - 1, 1), 4) for i in range(num_frames)]
        return times, [("fill", None) for _ in times], False, "uniform_fallback"

    for span_index, (start, end) in enumerate(spans):
        midpoint = round(max(0.0, min(duration, 0.5 * (start + end))), 4)
        labels.setdefault(midpoint, ("span", span_index))
    candidates = sorted(labels)
    if len(candidates) >= num_frames:
        indexes = np.linspace(0, len(candidates) - 1, num_frames)
        times = [candidates[int(round(float(index)))] for index in indexes]
        return times, [labels[time] for time in times], False, "oracle"

    times = list(candidates)
    padded = False
    span_cursor = 0
    while len(times) < num_frames and spans:
        start, end = spans[span_cursor % len(spans)]
        span_index = span_cursor % len(spans)
        if end - start < 1e-3:
            stamp = max(0.0, min(duration, start + 0.04 * (1 + len(times))))
        else:
            frac = ((len(times) + 1) % 5 + 1) / 6.0
            stamp = max(0.0, min(duration, start + frac * (end - start)))
        stamp = round(float(stamp), 4)
        labels.setdefault(stamp, ("span", span_index))
        times.append(stamp)
        times = sorted(set(times))
        span_cursor += 1
        padded = True
        if span_cursor > num_frames * 20:
            break
    seen = set(times)
    guard = 0
    while len(times) < num_frames and guard < num_frames * 50:
        guard += 1
        base = duration * len(times) / max(num_frames - 1, 1)
        stamp = round(max(0.0, min(duration, base)), 4)
        nudge = 0
        while stamp in seen and nudge < 100000:
            nudge += 1
            step = 0.0001 * nudge
            forward = round(min(duration, base + step), 4)
            backward = round(max(0.0, base - step), 4)
            if forward not in seen:
                stamp = forward
            elif backward not in seen:
                stamp = backward
        if stamp in seen:
            break
        seen.add(stamp)
        labels.setdefault(stamp, ("fill", None))
        times.append(stamp)
        times = sorted(times)
        padded = True
    if len(times) > num_frames:
        indexes = np.linspace(0, len(times) - 1, num_frames)
        times = [times[int(round(float(index)))] for index in indexes]
    times = times[:num_frames]
    return times, [labels[time] for time in times], padded, "oracle"


def rank_seed42_flips(workspace: Path, paper: Path) -> dict:
    """Seed-42 lite questions where Vanilla is wrong and Hybrid is correct.

    Other seeds are not consulted. The published CSV supplies the letters,
    timestamps, and video path used on the example figure.
    """
    paths = default_paths(workspace, paper)
    by_id = _paired_seed42(paths["csv"], tso_only=False)
    rows = []
    for question_id, pair in by_id.items():
        row = _score_flip(question_id, pair)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda row: (
        0 if row["task_type"] == "Temporal Shot Ordering" else 1,
        row["task_type"],
        row["question_id"],
    ))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["task_type"]] = counts.get(row["task_type"], 0) + 1
    return {
        "seed": 42,
        "herbench_config": "lite_v2",
        "n_questions": len(by_id),
        "n_flips": len(rows),
        "n_tso_flips": counts.get("Temporal Shot Ordering", 0),
        "counts_by_task": counts,
        "render_queue": [row["question_id"] for row in rows],
        "all_questions": rows,
        "checkpoint_path": str(paths["checkpoint"]),
        "checkpoint_sha256_expected": EXPECTED_SHA256,
        "seed42_predictions": str(paths["csv"]),
        "notes": [
            "A flip is a published seed-42 lite question where Vanilla is wrong and Hybrid is correct.",
            "Figure letters are those CSV predictions. Seeds 1, 100, and 2027 are not used.",
        ],
    }


def _score_flip(question_id: str, pair: dict) -> dict | None:
    vanilla = pair.get("vanilla") or {}
    hybrid = pair.get("hybrid") or {}
    if not vanilla or not hybrid:
        return None
    if str(vanilla.get("is_correct", "")).lower() == "true":
        return None
    if str(hybrid.get("is_correct", "")).lower() != "true":
        return None
    choices = json.loads(vanilla.get("choices_json") or hybrid.get("choices_json") or "[]")
    gold = str(vanilla.get("ground_truth") or hybrid.get("ground_truth") or "").strip().upper()[:1]
    vanilla_letter = str(vanilla.get("parsed_prediction") or "").strip().upper()[:1]
    hybrid_letter = str(hybrid.get("parsed_prediction") or "").strip().upper()[:1]
    timestamps = [
        float(value)
        for value in json.loads(vanilla.get("frame_timestamps") or hybrid.get("frame_timestamps") or "[]")
    ]
    video_path = vanilla.get("video_path") or hybrid.get("video_path") or ""
    if len(timestamps) < 2 or not video_path:
        return None
    return {
        "question_id": question_id,
        "task_type": vanilla.get("task_type") or "",
        "video_path": video_path,
        "question": vanilla.get("question") or "",
        "choices": choices,
        "gold": gold,
        "gold_text": _choice_text(choices, gold),
        "vanilla_letter": vanilla_letter,
        "vanilla_text": _choice_text(choices, vanilla_letter),
        "hybrid_letter": hybrid_letter,
        "hybrid_text": _choice_text(choices, hybrid_letter),
        "frame_timestamps": timestamps,
        "duration": max(timestamps),
        "eligible_for_render": True,
    }


def _paired_seed42(path: Path, tso_only: bool = True) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Seed-42 prediction CSV is missing: {path}")
    paired: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if tso_only and not _is_tso(row.get("task_type", ""), row.get("question_id", "")):
                continue
            paired.setdefault(row["question_id"], {})[row["method"]] = row
    return paired


def _upstream_by_id(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    found = {}
    for record in records:
        if _is_tso(record.get("task_type", ""), record.get("question_id", "")):
            found[str(record["question_id"])] = record
    return found


def _hybrid_correct_ids(path: Path) -> set[str]:
    correct = set()
    if not path.is_file():
        return correct
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("variant") != "hybrid":
                continue
            if not _is_tso(record.get("task", ""), record.get("question_id", "")):
                continue
            if record.get("correct") is True and record.get("predicted_letter") == record.get("gold_letter"):
                correct.add(str(record["question_id"]))
    return correct


def _score_question(question_id: str, pair: dict, upstream: dict | None, seed_hits: dict) -> dict:
    vanilla = pair.get("vanilla") or {}
    hybrid = pair.get("hybrid") or {}
    choices = json.loads(vanilla.get("choices_json") or hybrid.get("choices_json") or "[]")
    gold = str(vanilla.get("ground_truth") or hybrid.get("ground_truth") or "").strip().upper()
    vanilla_letter = str(vanilla.get("parsed_prediction") or "").strip().upper()
    hybrid_letter = str(hybrid.get("parsed_prediction") or "").strip().upper()
    vanilla_text = _choice_text(choices, vanilla_letter)
    gold_text = _choice_text(choices, gold)
    hybrid_text = _choice_text(choices, hybrid_letter)
    vanilla_abstain = is_abstention_choice(vanilla_text)
    gold_abstain = is_abstention_choice(gold_text)
    hybrid_correct = hybrid_letter == gold and str(hybrid.get("is_correct", "")).lower() == "true"
    vanilla_wrong = vanilla_letter != gold
    if vanilla_wrong and hybrid_correct and not vanilla_abstain and not gold_abstain:
        tier = "concrete_wrong"
    elif vanilla_wrong and hybrid_correct and vanilla_abstain and not gold_abstain:
        tier = "abstention_flip"
    else:
        tier = "other"
    timestamps = json.loads(vanilla.get("frame_timestamps") or hybrid.get("frame_timestamps") or "[]")
    timestamps = [float(value) for value in timestamps]
    metadata = (upstream or {}).get("metadata") or {}
    spans, span_mode = extract_evidence_spans(metadata) if metadata else ([], "missing_metadata")
    duration = float((upstream or {}).get("duration") or metadata.get("duration") or vanilla.get("duration") or 0.0)
    sampler_times, labels, padded, mode = label_oracle_times(spans, duration or 1.0, 8)
    official, official_padded, official_mode = choose_oracle_times(spans, duration or 1.0, num_frames=8)
    sampler_match = _close(sampler_times, [float(value) for value in official])
    csv_match = _close(timestamps, sampler_times) if timestamps else False
    span_ids = sorted({index for kind, index in labels if kind == "span"})
    n_evidence_frames = sum(kind == "span" for kind, _index in labels)
    seeds = {
        "42": bool(hybrid_correct),
        "1": question_id in seed_hits.get(1, set()),
        "100": question_id in seed_hits.get(100, set()),
        "2027": question_id in seed_hits.get(2027, set()),
    }
    n_seeds = sum(bool(value) for value in seeds.values())
    eligible = (
        tier == "concrete_wrong"
        and len(span_ids) >= 2
        and csv_match
        and sampler_match
        and len(timestamps) == 8
    )
    return {
        "question_id": question_id,
        "tier": tier,
        "video_id": vanilla.get("video_id") or (upstream or {}).get("video_id", ""),
        "video_path": vanilla.get("video_path") or (upstream or {}).get("video_path", ""),
        "question": vanilla.get("question") or (upstream or {}).get("question", ""),
        "choices": choices,
        "gold": gold,
        "gold_text": gold_text,
        "vanilla_letter": vanilla_letter,
        "vanilla_text": vanilla_text,
        "hybrid_letter": hybrid_letter,
        "hybrid_text": hybrid_text,
        "vanilla_is_abstention": vanilla_abstain,
        "gold_is_abstention": gold_abstain,
        "frame_timestamps": timestamps,
        "sampler_timestamps": sampler_times,
        "timestamps_match_sampler": bool(csv_match and sampler_match),
        "oracle_padded_csv": str(vanilla.get("oracle_padded", "")).lower() == "true",
        "sampler_padded": bool(padded or official_padded),
        "sampler_mode": official_mode or mode,
        "span_mode": span_mode,
        "evidence_spans": [[float(a), float(b)] for a, b in spans],
        "frame_source_labels": [
            {"time": time, "kind": kind, "span": index}
            for time, (kind, index) in zip(sampler_times, labels)
        ],
        "n_evidence_spans": len(spans),
        "n_distinct_evidence_spans_in_k8": len(span_ids),
        "n_evidence_frames_in_k8": n_evidence_frames,
        "seeds_hybrid_correct": seeds,
        "n_seeds_correct": n_seeds,
        "duration": duration,
        "eligible_for_render": eligible,
    }


def _sort_key(row: dict):
    return (
        -int(row["n_seeds_correct"]),
        -int(row["n_distinct_evidence_spans_in_k8"]),
        row["question_id"],
    )


def _batch_key(row: dict):
    tier_rank = {"concrete_wrong": 0, "abstention_flip": 1}.get(row["tier"], 2)
    return (
        0 if row["eligible_for_render"] else 1,
        tier_rank,
        row["question_id"],
    )


def _choice_text(choices: list, letter: str) -> str:
    if not letter or not letter.isalpha():
        return ""
    index = ord(letter.upper()) - ord("A")
    if 0 <= index < len(choices):
        return str(choices[index])
    return ""


def _is_tso(task: str, question_id: str) -> bool:
    text = str(task or "").strip().lower()
    return "temporal shot" in text or text == "tso" or str(question_id).startswith("TSO_")


def _close(left, right, atol: float = 1e-3) -> bool:
    if len(left) != len(right):
        return False
    return all(abs(float(a) - float(b)) <= atol for a, b in zip(left, right))
