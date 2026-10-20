"""Temporal Shot Ordering abstention mask. Inference only.

One score vector per model is turned into two answers: argmax over the
original letters, then argmax after the abstention letter is set to -inf.
The question, labels, and prompt stay as they were.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import HERBENCH_ORACLE_FRAMES
from .herbench_oracle import EXPECTED_TSO_N, TSO_TASK, is_tso_task
from .stats import holm_adjust, mcnemar_exact, win_tie_loss

ABSTENTION_TEXT = (
    "At least two descriptions do not accurately reflect any shot from the video"
)
SCORING_FIRST = "first_token_letter"
SCORING_SEQUENCE = "sequence_logprob"
MODELS = ("vanilla", "frameweave")
ANALYSIS_SEED = 42
N_BOOTSTRAP = 20_000
PUBLISHED_REFERENCE = {
    "n": EXPECTED_TSO_N,
    "vanilla_correct": 13,
    "vanilla_abstention": 136,
    "frameweave_correct": 97,
    "frameweave_abstention": 45,
}
PREDICTION_FIELDS = [
    "run_fingerprint",
    "input_fingerprint",
    "sample_id",
    "video_id",
    "video_path",
    "task_type",
    "model_variant",
    "masked",
    "condition",
    "abstention_letter",
    "original_labels",
    "ground_truth",
    "predicted_letter",
    "correct",
    "valid",
    "predicted_abstention",
    "scoring_mode",
    "prediction_source",
    "raw_native_output",
    "greedy_matches_unmasked",
    "first_new_token_id",
    "status",
    "error",
    "checkpoint_sha256",
    "question",
    "choices_json",
    "option_order",
    "frame_timestamps",
    "oracle_mode",
    "oracle_padded",
    "oracle_frame_times",
    "oracle_frame_indices",
    "loaded_frame_indices",
    "tokens_per_frame",
    "num_frames",
    "runtime_seconds",
]


class ProtocolError(RuntimeError):
    """The TSO table does not match the fixed 152-question protocol."""


class ScoreError(RuntimeError):
    """A score vector cannot be turned into a letter."""


class ManifestResult:
    def __init__(
        self,
        samples: list[dict],
        source_member: str,
        learned_missing_ids: list[str],
        learned_extra_ids: list[str],
    ):
        self.samples = samples
        self.source_member = source_member
        self.learned_missing_ids = learned_missing_ids
        self.learned_extra_ids = learned_extra_ids


_LETTER_PREFIX = re.compile(r"^[a-e][.)]\s+")


def normalize_choice(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def is_abstention_choice(text: str) -> bool:
    """Match the abstention sentence. HERBench lite stores it as ``D. ...``."""
    folded = _LETTER_PREFIX.sub("", normalize_choice(text)).rstrip(".")
    return folded == ABSTENTION_TEXT.casefold()


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def bool_cell(value: bool) -> str:
    return "true" if bool(value) else "false"


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def parse_jsonish(value, default):
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def sha256_canonical(payload) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def fingerprint16(payload) -> str:
    return sha256_canonical(payload)[:16]


def _prediction_csv_name(name: str) -> bool:
    base = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    return base.endswith(".csv") and (
        base.startswith("predictions_residual_") or base.startswith("predictions_reljoin_")
    )


def _zip_member_excluded(name: str) -> bool:
    norm = str(name).replace("\\", "/")
    parts = [part for part in norm.split("/") if part]
    if "smoke" in parts:
        return True
    return (
        "drive_residual_direction/" in norm
        or "drive_eval_full/" in norm
        or "drive_tso_abstention/" in norm
    )


def _direct_member(name: str, folder: str, prefix: str) -> bool:
    parts = [part for part in str(name).replace("\\", "/").split("/") if part]
    return (
        len(parts) == 2
        and parts[0] == folder
        and parts[1].startswith(prefix)
        and parts[1].endswith(".csv")
    )


def select_zip_member(names: list[str]) -> str:
    """One predictions CSV from a HERBench-lite-style ZIP.

    The eval ZIP stores the table at ``predictions/predictions_reljoin_*.csv``
    and may also pack ``drive_eval_full/``. The residual-direction ZIP stores
    ``predictions_residual_direction/predictions_residual_*.csv`` and may also
    pack ``smoke/`` plus ``drive_residual_direction/``. Smoke and Drive copies
    are not the 152-row source.
    """
    csvs = []
    for name in names:
        norm = str(name).replace("\\", "/")
        if _prediction_csv_name(norm):
            csvs.append(norm)
    pool = [name for name in csvs if not _zip_member_excluded(name)]
    residual = [
        name for name in pool if _direct_member(name, "predictions_residual_direction", "predictions_residual_")
    ]
    herbench = [
        name for name in pool if _direct_member(name, "predictions", "predictions_reljoin_")
    ]
    bare_residual = [
        name for name in pool
        if "/" not in name and name.startswith("predictions_residual_")
    ]
    bare_herbench = [
        name for name in pool
        if "/" not in name and name.startswith("predictions_reljoin_")
    ]
    chosen = residual or herbench or bare_residual or bare_herbench
    if len(chosen) != 1:
        raise ProtocolError(
            f"expected one predictions CSV, found {chosen or csvs}"
        )
    return chosen[0]


def load_source_table(path: Path) -> tuple[pd.DataFrame, str]:
    path = Path(path)
    if not path.is_file():
        raise ProtocolError(f"results path not found: {path}")
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as handle:
            member = select_zip_member(handle.namelist())
            with handle.open(member) as raw:
                frame = pd.read_csv(raw)
        return frame, member
    frame = pd.read_csv(path)
    return frame, path.name


def _sample_id(row: dict) -> str:
    for key in ("sample_id", "question_id"):
        text = _text(row.get(key))
        if text:
            return text
    raise ProtocolError("a source row has no sample_id")


def _letters_for(choices: list[str]) -> list[str]:
    letters = list("ABCDE")
    if len(choices) != len(letters):
        raise ProtocolError(f"expected 5 choices, found {len(choices)}")
    if any(not str(choice).strip() for choice in choices):
        raise ProtocolError("a TSO choice is empty")
    return letters


def _ground_truth_letter(value, letters: list[str], sample_id: str) -> str:
    text = _text(value).upper()
    if text in letters:
        return text
    if text and text[0] in letters and (len(text) == 1 or text[1] in ".)"):
        return text[0]
    raise ProtocolError(f"{sample_id}: ground truth {value!r} is not one of {''.join(letters)}")


def _timestamps(value, sample_id: str) -> list[float]:
    raw = parse_jsonish(value, None)
    if not isinstance(raw, list) or len(raw) != int(HERBENCH_ORACLE_FRAMES):
        raise ProtocolError(
            f"{sample_id}: expected {HERBENCH_ORACLE_FRAMES} stored frame timestamps"
        )
    try:
        return [float(item) for item in raw]
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{sample_id}: frame timestamps are not numbers") from exc


def _choice_list(row: dict, sample_id: str) -> list[str]:
    raw = parse_jsonish(row.get("choices_json", row.get("choices")), None)
    if not isinstance(raw, list):
        raise ProtocolError(f"{sample_id}: choices_json is missing")
    return [str(item) for item in raw]


def _row_identity(row: dict) -> tuple:
    sample_id = _sample_id(row)
    choices = _choice_list(row, sample_id)
    return (
        _text(row.get("question")),
        json.dumps(choices, ensure_ascii=False),
        _text(row.get("ground_truth", row.get("answer"))).upper(),
        json.dumps(_timestamps(row.get("frame_timestamps"), sample_id)),
        _text(row.get("video_id")),
        _text(row.get("video_path")),
    )


def _dedupe_rows(rows: list[dict], role: str) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(_sample_id(row), []).append(row)
    kept = []
    for sample_id, group in grouped.items():
        identities = {_row_identity(item) for item in group}
        if len(identities) != 1:
            raise ProtocolError(f"{role} rows for {sample_id} disagree on the question inputs")
        kept.append(group[-1])
    return kept


def _source_correct(row: dict, ground_truth: str) -> bool:
    if _text(row.get("is_correct")):
        return as_bool(row.get("is_correct"))
    return _text(row.get("parsed_prediction")).upper() == ground_truth


def _sample_from_vanilla(row: dict) -> dict:
    sample_id = _sample_id(row)
    choices = _choice_list(row, sample_id)
    letters = _letters_for(choices)
    matches = [
        letter
        for letter, choice in zip(letters, choices)
        if is_abstention_choice(choice)
    ]
    if len(matches) != 1:
        raise ProtocolError(
            f"{sample_id}: expected one abstention choice, found {len(matches)}"
        )
    abstention = matches[0]
    ground_truth = _ground_truth_letter(row.get("ground_truth", row.get("answer")), letters, sample_id)
    if ground_truth == abstention:
        raise ProtocolError(
            f"{sample_id}: ground truth is the abstention letter {abstention}"
        )
    video_id = _text(row.get("video_id"))
    if not video_id:
        raise ProtocolError(f"{sample_id}: missing video_id")
    timestamps = _timestamps(row.get("frame_timestamps"), sample_id)
    option_order = parse_jsonish(row.get("option_order"), letters)
    if not isinstance(option_order, list):
        option_order = letters
    prediction = _text(row.get("parsed_prediction")).upper()
    return {
        "sample_id": sample_id,
        "video_id": video_id,
        "video_path": _text(row.get("video_path")),
        "task_type": TSO_TASK,
        "question": _text(row.get("question")),
        "choices": choices,
        "letters": letters,
        "option_order": [str(item) for item in option_order],
        "ground_truth": ground_truth,
        "abstention_letter": abstention,
        "abstention_index": letters.index(abstention),
        "frame_timestamps": timestamps,
        "oracle_mode": _text(row.get("oracle_mode")),
        "oracle_padded": as_bool(row.get("oracle_padded")),
        "oracle_frame_times": parse_jsonish(row.get("oracle_frame_times"), timestamps),
        "oracle_frame_indices": parse_jsonish(row.get("oracle_frame_indices"), []),
        "tokens_per_frame": parse_jsonish(row.get("tokens_per_frame"), []),
        "num_frames": len(timestamps),
        "source_vanilla_prediction": prediction,
        "source_vanilla_correct": _source_correct(row, ground_truth),
        "source_vanilla_abstention": prediction == abstention,
        "source_learned_prediction": "",
        "source_learned_correct": False,
        "source_learned_abstention": False,
    }


def _attach_learned(sample: dict, row: dict) -> None:
    prediction = _text(row.get("parsed_prediction")).upper()
    sample["source_learned_prediction"] = prediction
    sample["source_learned_correct"] = _source_correct(row, sample["ground_truth"])
    sample["source_learned_abstention"] = prediction == sample["abstention_letter"]


_LEARNED_LABELS = {
    "learned",
    "hybrid",
    "reljoin",
    "frameweave-hybrid",
    "frameweave_hybrid",
}


def _condition_label(row: dict) -> str:
    """Residual-direction uses condition; HERBench lite eval uses method."""
    condition = _text(row.get("condition")).lower()
    if condition in {"vanilla", "learned"}:
        return condition
    method = _text(row.get("method")).lower().replace(" ", "")
    if method == "vanilla":
        return "vanilla"
    if method in _LEARNED_LABELS:
        return "learned"
    return ""


def build_tso_manifest(
    frame: pd.DataFrame,
    *,
    source_member: str = "",
    expected_n: int = EXPECTED_TSO_N,
) -> ManifestResult:
    if "task_type" not in frame.columns:
        raise ProtocolError("source table is missing task_type")
    if "condition" not in frame.columns and "method" not in frame.columns:
        raise ProtocolError("source table is missing condition or method")
    work = frame.where(frame.notna(), None)
    records = work.to_dict(orient="records")
    if "status" in frame.columns:
        records = [row for row in records if _text(row.get("status")).lower() == "ok"]
    tso_rows = [row for row in records if is_tso_task(_text(row.get("task_type")))]
    vanilla_rows = _dedupe_rows(
        [row for row in tso_rows if _condition_label(row) == "vanilla"],
        "vanilla",
    )
    learned_rows = _dedupe_rows(
        [row for row in tso_rows if _condition_label(row) == "learned"],
        "learned",
    )
    if len(vanilla_rows) != int(expected_n):
        raise ProtocolError(
            f"expected {expected_n} vanilla {TSO_TASK} samples, found {len(vanilla_rows)}"
        )
    samples = [_sample_from_vanilla(row) for row in vanilla_rows]
    samples.sort(key=lambda item: item["sample_id"])
    ids = [item["sample_id"] for item in samples]
    if len(set(ids)) != len(ids):
        raise ProtocolError("duplicate vanilla sample_id after the TSO filter")
    learned_by_id = {_sample_id(row): row for row in learned_rows}
    missing = []
    for sample in samples:
        learned = learned_by_id.get(sample["sample_id"])
        if learned is None:
            missing.append(sample["sample_id"])
            continue
        _attach_learned(sample, learned)
    extra = sorted(set(learned_by_id) - set(ids))
    return ManifestResult(samples, source_member, missing, extra)


def canonical_manifest_hash(samples: list[dict]) -> str:
    payload = []
    for sample in sorted(samples, key=lambda item: item["sample_id"]):
        payload.append(
            {
                "sample_id": sample["sample_id"],
                "video_id": sample["video_id"],
                "video_path": sample["video_path"],
                "question": sample["question"],
                "choices": sample["choices"],
                "letters": sample["letters"],
                "ground_truth": sample["ground_truth"],
                "abstention_letter": sample["abstention_letter"],
                "frame_timestamps": sample["frame_timestamps"],
            }
        )
    return sha256_canonical(payload)


def select_smoke_ids(samples: list[dict], n: int) -> list[str]:
    """First sample of each abstention letter A–E, then the next unused ids.

    Selection follows sample_id order and does not look at correctness.
    """
    ordered = sorted(samples, key=lambda item: str(item["sample_id"]))
    chosen: list[str] = []
    seen: set[str] = set()
    for letter in "ABCDE":
        for sample in ordered:
            if sample.get("abstention_letter") == letter and sample["sample_id"] not in seen:
                chosen.append(sample["sample_id"])
                seen.add(sample["sample_id"])
                break
        if len(chosen) >= int(n):
            return chosen[: int(n)]
    for sample in ordered:
        if len(chosen) >= int(n):
            break
        if sample["sample_id"] not in seen:
            chosen.append(sample["sample_id"])
            seen.add(sample["sample_id"])
    return chosen[: int(n)]


def smoke_id_payload(samples: list[dict], sample_ids: list[str]) -> dict:
    by_id = {sample["sample_id"]: sample for sample in samples}
    return {
        "sample_ids": list(sample_ids),
        "abstention_letters": {
            sample_id: by_id[sample_id]["abstention_letter"] for sample_id in sample_ids
        },
        "n": len(sample_ids),
    }


def generation_flags(model_variant: str) -> dict:
    if model_variant == "vanilla":
        return {"bypass": True, "inject": False}
    if model_variant == "frameweave":
        return {"bypass": False, "inject": True}
    raise ProtocolError(f"unknown model variant {model_variant}")


def assert_prompt_keeps_choices(prompt: str, choices: list[str], letters: list[str]) -> None:
    folded = normalize_choice(prompt)
    for letter, choice in zip(letters, choices):
        if normalize_choice(choice) not in folded:
            raise ProtocolError(f"prompt is missing choice {letter}")
        if letter not in prompt:
            raise ProtocolError(f"prompt is missing label {letter}")


def scoring_mode_for_letters(encoded: dict[str, list[int]]) -> str:
    seen = []
    for ids in encoded.values():
        if len(ids) != 1:
            return SCORING_SEQUENCE
        seen.append(int(ids[0]))
    if len(set(seen)) != len(seen):
        return SCORING_SEQUENCE
    return SCORING_FIRST


def greedy_matches_unmasked(
    *,
    first_token_id,
    letter_token_ids: dict[str, list[int]],
    pred_unmasked: str,
    parsed_letter: str,
    parsed_ok: bool,
) -> bool:
    expected = letter_token_ids.get(pred_unmasked) or []
    if len(expected) != 1 or first_token_id is None:
        return False
    if int(first_token_id) != int(expected[0]):
        return False
    return bool(parsed_ok) and str(parsed_letter) == str(pred_unmasked)


def apply_abstention_mask(
    scores: dict[str, float],
    letters: list[str],
    abstention: str,
) -> dict[str, float]:
    if abstention not in letters:
        raise ProtocolError(f"abstention letter {abstention} is not a choice")
    masked = {letter: float(scores[letter]) for letter in letters}
    masked[abstention] = float("-inf")
    return masked


def argmax_letter(scores: dict[str, float], letters: list[str]) -> str:
    best = None
    best_score = float("-inf")
    for letter in letters:
        value = float(scores[letter])
        if not math.isfinite(value):
            continue
        if best is None or value > best_score:
            best = letter
            best_score = value
    if best is None:
        raise ScoreError("no finite letter score")
    return best


def derive_pair(
    scores: dict[str, float],
    letters: list[str],
    abstention: str,
    ground_truth: str,
) -> dict:
    letters = list(letters)
    if ground_truth not in letters:
        raise ProtocolError(f"ground truth {ground_truth} is not in {''.join(letters)}")
    if ground_truth == abstention:
        raise ProtocolError("ground truth is the abstention letter")
    for letter in letters:
        if not math.isfinite(float(scores[letter])):
            raise ScoreError(f"non-finite score for {letter}")
    masked_scores = apply_abstention_mask(scores, letters, abstention)
    for letter in letters:
        if letter == abstention:
            if masked_scores[letter] != float("-inf"):
                raise ProtocolError("abstention score was not masked")
            continue
        if masked_scores[letter] != float(scores[letter]):
            raise ProtocolError(f"mask changed the score of {letter}")
    pred_unmasked = argmax_letter(scores, letters)
    pred_masked = argmax_letter(masked_scores, letters)
    if pred_masked == abstention or pred_masked not in letters or pred_unmasked not in letters:
        raise ProtocolError("prediction left the original letter set")
    correct_unmasked = pred_unmasked == ground_truth
    correct_masked = pred_masked == ground_truth
    if correct_unmasked and not correct_masked:
        raise ProtocolError("a correct unmasked answer became incorrect after the mask")
    return {
        "pred_unmasked": pred_unmasked,
        "pred_masked": pred_masked,
        "correct_unmasked": correct_unmasked,
        "correct_masked": correct_masked,
        "masked_scores": masked_scores,
        "predicted_abstention_unmasked": pred_unmasked == abstention,
        "predicted_abstention_masked": False,
    }


def encode_scores(scores: dict[str, float]) -> dict[str, float | str]:
    encoded: dict[str, float | str] = {}
    for letter, value in scores.items():
        number = float(value)
        if math.isnan(number):
            encoded[letter] = "nan"
        elif number == float("-inf"):
            encoded[letter] = "-inf"
        elif number == float("inf"):
            encoded[letter] = "inf"
        else:
            encoded[letter] = number
    return encoded


def decode_scores(scores: dict) -> dict[str, float]:
    decoded = {}
    for letter, value in scores.items():
        if value == "-inf":
            decoded[str(letter)] = float("-inf")
        elif value == "inf":
            decoded[str(letter)] = float("inf")
        elif value == "nan":
            decoded[str(letter)] = float("nan")
        else:
            decoded[str(letter)] = float(value)
    return decoded


def input_fingerprint(
    sample: dict,
    prompt: str,
    scoring_mode: str,
    min_pixels: int,
    max_pixels: int,
) -> str:
    return fingerprint16(
        {
            "sample_id": sample["sample_id"],
            "question": sample["question"],
            "choices": sample["choices"],
            "letters": sample["letters"],
            "frame_timestamps": sample["frame_timestamps"],
            "prompt": prompt,
            "scoring_mode": scoring_mode,
            "min_pixels": int(min_pixels),
            "max_pixels": int(max_pixels),
            "num_frames": len(sample["frame_timestamps"]),
        }
    )


def _shared_context(context: dict) -> dict:
    return {key: context.get(key, "") for key in PREDICTION_FIELDS}


def prediction_pair_records(
    *,
    scores: dict[str, float],
    letters: list[str],
    abstention_letter: str,
    ground_truth: str,
    context: dict,
) -> list[dict]:
    derived = derive_pair(scores, letters, abstention_letter, ground_truth)
    records = []
    for masked, pred, correct, abstained in (
        (
            False,
            derived["pred_unmasked"],
            derived["correct_unmasked"],
            derived["predicted_abstention_unmasked"],
        ),
        (
            True,
            derived["pred_masked"],
            derived["correct_masked"],
            derived["predicted_abstention_masked"],
        ),
    ):
        record = _shared_context(context)
        model = str(context.get("model_variant", ""))
        record.update(
            {
                "model_variant": model,
                "masked": bool_cell(masked),
                "condition": f"{model}_{'masked' if masked else 'unmasked'}",
                "abstention_letter": abstention_letter,
                "original_labels": json.dumps(list(letters)),
                "ground_truth": ground_truth,
                "predicted_letter": pred,
                "correct": bool_cell(correct),
                "valid": "true",
                "predicted_abstention": bool_cell(abstained),
                "prediction_source": "candidate_scores",
                "status": "ok",
                "error": "",
            }
        )
        records.append(record)
    if records[0]["raw_native_output"] != records[1]["raw_native_output"]:
        raise ProtocolError("masked and unmasked rows do not share the raw generation")
    return records


def error_pair_records(context: dict, message: str) -> list[dict]:
    records = []
    model = str(context.get("model_variant", ""))
    abstention = str(context.get("abstention_letter", ""))
    for masked in (False, True):
        record = _shared_context(context)
        record.update(
            {
                "model_variant": model,
                "masked": bool_cell(masked),
                "condition": f"{model}_{'masked' if masked else 'unmasked'}",
                "abstention_letter": abstention,
                "predicted_letter": "",
                "correct": "false",
                "valid": "false",
                "predicted_abstention": "false",
                "prediction_source": "",
                "status": "runtime_error",
                "error": message,
            }
        )
        records.append(record)
    return records


def append_csv(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    fields = PREDICTION_FIELDS
    if exists:
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), None)
        if header:
            fields = list(header)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: record.get(key, "") for key in fields})
        handle.flush()


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def load_prediction_rows(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return frame.to_dict(orient="records")


def load_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def dedupe_predictions(rows: list[dict]) -> list[dict]:
    latest = {}
    for row in rows:
        key = (
            _text(row.get("sample_id")),
            _text(row.get("model_variant")),
            bool_cell(as_bool(row.get("masked"))),
        )
        latest[key] = row
    return list(latest.values())


def dedupe_scores(rows: list[dict]) -> list[dict]:
    latest = {}
    for row in rows:
        latest[(_text(row.get("sample_id")), _text(row.get("model_variant")))] = row
    return list(latest.values())


def resume_plan(
    pred_rows: list[dict],
    score_rows: list[dict],
    fingerprint: str,
    expected_input: dict[tuple[str, str], str],
) -> dict:
    foreign = False
    preds = {}
    for row in pred_rows:
        row_fp = _text(row.get("run_fingerprint"))
        if row_fp and row_fp != fingerprint:
            foreign = True
            continue
        if _text(row.get("status")) != "ok":
            continue
        key = (
            _text(row.get("sample_id")),
            _text(row.get("model_variant")),
            bool_cell(as_bool(row.get("masked"))),
        )
        preds[key] = row
    scores = {}
    for row in dedupe_scores(score_rows):
        row_fp = _text(row.get("run_fingerprint"))
        if row_fp and row_fp != fingerprint:
            foreign = True
            continue
        scores[(_text(row.get("sample_id")), _text(row.get("model_variant")))] = row
    done = set()
    rebuild = []
    for (sample_id, model), expected in expected_input.items():
        unmasked = preds.get((sample_id, model, "false"))
        masked = preds.get((sample_id, model, "true"))
        score = scores.get((sample_id, model))
        pair_ok = (
            unmasked is not None
            and masked is not None
            and _text(unmasked.get("input_fingerprint")) == expected
            and _text(masked.get("input_fingerprint")) == expected
            and _text(unmasked.get("raw_native_output")) == _text(masked.get("raw_native_output"))
            and _text(unmasked.get("scoring_mode")) == _text(masked.get("scoring_mode"))
            and score is not None
            and _text(score.get("input_fingerprint")) == expected
        )
        if pair_ok:
            done.add((sample_id, model))
        elif score is not None and _text(score.get("input_fingerprint")) == expected:
            rebuild.append(score)
    return {"done": done, "rebuild": rebuild, "foreign": foreign}


def predictions_from_score_record(record: dict) -> list[dict]:
    context = dict(record.get("context") or {})
    return prediction_pair_records(
        scores=decode_scores(record["scores"]),
        letters=list(record["letters"]),
        abstention_letter=str(record["abstention_letter"]),
        ground_truth=str(record["ground_truth"]),
        context=context,
    )


def video_question_groups(samples: list[dict]) -> tuple[list[str], list[list[int]]]:
    buckets: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        video_id = str(sample["video_id"])
        buckets.setdefault(video_id, []).append(index)
    order = sorted(buckets)
    return order, [buckets[video_id] for video_id in order]


def video_cluster_draws(n_videos: int, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n_videos), size=(int(n_resamples), int(n_videos)))


def expand_video_draw(groups: list[list[int]], picks: np.ndarray) -> list[int]:
    chosen = []
    for pick in picks:
        chosen.extend(groups[int(pick)])
    return chosen


def _outcome_matrix(samples: list[dict]) -> dict[str, np.ndarray]:
    columns = {
        "vanilla_unmasked": np.zeros(len(samples), dtype=np.float64),
        "vanilla_masked": np.zeros(len(samples), dtype=np.float64),
        "frameweave_unmasked": np.zeros(len(samples), dtype=np.float64),
        "frameweave_masked": np.zeros(len(samples), dtype=np.float64),
    }
    for index, sample in enumerate(samples):
        for name in columns:
            columns[name][index] = 1.0 if bool(sample[name]) else 0.0
    return columns


def video_cluster_bootstrap(
    samples: list[dict],
    *,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = ANALYSIS_SEED,
    picks: np.ndarray | None = None,
    return_samples: bool = False,
) -> dict:
    """Resample video_id clusters once and apply that draw to all four conditions."""
    if not samples:
        raise ProtocolError("bootstrap received no samples")
    video_ids, groups = video_question_groups(samples)
    n_videos = len(groups)
    if picks is None:
        picks = video_cluster_draws(n_videos, n_resamples, seed)
    else:
        picks = np.asarray(picks)
        n_resamples = int(picks.shape[0])
    matrix = _outcome_matrix(samples)
    names = (
        "vanilla_unmasked",
        "vanilla_masked",
        "frameweave_unmasked",
        "frameweave_masked",
    )
    sums = np.zeros((n_videos, 4), dtype=np.float64)
    counts = np.zeros(n_videos, dtype=np.float64)
    for video_index, members in enumerate(groups):
        counts[video_index] = float(len(members))
        for column, name in enumerate(names):
            sums[video_index, column] = float(matrix[name][members].sum())
    boot_sums = sums[picks].sum(axis=1)
    boot_counts = counts[picks].sum(axis=1)
    means = boot_sums / boot_counts[:, None]
    g_unmasked = means[:, 2] - means[:, 0]
    g_masked = means[:, 3] - means[:, 1]
    difference = g_masked - g_unmasked
    point_u = float(matrix["frameweave_unmasked"].mean() - matrix["vanilla_unmasked"].mean())
    point_m = float(matrix["frameweave_masked"].mean() - matrix["vanilla_masked"].mean())
    payload = {
        "n_videos": n_videos,
        "n_questions": len(samples),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "video_ids": video_ids,
        "G_unmasked": point_u,
        "G_masked": point_m,
        "difference": point_m - point_u,
        "G_masked_ci": [
            float(np.quantile(g_masked, 0.025)),
            float(np.quantile(g_masked, 0.975)),
        ],
        "difference_ci": [
            float(np.quantile(difference, 0.025)),
            float(np.quantile(difference, 0.975)),
        ],
    }
    if return_samples:
        payload["G_masked_samples"] = g_masked.tolist()
        payload["difference_samples"] = difference.tolist()
    return payload


def _condition_rows(rows: list[dict], model: str, masked: bool) -> dict[str, dict]:
    wanted = bool_cell(masked)
    found = {}
    for row in rows:
        if _text(row.get("model_variant")) != model:
            continue
        if bool_cell(as_bool(row.get("masked"))) != wanted:
            continue
        found[_text(row.get("sample_id"))] = row
    return found


def _paired_bools(left: dict[str, dict], right: dict[str, dict], sample_ids: list[str]) -> tuple[list[bool], list[bool]]:
    a, b = [], []
    for sample_id in sample_ids:
        a.append(as_bool(left[sample_id].get("correct")))
        b.append(as_bool(right[sample_id].get("correct")))
    return a, b


def _contrast(name: str, role: str, a_name: str, b_name: str, left: dict, right: dict, sample_ids: list[str]) -> dict:
    correct_a, correct_b = _paired_bools(left, right, sample_ids)
    paired = mcnemar_exact(correct_a, correct_b)
    wins = win_tie_loss(correct_a, correct_b)
    return {
        "name": name,
        "role": role,
        "a": a_name,
        "b": b_name,
        "mcnemar": paired,
        "rescues": wins["wins_b"],
        "regressions": wins["wins_a"],
        "ties": wins["ties"],
        "holm_p_value": None,
    }


def _coverage_row(model: str, masked: bool, found: dict[str, dict], sample_ids: list[str]) -> dict:
    n_expected = len(sample_ids)
    present = [found[sample_id] for sample_id in sample_ids if sample_id in found]
    ok = [row for row in present if _text(row.get("status")) == "ok"]
    correct = sum(1 for row in ok if as_bool(row.get("correct")))
    valid = sum(1 for row in ok if as_bool(row.get("valid")))
    abstention = sum(1 for row in ok if as_bool(row.get("predicted_abstention")))
    return {
        "model_variant": model,
        "masked": bool(masked),
        "condition": f"{model}_{'masked' if masked else 'unmasked'}",
        "correct": int(correct),
        "n": int(n_expected),
        "accuracy": float(correct / n_expected) if n_expected else 0.0,
        "abstention_rate": float(abstention / n_expected) if n_expected else 0.0,
        "valid_coverage": float(valid / n_expected) if n_expected else 0.0,
        "n_ok": len(ok),
        "n_errors": int(sum(1 for row in present if _text(row.get("status")) != "ok")),
        "n_missing": int(n_expected - len(present)),
    }


def source_reference(samples: list[dict]) -> dict:
    vanilla_correct = sum(1 for sample in samples if sample["source_vanilla_correct"])
    vanilla_abstention = sum(1 for sample in samples if sample["source_vanilla_abstention"])
    frameweave_correct = sum(1 for sample in samples if sample["source_learned_correct"])
    frameweave_abstention = sum(1 for sample in samples if sample["source_learned_abstention"])
    source = {
        "n": len(samples),
        "vanilla_correct": vanilla_correct,
        "vanilla_abstention": vanilla_abstention,
        "frameweave_correct": frameweave_correct,
        "frameweave_abstention": frameweave_abstention,
    }
    published = dict(PUBLISHED_REFERENCE)
    matches = source == published
    return {"published": published, "source": source, "published_matches_source": matches}


def prediction_changes(samples: list[dict], rows: list[dict]) -> dict:
    by_id = {sample["sample_id"]: sample for sample in samples}
    changes = {
        "vanilla_unmasked_changed": 0,
        "vanilla_unmasked_compared": 0,
        "frameweave_unmasked_changed": 0,
        "frameweave_unmasked_compared": 0,
    }
    for row in rows:
        if bool_cell(as_bool(row.get("masked"))) != "false":
            continue
        if _text(row.get("status")) != "ok":
            continue
        sample = by_id.get(_text(row.get("sample_id")))
        if sample is None:
            continue
        model = _text(row.get("model_variant"))
        predicted = _text(row.get("predicted_letter")).upper()
        if model == "vanilla" and sample["source_vanilla_prediction"]:
            changes["vanilla_unmasked_compared"] += 1
            if predicted != sample["source_vanilla_prediction"]:
                changes["vanilla_unmasked_changed"] += 1
        if model == "frameweave" and sample["source_learned_prediction"]:
            changes["frameweave_unmasked_compared"] += 1
            if predicted != sample["source_learned_prediction"]:
                changes["frameweave_unmasked_changed"] += 1
    return changes


def build_reports(
    samples: list[dict],
    rows: list[dict],
    *,
    n_manifest: int,
    run_fingerprint: str,
    scoring_mode: str,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = ANALYSIS_SEED,
    full_manifest: list[dict] | None = None,
) -> tuple[dict, dict]:
    rows = dedupe_predictions(rows)
    sample_ids = [sample["sample_id"] for sample in samples]
    modes = {_text(row.get("scoring_mode")) for row in rows if _text(row.get("status")) == "ok"}
    mixed_modes = len(modes) > 1
    table = []
    found = {}
    for model in MODELS:
        for masked in (False, True):
            condition_found = _condition_rows(rows, model, masked)
            found[(model, masked)] = condition_found
            table.append(_coverage_row(model, masked, condition_found, sample_ids))
    n_expected_rows = 4 * len(samples)
    n_ok = sum(item["n_ok"] for item in table)
    reasons = []
    if n_ok != n_expected_rows:
        reasons.append(f"ok rows {n_ok} / {n_expected_rows}")
    if mixed_modes:
        reasons.append("mixed scoring modes")
    for model in MODELS:
        unmasked = next(item for item in table if item["model_variant"] == model and not item["masked"])
        masked = next(item for item in table if item["model_variant"] == model and item["masked"])
        if unmasked["n_ok"] == unmasked["n"] and masked["n_ok"] == masked["n"]:
            if masked["accuracy"] + 1e-12 < unmasked["accuracy"]:
                reasons.append(f"{model} masked accuracy is below unmasked accuracy")
    incomplete = bool(reasons)
    reference_samples = full_manifest if full_manifest is not None else samples
    summary = {
        "incomplete": incomplete,
        "incomplete_reasons": reasons,
        "population": "full" if len(samples) == EXPECTED_TSO_N == n_manifest else "subset",
        "n_expected_samples": len(samples),
        "n_manifest": int(n_manifest),
        "n_expected_rows": n_expected_rows,
        "n_ok_rows": n_ok,
        "n_videos": len({sample["video_id"] for sample in samples}),
        "scoring_mode": scoring_mode,
        "scoring_modes_seen": sorted(modes),
        "run_fingerprint": run_fingerprint,
        "table": table,
        "gains": None,
        "reference": source_reference(reference_samples),
        "prediction_changes_vs_source": prediction_changes(samples, rows),
    }
    stats = {
        "incomplete": True,
        "analysis_seed": int(seed),
        "n_resamples": int(n_resamples),
        "cluster": "video_id",
        "n_videos": summary["n_videos"],
        "primary": None,
        "secondary": [],
        "difference_of_gains": None,
    }
    if incomplete:
        return summary, stats
    aligned = []
    for sample in samples:
        aligned.append(
            {
                "sample_id": sample["sample_id"],
                "video_id": sample["video_id"],
                "vanilla_unmasked": as_bool(found[("vanilla", False)][sample["sample_id"]]["correct"]),
                "vanilla_masked": as_bool(found[("vanilla", True)][sample["sample_id"]]["correct"]),
                "frameweave_unmasked": as_bool(found[("frameweave", False)][sample["sample_id"]]["correct"]),
                "frameweave_masked": as_bool(found[("frameweave", True)][sample["sample_id"]]["correct"]),
            }
        )
    primary = _contrast(
        "frameweave_masked_vs_vanilla_masked",
        "primary",
        "vanilla_masked",
        "frameweave_masked",
        found[("vanilla", True)],
        found[("frameweave", True)],
        sample_ids,
    )
    secondary = [
        _contrast(
            "frameweave_unmasked_vs_vanilla_unmasked",
            "secondary",
            "vanilla_unmasked",
            "frameweave_unmasked",
            found[("vanilla", False)],
            found[("frameweave", False)],
            sample_ids,
        ),
        _contrast(
            "vanilla_masked_vs_vanilla_unmasked",
            "secondary",
            "vanilla_unmasked",
            "vanilla_masked",
            found[("vanilla", False)],
            found[("vanilla", True)],
            sample_ids,
        ),
        _contrast(
            "frameweave_masked_vs_frameweave_unmasked",
            "secondary",
            "frameweave_unmasked",
            "frameweave_masked",
            found[("frameweave", False)],
            found[("frameweave", True)],
            sample_ids,
        ),
    ]
    adjusted = holm_adjust([item["mcnemar"]["p_value"] for item in secondary])
    for item, p_value in zip(secondary, adjusted):
        item["holm_p_value"] = p_value
    boot = video_cluster_bootstrap(aligned, n_resamples=n_resamples, seed=seed)
    summary["gains"] = {
        "G_unmasked": boot["G_unmasked"],
        "G_masked": boot["G_masked"],
        "G_unmasked_pp": boot["G_unmasked"] * 100.0,
        "G_masked_pp": boot["G_masked"] * 100.0,
        "difference": boot["difference"],
        "difference_pp": boot["difference"] * 100.0,
    }
    stats = {
        "incomplete": False,
        "analysis_seed": int(seed),
        "n_resamples": int(boot["n_resamples"]),
        "cluster": "video_id",
        "n_videos": boot["n_videos"],
        "video_ids": boot["video_ids"],
        "primary": primary,
        "primary_bootstrap": {
            "delta": boot["G_masked"],
            "ci_low": boot["G_masked_ci"][0],
            "ci_high": boot["G_masked_ci"][1],
            "delta_pp": boot["G_masked"] * 100.0,
            "ci_low_pp": boot["G_masked_ci"][0] * 100.0,
            "ci_high_pp": boot["G_masked_ci"][1] * 100.0,
        },
        "secondary": secondary,
        "difference_of_gains": {
            "delta": boot["difference"],
            "ci_low": boot["difference_ci"][0],
            "ci_high": boot["difference_ci"][1],
            "delta_pp": boot["difference"] * 100.0,
            "ci_low_pp": boot["difference_ci"][0] * 100.0,
            "ci_high_pp": boot["difference_ci"][1] * 100.0,
        },
    }
    return summary, stats


def _fmt_pp(value: float) -> str:
    return f"{value:.2f}"


def _fmt_p(value: float) -> str:
    return f"{value:.4g}"


def render_results_md(summary: dict, stats: dict) -> str:
    lines = ["# TSO abstention mask", ""]
    if summary.get("incomplete"):
        reason = ", ".join(summary.get("incomplete_reasons") or [])
        lines.append(
            f"Run incomplete ({reason or 'missing rows'}). No gain is reported."
        )
        lines.append("")
    else:
        population = summary.get("population")
        if population == "full":
            lines.append(
                f"Full Temporal Shot Ordering set: {summary['n_expected_samples']} questions, "
                f"{summary['n_videos']} videos. Scoring mode `{summary['scoring_mode']}`."
            )
        else:
            lines.append(
                f"Subset of {summary['n_expected_samples']} questions "
                f"({summary['n_videos']} videos). Gains below describe this subset. "
                f"Scoring mode `{summary['scoring_mode']}`."
            )
        lines.append("")
    lines.extend(
        [
            "## Coverage",
            "",
            "| Model | Mask | Correct | N | Accuracy | Abstention | Valid |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary.get("table") or []:
        lines.append(
            "| {model} | {mask} | {correct} | {n} | {acc:.4f} | {abst:.4f} | {valid:.4f} |".format(
                model=row["model_variant"],
                mask="on" if row["masked"] else "off",
                correct=row["correct"],
                n=row["n"],
                acc=row["accuracy"],
                abst=row["abstention_rate"],
                valid=row["valid_coverage"],
            )
        )
    lines.append("")
    if not summary.get("incomplete") and summary.get("gains") and stats.get("primary"):
        gains = summary["gains"]
        primary = stats["primary"]
        boot = stats["primary_bootstrap"]
        diff = stats["difference_of_gains"]
        lines.extend(
            [
                "## Contrasts",
                "",
                "Primary contrast: FrameWeave masked minus Vanilla masked.",
                (
                    f"G_masked = {_fmt_pp(gains['G_masked_pp'])} percentage points "
                    f"(95% video-cluster CI {_fmt_pp(boot['ci_low_pp'])} to {_fmt_pp(boot['ci_high_pp'])})."
                ),
                (
                    f"McNemar exact p = {_fmt_p(primary['mcnemar']['p_value'])} "
                    "(primary, not Holm-adjusted). "
                    f"Rescues {primary['rescues']}, regressions {primary['regressions']}."
                ),
                (
                    f"G_unmasked = {_fmt_pp(gains['G_unmasked_pp'])} percentage points."
                ),
                (
                    "Difference of gains G_masked - G_unmasked = "
                    f"{_fmt_pp(gains['difference_pp'])} percentage points "
                    f"(95% CI {_fmt_pp(diff['ci_low_pp'])} to {_fmt_pp(diff['ci_high_pp'])})."
                ),
                "",
                f"Video clusters: {stats['n_videos']}. Bootstrap resamples: {stats['n_resamples']}. Seed {stats['analysis_seed']}.",
                "",
                "Secondary tests, Holm-adjusted p:",
                "",
            ]
        )
        for item in stats.get("secondary") or []:
            lines.append(
                f"- {item['name']}: McNemar p = {_fmt_p(item['mcnemar']['p_value'])}, "
                f"Holm p = {_fmt_p(item['holm_p_value'])}, "
                f"rescues {item['rescues']}, regressions {item['regressions']}."
            )
        lines.append("")
    reference = summary.get("reference") or {}
    source = reference.get("source") or {}
    published = reference.get("published") or {}
    changes = summary.get("prediction_changes_vs_source") or {}
    if source:
        lines.extend(
            [
                "## Source reference",
                "",
                (
                    f"Published Vanilla {published.get('vanilla_correct')}/{published.get('n')} correct "
                    f"and {published.get('vanilla_abstention')}/{published.get('n')} abstention. "
                    f"This CSV has {source.get('vanilla_correct')}/{source.get('n')} correct "
                    f"and {source.get('vanilla_abstention')}/{source.get('n')} abstention."
                ),
                (
                    f"Published FrameWeave {published.get('frameweave_correct')}/{published.get('n')} correct "
                    f"and {published.get('frameweave_abstention')}/{published.get('n')} abstention. "
                    f"This CSV has {source.get('frameweave_correct')}/{source.get('n')} correct "
                    f"and {source.get('frameweave_abstention')}/{source.get('n')} abstention."
                ),
                (
                    "Unmasked predictions changed on "
                    f"{changes.get('vanilla_unmasked_changed', 0)}/"
                    f"{changes.get('vanilla_unmasked_compared', 0)} Vanilla questions and "
                    f"{changes.get('frameweave_unmasked_changed', 0)}/"
                    f"{changes.get('frameweave_unmasked_compared', 0)} FrameWeave questions."
                ),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def render_results_tex(summary: dict, stats: dict) -> str:
    lines = [
        "% TSO abstention mask",
        "\\begin{tabular}{llrrrrr}",
        "Model & Mask & Correct & N & Accuracy & Abstention & Valid \\\\",
        "\\hline",
    ]
    for row in summary.get("table") or []:
        lines.append(
            "{model} & {mask} & {correct} & {n} & {acc:.4f} & {abst:.4f} & {valid:.4f} \\\\".format(
                model=_tex_escape(row["model_variant"]),
                mask="on" if row["masked"] else "off",
                correct=row["correct"],
                n=row["n"],
                acc=row["accuracy"],
                abst=row["abstention_rate"],
                valid=row["valid_coverage"],
            )
        )
    lines.append("\\end{tabular}")
    if summary.get("incomplete"):
        lines.append("")
        lines.append("Run incomplete. No gain is reported.")
    elif summary.get("gains") and stats.get("difference_of_gains"):
        gains = summary["gains"]
        diff = stats["difference_of_gains"]
        lines.append("")
        lines.append(
            "Difference of gains "
            f"{gains['difference_pp']:.2f} percentage points "
            f"(95\\% CI {diff['ci_low_pp']:.2f} to {diff['ci_high_pp']:.2f})."
        )
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_reports(output_dir: Path, summary: dict, stats: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "statistical_tests.json", stats)
    (output_dir / "results.md").write_text(render_results_md(summary, stats), encoding="utf-8")
    (output_dir / "results.tex").write_text(render_results_tex(summary, stats), encoding="utf-8")
