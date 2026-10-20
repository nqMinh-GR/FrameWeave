"""Choose visualization candidates before any routing figure exists.

Selection uses task family, evidence-span count, and, when an official eval
CSV is present, vanilla versus hybrid correctness. It does not rank heatmaps
and it does not replace oracle frames.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ..herbench_oracle import extract_evidence_spans, is_tso_task

SURVEY_RULE_VERSION = "1"
SURVEY_SORT_FIELDS = ("priority", "question_id")
QUOTA_SUCCESS = 12
QUOTA_CONTROL = 8
QUOTA_FAILURE = 6
QUOTA_FAILURE_EACH = 3
QUOTA_TSO = 2
QUOTA_PENDING_MAIN = 26

ENTITY_TASKS = {
    "Multi-Entities Grounding and Localization",
    "Appearance-Grounded Localization & Trajectory",
    "Appearance-Grounded Behavior & Interactions",
    "Appearance-Grounded Attribute Recognition",
    "Region Localized People Counting",
}
STATE_TASKS = {
    "False Object Memory",
    "False Action Memory",
    "Action Sequence Integrity Identification",
    "Action Counting",
}
PRIORITY_FAMILIES = {"entity", "state"}
PAPER_SLOTS = (
    "main_success_1",
    "main_success_2",
    "supplementary_control",
    "supplementary_failure",
)
ROLE_ORDER = {
    "success": 0,
    "control": 1,
    "still_wrong": 2,
    "regression": 3,
    "tso_abstain": 4,
    "tso_wrong_order": 5,
    "tso_other": 6,
    "tso_correct": 7,
    "pending_prediction": 8,
}
CANDIDATE_FIELDS = [
    "question_id",
    "video_id",
    "task",
    "family",
    "n_evidence_spans",
    "oracle_mode",
    "span_text",
    "eligible",
    "selected_for_run",
    "run_order",
    "quota",
    "prior_role",
    "reason",
    "vanilla_correct",
    "hybrid_correct",
    "question",
]


def family_of(task: str) -> str:
    if is_tso_task(task):
        return "tso"
    if task in ENTITY_TASKS:
        return "entity"
    if task in STATE_TASKS:
        return "state"
    return "other"


def priority_of(family: str) -> int:
    return {"entity": 0, "state": 1, "other": 2, "tso": 3}.get(family, 2)


def contact_sort_key(row: dict) -> tuple:
    role = str(row.get("role") or "")
    return (ROLE_ORDER.get(role, 50), str(row.get("example_id") or row.get("question_id") or ""))


def find_predictions_csv(explicit: str, root: Path) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"predictions csv not found: {path}")
        return path
    folders = [
        Path(root) / "reljoin_v6_ovis_outputs" / "eval_herbench_v2lite",
        Path(root) / "outputs_reljoin_v6_ovis" / "eval_herbench_v2lite",
        Path(root) / "drive" / "MyDrive" / "reljoin_v6_ovis_outputs" / "eval_herbench_v2lite",
        Path(root) / "drive" / "MyDrive" / "outputs_reljoin_v6_ovis" / "eval_herbench_v2lite",
        Path(__file__).resolve().parents[2] / "reljoin_v6_ovis_outputs" / "eval_herbench_v2lite",
    ]
    found: list[Path] = []
    seen: set[str] = set()
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in folder.glob("predictions_reljoin_*.csv"):
            if "outputs_frameweave" in str(path):
                continue
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            found.append(path)
    if not found:
        return None
    found.sort(key=lambda path: str(path))
    return found[-1]


def load_prediction_index(path: Path) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            question_id = str(row.get("question_id") or "").strip()
            method = str(row.get("method") or "").strip()
            if not question_id or method not in {"vanilla", "hybrid"}:
                continue
            slot = grouped.setdefault(question_id, {})
            slot[method] = {
                "correct": _as_bool(row.get("is_correct")),
                "parsed": str(row.get("parsed_prediction") or "").strip(),
            }
    ready = {}
    for question_id, slot in grouped.items():
        if "vanilla" in slot and "hybrid" in slot:
            ready[question_id] = slot
    return ready


def build_survey(rows: list[dict], predictions: dict[str, dict] | None) -> dict:
    """Return the full candidate table and the run list. ``predictions`` is optional."""
    candidates = []
    for row in rows:
        if not str(row.get("resolved_video_path") or "").strip():
            continue
        outcome = None if predictions is None else predictions.get(str(row.get("question_id") or ""))
        candidates.append(annotate_row(row, outcome))
    selected, shortfalls = allocate(candidates, has_predictions=predictions is not None)
    selected_ids = {item["question_id"] for item in selected}
    order = {item["question_id"]: index + 1 for index, item in enumerate(selected)}
    for item in candidates:
        if item["question_id"] in selected_ids:
            continue
        item["selected_for_run"] = False
        item["run_order"] = ""
        item["quota"] = ""
        if item["eligible"] and not item["reason"]:
            item["reason"] = "eligible but outside the run quotas"
    for item in selected:
        item["selected_for_run"] = True
        item["run_order"] = order[item["question_id"]]
    table = sorted(candidates, key=lambda item: item["question_id"])
    return {
        "survey_rule": SURVEY_RULE_VERSION,
        "heatmap_used_for_selection": False,
        "frame_reselection": False,
        "paper_figures_chosen": False,
        "has_predictions": predictions is not None,
        "shortfalls": shortfalls,
        "example_ids": [item["question_id"] for item in selected],
        "selected": [_public_selected(item) for item in selected],
        "candidates": table,
    }


def annotate_row(row: dict, outcome: dict | None) -> dict:
    spans, mode = extract_evidence_spans(row.get("metadata") or {})
    task = str(row.get("task_type") or "")
    family = family_of(task)
    n_spans = len(spans)
    eligible = mode == "oracle" and n_spans >= 2
    if not eligible:
        if mode != "oracle":
            reason = f"excluded: {mode or 'missing_evidence'}"
        else:
            reason = f"excluded: {n_spans} evidence span"
    else:
        reason = ""
    vanilla_correct, hybrid_correct, vanilla_parsed = _outcome_flags(outcome)
    choices = _as_list(row.get("choices"))
    if family == "tso" and outcome is not None:
        prior_role = vanilla_tso_tag(vanilla_parsed, choices, str(row.get("answer") or ""))
    elif outcome is not None and vanilla_correct is not None and hybrid_correct is not None and family != "tso":
        prior_role = prior_role_from_flags(vanilla_correct, hybrid_correct)
    else:
        prior_role = ""
    return {
        "question_id": str(row.get("question_id") or ""),
        "video_id": str(row.get("video_id") or Path(str(row.get("resolved_video_path") or "")).stem),
        "task": task,
        "family": family,
        "priority": priority_of(family),
        "n_evidence_spans": n_spans,
        "oracle_mode": mode,
        "span_text": ";".join(f"{start:.2f}-{end:.2f}" for start, end in spans),
        "eligible": eligible,
        "selected_for_run": False,
        "run_order": "",
        "quota": "",
        "prior_role": prior_role,
        "reason": reason,
        "vanilla_correct": "" if vanilla_correct is None else str(bool(vanilla_correct)),
        "hybrid_correct": "" if hybrid_correct is None else str(bool(hybrid_correct)),
        "question": " ".join(str(row.get("question") or "").split())[:240],
        "choices": choices,
        "gold": str(row.get("answer") or ""),
    }


def prior_role_from_flags(vanilla_correct: bool, hybrid_correct: bool) -> str:
    if (not vanilla_correct) and hybrid_correct:
        return "success"
    if vanilla_correct and hybrid_correct:
        return "control"
    if vanilla_correct and (not hybrid_correct):
        return "regression"
    return "still_wrong"


def vanilla_tso_tag(parsed: str, choices: list[str], gold: str) -> str:
    from reljoin.tso_abstention import is_abstention_choice

    letter = _parsed_letter(parsed)
    choice_text = ""
    if letter:
        index = ord(letter) - ord("A")
        if 0 <= index < len(choices):
            choice_text = choices[index]
    if is_abstention_choice(choice_text or str(parsed or "")):
        return "tso_abstain"
    gold_letter = _parsed_letter(gold) or str(gold or "").strip().upper()
    if letter and gold_letter and letter != gold_letter:
        return "tso_wrong_order"
    if letter and gold_letter and letter == gold_letter:
        return "tso_correct"
    return "tso_other"


def allocate(candidates: list[dict], *, has_predictions: bool) -> tuple[list[dict], list[str]]:
    used: set[str] = set()
    shortfalls: list[str] = []
    if not has_predictions:
        main_pool = _pool(candidates, lambda item: item["family"] in PRIORITY_FAMILIES)
        tso_pool = _pool(candidates, lambda item: item["family"] == "tso")
        main = _take_balanced(main_pool, QUOTA_PENDING_MAIN, used)
        tso = _take(tso_pool, QUOTA_TSO, used)
        _mark(main, "pending_prediction", "selected before predictions; role is assigned after collect")
        _mark(tso, "tso", "selected before predictions; TSO slice, role is assigned after collect")
        _note_count(shortfalls, "pending_prediction", len(main), QUOTA_PENDING_MAIN)
        _note_count(shortfalls, "tso", len(tso), QUOTA_TSO)
        return main + tso, shortfalls

    success_pool = _pool(
        candidates,
        lambda item: item["prior_role"] == "success" and item["family"] in PRIORITY_FAMILIES,
    )
    control_pool = _pool(
        candidates,
        lambda item: item["prior_role"] == "control" and item["family"] in PRIORITY_FAMILIES,
    )
    still_pool = _pool(
        candidates,
        lambda item: item["prior_role"] == "still_wrong" and item["family"] in PRIORITY_FAMILIES,
    )
    regression_pool = _pool(
        candidates,
        lambda item: item["prior_role"] == "regression" and item["family"] in PRIORITY_FAMILIES,
    )
    success = _take_balanced(success_pool, QUOTA_SUCCESS, used)
    control = _take_balanced(control_pool, QUOTA_CONTROL, used)
    failure = _take_failure(still_pool, regression_pool, used)
    tso = _take_tso(candidates, used)
    _mark(success, "success", "vanilla incorrect and hybrid correct")
    _mark(control, "control", "vanilla and hybrid both correct")
    _mark(failure, "failure", "hybrid incorrect")
    _mark(tso, "tso", "TSO slice")
    _note_count(shortfalls, "success", len(success), QUOTA_SUCCESS)
    _note_families(shortfalls, "success", success_pool, success)
    _note_count(shortfalls, "control", len(control), QUOTA_CONTROL)
    _note_families(shortfalls, "control", control_pool, control)
    _note_count(shortfalls, "failure", len(failure), QUOTA_FAILURE)
    _note_failure_mix(shortfalls, failure)
    _note_count(shortfalls, "tso", len(tso), QUOTA_TSO)
    return success + control + failure + tso, shortfalls


def write_survey(output_dir: Path, survey: dict, *, predictions_csv: str) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_candidates(output_dir / "survey_candidates.csv", survey["candidates"])
    manifest = {
        "survey_rule": survey["survey_rule"],
        "predictions_csv": predictions_csv,
        "heatmap_used_for_selection": False,
        "frame_reselection": False,
        "paper_figures_chosen": False,
        "has_predictions": survey["has_predictions"],
        "shortfalls": survey["shortfalls"],
        "example_ids": survey["example_ids"],
        "selected": survey["selected"],
        "sort_fields": list(SURVEY_SORT_FIELDS),
    }
    _dump_json(output_dir / "survey_manifest.json", manifest)
    _write_empty_paper_selection(output_dir / "paper_selection.json")
    return manifest


def write_paper_groups(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    manifest = {}
    manifest_path = output_dir / "survey_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {str(item.get("question_id")): item for item in manifest.get("selected") or []}
    rows = []
    for example_dir in sorted(path for path in output_dir.iterdir() if (path / "metadata.json").is_file()):
        meta = json.loads((example_dir / "metadata.json").read_text(encoding="utf-8"))
        predictions_path = example_dir / "predictions.json"
        if not predictions_path.is_file():
            continue
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        question_id = str(meta.get("example_id") or example_dir.name)
        survey_row = by_id.get(question_id, {})
        gold = str(meta.get("gold") or "")
        vanilla = str((predictions.get("vanilla") or {}).get("parsed") or "")
        frameweave = str((predictions.get("frameweave") or {}).get("parsed") or "")
        role = viz_role(
            str(meta.get("task") or ""),
            gold,
            vanilla,
            frameweave,
            _as_list(meta.get("choices")),
        )
        prior = str(meta.get("prior_role") or survey_row.get("prior_role") or "")
        rows.append(
            {
                "question_id": question_id,
                "task": meta.get("task") or survey_row.get("task") or "",
                "role": role,
                "vanilla": vanilla,
                "frameweave": frameweave,
                "gold": gold,
                "n_evidence_spans": meta.get("n_evidence_spans", survey_row.get("n_evidence_spans", "")),
                "survey_reason": meta.get("survey_reason") or survey_row.get("reason") or "",
                "prior_role": prior,
                "role_disagrees_with_prior_eval": str(roles_disagree(prior, role)),
                "mapping_status": meta.get("mapping_status", ""),
                "figure_cross_frame": str(example_dir / "figure_cross_frame.png"),
                "figure_read_update": str(example_dir / "figure_read_update.png"),
                "figure_frame_matrix": str(example_dir / "figure_frame_matrix.png"),
            }
        )
    rows.sort(key=contact_sort_key)
    path = output_dir / "paper_groups.csv"
    fields = [
        "question_id", "task", "role", "vanilla", "frameweave", "gold",
        "n_evidence_spans", "survey_reason", "prior_role", "role_disagrees_with_prior_eval",
        "mapping_status", "figure_cross_frame", "figure_read_update", "figure_frame_matrix",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def viz_role(task: str, gold: str, vanilla: str, frameweave: str, choices: list[str]) -> str:
    if is_tso_task(task):
        tag = vanilla_tso_tag(vanilla, choices, gold)
        if tag == "tso_correct":
            return "tso_correct"
        return tag
    vanilla_ok = str(vanilla).strip().upper() == str(gold).strip().upper() and bool(str(gold).strip())
    frameweave_ok = str(frameweave).strip().upper() == str(gold).strip().upper() and bool(str(gold).strip())
    return prior_role_from_flags(vanilla_ok, frameweave_ok)


def roles_disagree(prior: str, viz: str) -> bool:
    if not str(prior).strip() or not str(viz).strip():
        return False
    return str(prior).strip() != str(viz).strip()


def survey_command(args) -> None:
    from reljoin.herbench_oracle import load_rows
    from run_herbench_oracle_reljoin import row_task_type

    output_dir = Path(args.output_dir)
    _refuse_eval_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.herbench_config != "lite_v2":
        raise RuntimeError("The survey reads HERBench lite_v2.")
    rows = load_rows(
        Path(args.herbench_root),
        Path(args.videos_root),
        args.herbench_revision,
        config="lite_v2",
    )
    for row in rows:
        if not row.get("task_type"):
            row["task_type"] = row_task_type(row)
    csv_path = find_predictions_csv(str(getattr(args, "predictions_csv", "") or ""), Path(args.root))
    predictions = load_prediction_index(csv_path) if csv_path is not None else None
    survey = build_survey(rows, predictions)
    manifest = write_survey(
        output_dir,
        survey,
        predictions_csv="" if csv_path is None else str(csv_path),
    )
    print(
        json.dumps(
            {
                "survey_rule": SURVEY_RULE_VERSION,
                "candidates": len(survey["candidates"]),
                "selected": len(manifest["example_ids"]),
                "predictions_csv": manifest["predictions_csv"],
                "shortfalls": manifest["shortfalls"],
                "heatmap_used_for_selection": False,
                "first_example": (manifest["example_ids"] or [""])[0],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print("SURVEY OK", flush=True)


def _pool(candidates: list[dict], predicate) -> list[dict]:
    pool = [item for item in candidates if item["eligible"] and predicate(item)]
    pool.sort(key=lambda item: (item["priority"], item["question_id"]))
    return pool


def _take(pool: list[dict], count: int, used: set[str]) -> list[dict]:
    picked = []
    for item in pool:
        if len(picked) >= count:
            break
        if item["question_id"] in used:
            continue
        picked.append(item)
        used.add(item["question_id"])
    return _sort_selected(picked)


def _take_balanced(pool: list[dict], count: int, used: set[str]) -> list[dict]:
    families: list[str] = []
    for item in pool:
        if item["family"] not in families:
            families.append(item["family"])
    picked: list[dict] = []
    seen: set[str] = set()
    for family in families:
        if len(picked) >= count:
            break
        for item in pool:
            if item["question_id"] in used or item["question_id"] in seen:
                continue
            if item["family"] != family:
                continue
            picked.append(item)
            seen.add(item["question_id"])
            break
    for item in pool:
        if len(picked) >= count:
            break
        if item["question_id"] in used or item["question_id"] in seen:
            continue
        picked.append(item)
        seen.add(item["question_id"])
    for item in picked:
        used.add(item["question_id"])
    return _sort_selected(picked)


def _take_failure(still_pool: list[dict], regression_pool: list[dict], used: set[str]) -> list[dict]:
    still = _take(still_pool, QUOTA_FAILURE_EACH, used)
    regression = _take(regression_pool, QUOTA_FAILURE_EACH, used)
    picked = still + regression
    if len(picked) < QUOTA_FAILURE:
        extra = _take(still_pool + regression_pool, QUOTA_FAILURE - len(picked), used)
        picked.extend(extra)
    return _sort_selected(picked)


def _take_tso(candidates: list[dict], used: set[str]) -> list[dict]:
    abstain = _pool(candidates, lambda item: item["prior_role"] == "tso_abstain")
    wrong = _pool(candidates, lambda item: item["prior_role"] == "tso_wrong_order")
    picked = []
    if abstain:
        picked.extend(_take(abstain, 1, used))
    if wrong and len(picked) < QUOTA_TSO:
        picked.extend(_take(wrong, 1, used))
    if len(picked) < QUOTA_TSO:
        picked.extend(_take(abstain + wrong, QUOTA_TSO - len(picked), used))
    return _sort_selected(picked)


def _mark(rows: list[dict], quota: str, because: str) -> None:
    for item in rows:
        item["quota"] = quota
        item["reason"] = (
            f"selected for {quota}; {because}; family={item['family']}; "
            f"prior_role={item['prior_role'] or 'pending'}; evidence_spans={item['n_evidence_spans']}"
        )


def _sort_selected(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda item: (item["priority"], item["question_id"]))


def _note_count(shortfalls: list[str], name: str, got: int, target: int) -> None:
    if got < target:
        shortfalls.append(f"{name} {got}/{target}")


def _note_families(shortfalls: list[str], name: str, pool: list[dict], picked: list[dict]) -> None:
    available = {item["family"] for item in pool}
    got = {item["family"] for item in picked}
    if len(available) >= 2 and len(got) < 2:
        raise RuntimeError(f"{name} survey dropped a task family that was available")
    if len(got) < 2:
        shortfalls.append(f"{name} task families: {', '.join(sorted(got)) or 'none'}")


def _note_failure_mix(shortfalls: list[str], picked: list[dict]) -> None:
    roles = {item["prior_role"] for item in picked}
    if picked and "still_wrong" not in roles:
        shortfalls.append("failure has no still-wrong example")
    if picked and "regression" not in roles:
        shortfalls.append("failure has no regression example")


def _public_selected(item: dict) -> dict:
    return {
        "question_id": item["question_id"],
        "task": item["task"],
        "family": item["family"],
        "quota": item["quota"],
        "prior_role": item["prior_role"],
        "n_evidence_spans": item["n_evidence_spans"],
        "oracle_mode": item["oracle_mode"],
        "reason": item["reason"],
    }


def _write_candidates(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["eligible"] = "true" if row["eligible"] else "false"
            payload["selected_for_run"] = "true" if row["selected_for_run"] else "false"
            writer.writerow(payload)


def _write_empty_paper_selection(path: Path) -> None:
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
        for slot in PAPER_SLOTS:
            question_id = str((current.get(slot) or {}).get("question_id") or "").strip()
            if question_id:
                return
    payload = {
        slot: {"question_id": "", "reason": ""}
        for slot in PAPER_SLOTS
    }
    payload["note"] = (
        "Choose after the contact sheet. Code leaves question_id empty and does not rank heatmap sharpness."
    )
    _dump_json(path, payload)


def _dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _outcome_flags(outcome: dict | None) -> tuple[bool | None, bool | None, str]:
    if not outcome:
        return None, None, ""
    vanilla = outcome.get("vanilla") or {}
    hybrid = outcome.get("hybrid") or {}
    if "correct" not in vanilla or "correct" not in hybrid:
        return None, None, str(vanilla.get("parsed") or "")
    return bool(vanilla["correct"]), bool(hybrid["correct"]), str(vanilla.get("parsed") or "")


def _parsed_letter(parsed: str) -> str:
    text = str(parsed or "").strip().upper()
    if text[:1] in "ABCDE":
        return text[:1]
    return ""


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        return [value] if text else []
    return [str(item) for item in list(value)]


def _refuse_eval_dir(path: Path) -> None:
    text = str(path)
    for banned in ("predictions_reljoin", "predictions_residual", "eval_herbench"):
        if banned in text:
            raise RuntimeError(f"Refusing to write visualizations into an evaluation output: {path}")
