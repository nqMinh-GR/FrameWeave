"""Macro HERBench tables. Non-TSO is an unweighted diagnostic, not a micro score."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .common import ensure_v6_on_path, write_json
from .config import TASKS, TASK_FAMILIES, TSO_TASK

ensure_v6_on_path()

from reljoin.stats import mcnemar_exact, win_tie_loss  # noqa: E402

PAIRED_SEED = 42
PAIRED_RESAMPLES = 20_000


def _task_accuracy(rows: list[dict], task: str) -> dict | None:
    group = [row for row in rows if str(row.get("task")) == task and row.get("status") == "ok"]
    if not group:
        return None
    correct = sum(1 for row in group if bool(row.get("correct")))
    return {
        "correct": correct,
        "n": len(group),
        "accuracy": correct / len(group),
        "videos": len({str(row.get("video_id")) for row in group}),
    }


def _macro_scores(per_task: dict[str, dict | None]) -> tuple[float | None, float | None]:
    present = [stats["accuracy"] for stats in per_task.values() if stats is not None]
    non_tso = [
        per_task[task]["accuracy"]
        for task in TASKS
        if task != TSO_TASK and per_task[task] is not None
    ]
    overall = (sum(present) / len(TASKS)) if len(present) == len(TASKS) else None
    non_tso_avg = (sum(non_tso) / len(non_tso)) if len(non_tso) == len(TASKS) - 1 else None
    return overall, non_tso_avg


def summarize_predictions(
    rows: list[dict],
    *,
    variant: str,
    seed: int,
    expected_questions: int | None = None,
    checkpoint_id: str = "",
    selected_epoch: int | None = None,
    trainable_params: int | None = None,
) -> dict:
    chosen = [row for row in rows if str(row.get("variant")) == variant]
    ok = [row for row in chosen if row.get("status") == "ok"]
    correct = sum(1 for row in ok if bool(row.get("correct")))
    invalid = sum(1 for row in ok if not bool(row.get("valid")))
    per_task = {task: _task_accuracy(ok, task) for task in TASKS}
    overall_avg, non_tso_avg = _macro_scores(per_task)
    families = {}
    for name, members in TASK_FAMILIES.items():
        scores = [per_task[task]["accuracy"] for task in members if per_task[task] is not None]
        families[name] = sum(scores) / len(scores) if len(scores) == len(members) else None
    n_expected = int(expected_questions or 0)
    coverage = (len(ok) / n_expected) if n_expected else (len(ok) / len(chosen) if chosen else 0.0)
    complete = bool(ok) and (n_expected == 0 or len(ok) == n_expected) and all(stats is not None for stats in per_task.values())
    return {
        "variant": variant,
        "seed": seed,
        "checkpoint_id": checkpoint_id,
        "selected_epoch": selected_epoch,
        "trainable_params": trainable_params,
        "n_rows": len(chosen),
        "n_ok": len(ok),
        "n_correct": correct,
        "n_invalid": invalid,
        "n_failed": len(chosen) - len(ok),
        "expected_questions": n_expected or None,
        "coverage": coverage,
        "n_videos": len({str(row.get("video_id")) for row in ok}),
        "qa_accuracy": (correct / len(ok)) if ok else None,
        "overall_avg": overall_avg,
        "tso": None if per_task[TSO_TASK] is None else per_task[TSO_TASK]["accuracy"],
        "non_tso_avg": non_tso_avg,
        "non_tso_definition": "unweighted mean of the 11 non-TSO task accuracies; diagnostic, not micro accuracy",
        "families": families,
        "per_task": per_task,
        "complete": complete,
    }


def _pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{100.0 * value:.2f}"


def _table_row(summary: dict) -> dict[str, str]:
    families = summary["families"]
    return {
        "Variant": summary["variant"],
        "Trainable Params": "" if summary["trainable_params"] is None else str(summary["trainable_params"]),
        "Selected Epoch": "" if summary["selected_epoch"] is None else str(summary["selected_epoch"]),
        "Overall Avg.": _pct(summary["overall_avg"]),
        "QA Acc.": _pct(summary["qa_accuracy"]),
        "TSO": _pct(summary["tso"]),
        "Non-TSO Avg.": _pct(summary["non_tso_avg"]),
        "TR&C": _pct(families.get("TR&C")),
        "R&T": _pct(families.get("R&T")),
        "GC&V": _pct(families.get("GC&V")),
        "ME&N": _pct(families.get("ME&N")),
        "Coverage": _pct(summary["coverage"]),
        "Questions": str(summary["n_ok"]),
        "Videos": str(summary["n_videos"]),
        "Complete": "yes" if summary["complete"] else "no",
    }


def _per_task_columns(summary: dict) -> dict[str, str]:
    rendered = {}
    for task, stats in summary["per_task"].items():
        rendered[task] = "" if stats is None else _pct(stats["accuracy"])
    return rendered


def write_tables(summaries: list[dict], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = list(_table_row(summaries[0]).keys()) if summaries else []
    task_columns = list(TASKS)
    csv_path = out_dir / "results.csv"
    md_path = out_dir / "results.md"
    tex_path = out_dir / "results.tex"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns + task_columns)
        writer.writeheader()
        for summary in summaries:
            row = _table_row(summary)
            row.update(_per_task_columns(summary))
            writer.writerow(row)
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [
        header,
        rule,
    ]
    for summary in summaries:
        rendered = _table_row(summary)
        lines.append("| " + " | ".join(rendered[column] for column in columns) + " |")
    lines.extend(
        [
            "",
            "Non-TSO Avg. is the unweighted mean of the 11 non-TSO task accuracies.",
            "It is a diagnostic, not the micro accuracy after dropping TSO.",
            "Overall Avg. is the unweighted mean of all 12 tasks. QA Acc. is micro accuracy.",
            "Invalid generations stay in the QA denominator. A row with Complete=no is not a full lite_v2 result.",
            "Paper reference numbers are not tuning targets: Vanilla Overall 61.19 / QA 56.82; Hybrid Overall 66.54 / QA 60.98.",
            "",
            "Per-task accuracy:",
            "",
            "| Variant | " + " | ".join(task_columns) + " |",
            "| " + " | ".join(["---", *("---" for _ in task_columns)]) + " |",
        ]
    )
    for summary in summaries:
        tasks = _per_task_columns(summary)
        lines.append("| " + summary["variant"] + " | " + " | ".join(tasks[task] for task in task_columns) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tex_columns = [
        "Variant",
        "Trainable Params",
        "Selected Epoch",
        "Overall Avg.",
        "QA Acc.",
        "TSO",
        "Non-TSO Avg.",
        "TR&C",
        "R&T",
        "GC&V",
        "ME&N",
    ]
    tex = [
        "\\begin{tabular}{lrrrrrrrrrr}",
        " & ".join(column.replace("&", "\\&") for column in tex_columns) + " \\\\",
        "\\hline",
    ]
    for summary in summaries:
        rendered = _table_row(summary)
        tex.append(" & ".join(rendered[column].replace("%", "\\%") for column in tex_columns) + " \\\\")
    tex.append("\\end{tabular}")
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    write_json(out_dir / "metrics.json", {"variants": summaries})
    return {"csv": str(csv_path), "markdown": str(md_path), "latex": str(tex_path)}


def _ok_rows(rows: list[dict], variant: str) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for row in rows:
        if str(row.get("variant")) == variant and row.get("status") == "ok":
            found.setdefault(str(row["question_id"]), row)
    return found


def video_cluster_delta(
    correct_reference: list[bool],
    correct_variant: list[bool],
    video_ids: list[str],
    *,
    n_resamples: int = PAIRED_RESAMPLES,
    seed: int = PAIRED_SEED,
) -> dict:
    """Accuracy delta (variant - reference). Each draw resamples whole videos for both arms."""
    diff = np.asarray(correct_variant, dtype=np.float64) - np.asarray(correct_reference, dtype=np.float64)
    order = sorted(set(video_ids))
    position = {video: index for index, video in enumerate(order)}
    cluster = np.asarray([position[video] for video in video_ids], dtype=np.int64)
    sums = np.bincount(cluster, weights=diff, minlength=len(order))
    counts = np.bincount(cluster, minlength=len(order)).astype(np.float64)
    picks = np.random.default_rng(int(seed)).integers(0, len(order), size=(int(n_resamples), len(order)))
    draws = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {
        "delta": float(diff.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_videos": len(order),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def paired_comparison(
    rows: list[dict],
    *,
    variant: str,
    reference: str,
    n_resamples: int = PAIRED_RESAMPLES,
    seed: int = PAIRED_SEED,
) -> dict:
    """Pair two variants by question_id on questions that are status=ok for both."""
    reference_rows = _ok_rows(rows, reference)
    variant_rows = _ok_rows(rows, variant)
    shared = sorted(set(reference_rows) & set(variant_rows))
    for question_id in shared:
        for key in ("task", "video_id", "gold_letter"):
            if str(reference_rows[question_id].get(key)) != str(variant_rows[question_id].get(key)):
                raise ValueError(
                    f"{question_id}: {reference} and {variant} disagree on {key}. "
                    "Their predictions do not come from the same eval manifest."
                )
    subsets = {
        "QA": shared,
        "TSO": [qid for qid in shared if str(variant_rows[qid].get("task")) == TSO_TASK],
        "Non-TSO questions": [qid for qid in shared if str(variant_rows[qid].get("task")) != TSO_TASK],
    }
    tests: dict[str, dict | None] = {}
    for name, ids in subsets.items():
        if not ids:
            tests[name] = None
            continue
        ref = [bool(reference_rows[qid].get("correct")) for qid in ids]
        var = [bool(variant_rows[qid].get("correct")) for qid in ids]
        outcome = win_tie_loss(ref, var)
        tests[name] = {
            "n": len(ids),
            "reference_acc": sum(ref) / len(ids),
            "variant_acc": sum(var) / len(ids),
            "variant_only_correct": outcome["wins_b"],
            "reference_only_correct": outcome["wins_a"],
            "ties": outcome["ties"],
            "mcnemar_p": mcnemar_exact(ref, var)["p_value"],
            "bootstrap": video_cluster_delta(
                ref,
                var,
                [str(variant_rows[qid].get("video_id")) for qid in ids],
                n_resamples=n_resamples,
                seed=seed,
            ),
        }
    ref_tasks = {task: _task_accuracy([reference_rows[qid] for qid in shared], task) for task in TASKS}
    var_tasks = {task: _task_accuracy([variant_rows[qid] for qid in shared], task) for task in TASKS}
    macro = {}
    for key, ref_score, var_score in zip(
        ("overall_avg", "non_tso_avg"), _macro_scores(ref_tasks), _macro_scores(var_tasks)
    ):
        macro[key] = {
            "reference": ref_score,
            "variant": var_score,
            "delta": None if ref_score is None or var_score is None else var_score - ref_score,
        }
    per_task = {}
    for task in TASKS:
        ref_stats, var_stats = ref_tasks[task], var_tasks[task]
        per_task[task] = None if ref_stats is None else {
            "n": ref_stats["n"],
            "reference": ref_stats["accuracy"],
            "variant": var_stats["accuracy"],
            "delta": var_stats["accuracy"] - ref_stats["accuracy"],
        }
    return {
        "variant": variant,
        "reference": reference,
        "delta_definition": f"{variant} minus {reference}",
        "n_reference_ok": len(reference_rows),
        "n_variant_ok": len(variant_rows),
        "n_paired": len(shared),
        "checkpoint_ids": {
            reference: sorted({str(row.get("checkpoint_id", "")) for row in reference_rows.values()}),
            variant: sorted({str(row.get("checkpoint_id", "")) for row in variant_rows.values()}),
        },
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "macro": macro,
        "tests": tests,
        "per_task": per_task,
    }


def _signed(value: float | None) -> str:
    if value is None:
        return ""
    return f"{100.0 * value:+.2f}"


def _p_value(value: float) -> str:
    return "<1e-4" if value < 1e-4 else f"{value:.4f}"


def _paired_markdown(report: dict) -> list[str]:
    variant, reference = report["variant"], report["reference"]
    tests, macro = report["tests"], report["macro"]
    checkpoints = report["checkpoint_ids"]
    lines = [
        f"## {variant} vs {reference}",
        "",
        f"Paired questions: {report['n_paired']} ({variant} ok {report['n_variant_ok']}, "
        f"{reference} ok {report['n_reference_ok']}). Delta = {variant} - {reference}, in points.",
        f"Checkpoints: {reference} {', '.join(checkpoints[reference]) or '-'}; "
        f"{variant} {', '.join(checkpoints[variant]) or '-'}.",
        "",
        f"| Metric | {reference} | {variant} | Delta | 95% CI | McNemar p "
        f"| {variant} only correct | {reference} only correct | n |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def macro_row(label: str, key: str, n: int) -> None:
        item = macro[key]
        lines.append(
            f"| {label} | {_pct(item['reference'])} | {_pct(item['variant'])} | {_signed(item['delta'])} "
            f"|  |  |  |  | {n} |"
        )

    def test_row(label: str, key: str) -> None:
        item = tests[key]
        if item is None:
            lines.append(f"| {label} |  |  |  |  |  |  |  | 0 |")
            return
        boot = item["bootstrap"]
        lines.append(
            f"| {label} | {_pct(item['reference_acc'])} | {_pct(item['variant_acc'])} | {_signed(boot['delta'])} "
            f"| [{_signed(boot['ci_low'])}, {_signed(boot['ci_high'])}] | {_p_value(item['mcnemar_p'])} "
            f"| {item['variant_only_correct']} | {item['reference_only_correct']} | {item['n']} |"
        )

    non_tso = tests["Non-TSO questions"]
    macro_row("Overall Avg. (12-task macro)", "overall_avg", report["n_paired"])
    test_row("QA Acc.", "QA")
    test_row("TSO", "TSO")
    macro_row("Non-TSO Avg. (11-task macro)", "non_tso_avg", non_tso["n"] if non_tso else 0)
    test_row("Non-TSO questions (micro)", "Non-TSO questions")
    lines.extend(
        [
            "",
            "Per-task accuracy on paired questions:",
            "",
            f"| Task | n | {reference} | {variant} | Delta |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for task in TASKS:
        item = report["per_task"][task]
        if item is None:
            lines.append(f"| {task} | 0 |  |  |  |")
        else:
            lines.append(
                f"| {task} | {item['n']} | {_pct(item['reference'])} | {_pct(item['variant'])} "
                f"| {_signed(item['delta'])} |"
            )
    lines.append("")
    return lines


def write_paired(reports: list[dict], out_dir: Path) -> dict[str, str]:
    if not reports:
        raise ValueError("write_paired needs at least one comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Paired HERBench lite_v2 comparison", ""]
    for report in reports:
        lines.extend(_paired_markdown(report))
    lines.extend(
        [
            "Overall Avg. and Non-TSO Avg. are task macros. Only their point delta is reported.",
            f"95% CI: video-cluster bootstrap, {reports[0]['n_resamples']} draws, seed {reports[0]['seed']}. "
            "Each draw resamples whole videos and applies them to both variants.",
            "McNemar: exact two-sided binomial test on discordant questions.",
            "Only questions with status=ok for both variants are paired. Invalid generations count as wrong.",
        ]
    )
    md_path = out_dir / "paired.md"
    json_path = out_dir / "paired.json"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(json_path, {"comparisons": reports})
    return {"paired_markdown": str(md_path), "paired_json": str(json_path)}
