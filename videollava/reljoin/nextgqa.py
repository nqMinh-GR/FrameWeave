"""NExT-GQA adapter: CSV/gsub load, A–E mapping, type groups, coverage, metrics.

Evidence spans are used only to pick oracle frames. They never enter the prompt.
Oracle answer accuracy is not Acc@GQA: this pipeline does not predict spans.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

DATASET_ID = "shuzhig/nextgqa"
DATASET_REVISION = "3fc4ace54d7566f46145ec29138e912e557ae870"
NUM_FRAMES = 8
EXPECTED_COUNTS = {"val": 3358, "test": 5553}
EXPECTED_VIDEOS = 1570
FRAME_PROTOCOL_UNIFORM = "nextgqa_uniform_full_video_k8"
FRAME_PROTOCOL_ORACLE = "nextgqa_oracle_evidence_k8"
PARSER_VERSION = "nextgqa_letter_a_e_v1"
AUTHOR_REPO = "https://github.com/doc-doc/NExT-GQA"
AUTHOR_README = "https://github.com/doc-doc/NExT-GQA/blob/main/datasets/nextgqa/README.md"
NEXTQA_EVAL = "https://github.com/doc-doc/NExT-QA/blob/main/eval_mc.py"
LETTERS = "ABCDE"
CHOICE_COLS = ("a0", "a1", "a2", "a3", "a4")
SPLITS = ("val", "test")
MODES = ("oracle", "uniform")
RAW_TYPES = ("CW", "CH", "TN", "TP", "TC", "DC", "DL", "DO")
EVAL_TYPES = ("CW", "CH", "TN", "TC", "DC", "DL", "DO")
TYPE_LABELS = {
    "CW": "Why",
    "CH": "How",
    "TN": "Bef&Aft",
    "TC": "When",
    "DC": "Cnt",
    "DL": "Loc",
    "DO": "Other",
    "C": "Acc_C",
    "T": "Acc_T",
    "D": "Acc_D",
    "TP": "TP",
}
GROUP_PREFIX = {"C": ("CW", "CH"), "T": ("TN", "TC"), "D": ("DC", "DL", "DO")}


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).casefold()


def question_id(video_id: Any, qid: Any) -> str:
    return f"{str(video_id).strip()}_{str(qid).strip()}"


def eval_type(raw_type: Any) -> str:
    t = str(raw_type or "").strip().upper()
    if t == "TP":
        return "TN"
    return t


def map_answer_to_letter(answer: Any, choices: list[str]) -> tuple[str, int, str]:
    """Map gold to a unique A–E letter. Prefer unique text, then letter, then index."""
    letters = LETTERS[: len(choices)]
    gold = str(answer).strip() if answer is not None else ""
    if not gold:
        return "", -1, "unmapped_answer"
    norm_gold = normalize_text(gold)
    hits = [i for i, c in enumerate(choices) if normalize_text(c) == norm_gold and normalize_text(c)]
    if len(hits) == 1:
        i = hits[0]
        return letters[i], i, ""
    if len(hits) > 1:
        return "", -1, "ambiguous_answer"
    if len(gold) == 1 and gold.upper() in letters:
        i = letters.index(gold.upper())
        return letters[i], i, ""
    try:
        idx = int(gold)
    except (TypeError, ValueError):
        idx = None
    if idx is not None and 0 <= idx < len(choices) and str(choices[idx]).strip():
        return letters[idx], idx, ""
    return "", -1, "unmapped_answer"


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_gsub(path: Path) -> dict:
    data = load_json(path)
    if not isinstance(data, dict):
        raise RuntimeError(f"gsub is not an object: {path}")
    return data


def gsub_entry(gsub: dict, video_id: str) -> dict:
    if video_id in gsub:
        return gsub[video_id] or {}
    if video_id.isdigit() and int(video_id) in gsub:
        return gsub[int(video_id)] or {}
    return {}


def gsub_spans(entry: dict, qid: Any) -> list[list[float]]:
    loc = entry.get("location") or {}
    raw = None
    for key in (str(qid), qid):
        if key in loc:
            raw = loc[key]
            break
    if raw is None and str(qid).isdigit():
        ik = int(str(qid))
        if ik in loc:
            raw = loc[ik]
    if not raw:
        return []
    spans: list[list[float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        a, b = float(item[0]), float(item[1])
        if a > b:
            a, b = b, a
        spans.append([a, b])
    spans.sort(key=lambda s: (s[0], s[1]))
    return spans


def annotations_dir(root: Path) -> Path:
    root = Path(root)
    if (root / "annotations").is_dir():
        return root / "annotations"
    return root


def videos_dir(root: Path) -> Path:
    root = Path(root)
    if (root / "videos").is_dir():
        return root / "videos"
    return root


def csv_path(root: Path, split: str) -> Path:
    return annotations_dir(root) / f"{split}.csv"


def gsub_path(root: Path, split: str) -> Path:
    return annotations_dir(root) / f"gsub_{split}.json"


def resolve_video(root: Path, video_id: str) -> Path | None:
    vdir = videos_dir(root)
    for name in (f"{video_id}.mp4", f"{video_id}.MP4"):
        p = vdir / name
        if p.is_file():
            return p
    return None


def _cell(row: dict, key: str) -> str:
    val = row.get(key, "")
    if val is None:
        return ""
    text = str(val).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def load_csv_dicts(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_nextgqa_rows(root: Path, split: str) -> tuple[list[dict], list[dict]]:
    split = str(split).strip().lower()
    if split not in SPLITS:
        raise ValueError(f"split must be val or test, got {split!r}")
    root = Path(root)
    path = csv_path(root, split)
    gpath = gsub_path(root, split)
    if not path.is_file():
        raise FileNotFoundError(path)
    gsub = load_gsub(gpath) if gpath.is_file() else {}
    rows: list[dict] = []
    missing: list[dict] = []
    for rec in load_csv_dicts(path):
        video_id = _cell(rec, "video_id") or _cell(rec, "video")
        qid = _cell(rec, "qid")
        qid_key = question_id(video_id, qid)
        choices = [_cell(rec, c) for c in CHOICE_COLS]
        answer = _cell(rec, "answer")
        letter, idx, map_err = map_answer_to_letter(answer, choices)
        raw_type = _cell(rec, "type")
        entry = gsub_entry(gsub, video_id)
        spans = gsub_spans(entry, qid)
        vpath = resolve_video(root, video_id)
        skip = ""
        if not video_id or not qid:
            skip = "bad_id"
        elif map_err:
            skip = map_err
        elif vpath is None:
            skip = "missing_video"
        duration = float(entry.get("duration") or rec.get("duration") or 0.0 or 0)
        fps = float(entry.get("fps") or 0.0 or 0)
        try:
            frame_count = int(float(_cell(rec, "frame_count") or 0))
        except ValueError:
            frame_count = 0
        row = {
            "question_id": qid_key,
            "video_id": video_id,
            "qid": qid,
            "split": split,
            "question": _cell(rec, "question"),
            "choices": choices,
            "answer": answer,
            "answer_letter": letter,
            "answer_index": idx,
            "raw_type": raw_type,
            "eval_type": eval_type(raw_type),
            "spans": spans,
            "duration": duration,
            "fps": fps,
            "frame_count": frame_count,
            "resolved_video_path": str(vpath) if vpath is not None else "",
            "skip_reason": skip,
            "oracle_skip_reason": skip or ("" if spans else "missing_spans"),
            "width": _cell(rec, "width"),
            "height": _cell(rec, "height"),
        }
        rows.append(row)
        if skip or not spans:
            missing.append({k: row[k] for k in ("question_id", "video_id", "qid", "skip_reason", "oracle_skip_reason")})
    return rows, missing


def type_counts(rows: Iterable[dict], key: str = "raw_type") -> dict[str, int]:
    counts: dict[str, int] = Counter(str(r.get(key) or "") for r in rows)
    return dict(sorted(counts.items()))


def audit_nextgqa(root: Path, split: str | None = None) -> dict:
    root = Path(root)
    splits = [split] if split else list(SPLITS)
    videos = list(videos_dir(root).glob("*.mp4")) + list(videos_dir(root).glob("*.MP4"))
    payload: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "dataset_revision_pin": DATASET_REVISION,
        "root": str(root),
        "n_videos": len(videos),
        "expected_videos": EXPECTED_VIDEOS,
        "author_repo": AUTHOR_REPO,
        "splits": {},
    }
    for sp in splits:
        rows, _ = load_nextgqa_rows(root, sp)
        n_csv = len(rows)
        n_video_ok = sum(1 for r in rows if r["resolved_video_path"])
        n_map_ok = sum(1 for r in rows if r["answer_letter"])
        n_spans = sum(1 for r in rows if r["spans"])
        n_valid_both = sum(
            1 for r in rows if r["resolved_video_path"] and r["answer_letter"] and r["spans"]
        )
        reasons = Counter(r["skip_reason"] or "ok" for r in rows)
        oracle_reasons = Counter(r["oracle_skip_reason"] or "ok" for r in rows)
        payload["splits"][sp] = {
            "expected_qa": EXPECTED_COUNTS.get(sp),
            "n_csv": n_csv,
            "n_with_video": n_video_ok,
            "n_mapped_answer": n_map_ok,
            "n_with_spans": n_spans,
            "n_valid_oracle": n_valid_both,
            "n_valid_uniform": sum(1 for r in rows if r["resolved_video_path"] and r["answer_letter"]),
            "n_missing_video": int(reasons.get("missing_video", 0)),
            "n_unmapped_answer": int(reasons.get("unmapped_answer", 0) + reasons.get("ambiguous_answer", 0)),
            "skip_reasons": dict(reasons),
            "oracle_skip_reasons": dict(oracle_reasons),
            "raw_type_counts": type_counts(rows, "raw_type"),
            "eval_type_counts": type_counts(rows, "eval_type"),
            "csv_path": str(csv_path(root, sp)),
            "gsub_path": str(gsub_path(root, sp)),
            "gsub_exists": gsub_path(root, sp).is_file(),
            "count_matches_card": n_csv == EXPECTED_COUNTS.get(sp),
        }
    return payload


def format_nextgqa_prompt(question: str, choices: list[str]) -> str:
    lines = [str(question).strip(), ""]
    for i, choice in enumerate(choices):
        if str(choice).strip():
            lines.append(f"{LETTERS[i]}. {choice}")
    lines.append("")
    lines.append("Answer with the option letter only.")
    return "\n".join(lines)


def prompt_has_leak(prompt: str, row: dict) -> bool:
    text = prompt.casefold()
    banned = (
        "ground-truth evidence",
        "gsub",
        "oracle span",
        "correct interval",
        "the correct answer is",
        "evidence interval",
    )
    if any(token in text for token in banned):
        return True
    expected = format_nextgqa_prompt(row.get("question") or "", list(row.get("choices") or []))
    return prompt.strip() != expected.strip()


def group_accuracy(records: list[dict], type_key: str = "eval_type") -> dict[str, dict]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for rec in records:
        buckets[str(rec.get(type_key) or "")].append(bool(rec.get("is_correct")))
    out = {}
    for key, vals in buckets.items():
        n = len(vals)
        out[key] = {"n": n, "n_correct": int(sum(vals)), "accuracy": (sum(vals) / n) if n else 0.0}
    return out


def official_groups(records: list[dict]) -> dict[str, dict]:
    """NExT-QA eval_mc.py: TP→TN, then Acc_C / Acc_T / Acc_D and overall Acc."""
    mapped = []
    for rec in records:
        item = dict(rec)
        item["eval_type"] = eval_type(rec.get("eval_type") or rec.get("raw_type") or rec.get("task_type"))
        mapped.append(item)
    by_eval = group_accuracy(mapped, "eval_type")
    out: dict[str, dict] = {}
    for key in EVAL_TYPES:
        out[key] = by_eval.get(key, {"n": 0, "n_correct": 0, "accuracy": 0.0})
        out[key]["label"] = TYPE_LABELS.get(key, key)
    for prefix, members in GROUP_PREFIX.items():
        n = sum(out[m]["n"] for m in members)
        n_correct = sum(out[m]["n_correct"] for m in members)
        out[prefix] = {
            "n": n,
            "n_correct": n_correct,
            "accuracy": (n_correct / n) if n else 0.0,
            "label": TYPE_LABELS[prefix],
            "members": list(members),
        }
    n = len(mapped)
    n_correct = sum(1 for r in mapped if r.get("is_correct"))
    out["Acc"] = {"n": n, "n_correct": n_correct, "accuracy": (n_correct / n) if n else 0.0, "label": "Acc"}
    out["raw_types"] = group_accuracy(records, "raw_type")
    return out


def common_id_delta(vanilla: list[dict], hybrid: list[dict]) -> dict:
    v = {str(r["question_id"]): bool(r.get("is_correct")) for r in vanilla}
    h = {str(r["question_id"]): bool(r.get("is_correct")) for r in hybrid}
    ids = sorted(set(v) & set(h))
    n = len(ids)
    if not n:
        return {"n": 0, "vanilla": 0.0, "hybrid": 0.0, "delta": 0.0}
    va = sum(v[i] for i in ids) / n
    ha = sum(h[i] for i in ids) / n
    return {
        "n": n,
        "vanilla": va,
        "hybrid": ha,
        "delta": ha - va,
        "label": "hybrid_minus_vanilla",
    }


def score_predictions(records: list[dict]) -> dict:
    by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for rec in records:
        key = (str(rec.get("split") or ""), str(rec.get("mode") or ""), str(rec.get("method") or ""))
        by_key[key].append(rec)
    tables = []
    for (split, mode, method), rows in sorted(by_key.items()):
        n = len(rows)
        n_parse_fail = sum(1 for r in rows if not r.get("is_valid"))
        n_error = sum(1 for r in rows if str(r.get("status") or "") != "ok")
        groups = official_groups(rows)
        tables.append(
            {
                "split": split,
                "mode": mode,
                "method": method,
                "n": n,
                "n_correct": int(sum(bool(r.get("is_correct")) for r in rows)),
                "accuracy": (sum(bool(r.get("is_correct")) for r in rows) / n) if n else 0.0,
                "n_parse_fail": n_parse_fail,
                "n_error": n_error,
                "not_acc_gqa": True,
                "groups": groups,
            }
        )
    deltas = []
    modes = {(s, m) for s, m, _ in by_key}
    for split, mode in sorted(modes):
        van = by_key.get((split, mode, "vanilla"), [])
        hyb = by_key.get((split, mode, "hybrid"), [])
        if van and hyb:
            d = common_id_delta(van, hyb)
            d.update({"split": split, "mode": mode})
            deltas.append(d)
    return {"tables": tables, "deltas": deltas, "metric_note": "oracle accuracy is not Acc@GQA"}


def markdown_table(metrics: dict) -> str:
    lines = [
        "# NExT-GQA FrameWeave V6",
        "",
        "Oracle evidence accuracy is **not** Acc@GQA. FrameWeave does not predict evidence spans.",
        "",
        "| split | mode | method | n | acc | Acc_C | Acc_T | Acc_D | parse_fail | error |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in metrics.get("tables") or []:
        g = row.get("groups") or {}
        def _acc(key: str) -> str:
            cell = g.get(key) or {}
            n = cell.get("n") or 0
            a = cell.get("accuracy") or 0.0
            return f"{a:.4f} (n={n})"

        lines.append(
            f"| {row['split']} | {row['mode']} | {row['method']} | {row['n']} | "
            f"{row['accuracy']:.4f} | {_acc('C')} | {_acc('T')} | {_acc('D')} | "
            f"{row['n_parse_fail']} | {row['n_error']} |"
        )
    lines += ["", "## Hybrid − Vanilla on shared IDs", ""]
    for d in metrics.get("deltas") or []:
        lines.append(
            f"- `{d['split']}` `{d['mode']}` n={d['n']}: "
            f"vanilla={d['vanilla']:.4f} hybrid={d['hybrid']:.4f} delta={d['delta']:+.4f}"
        )
    return "\n".join(lines) + "\n"


def csv_table(metrics: dict) -> str:
    header = [
        "split",
        "mode",
        "method",
        "n",
        "n_correct",
        "accuracy",
        "acc_c",
        "n_c",
        "acc_t",
        "n_t",
        "acc_d",
        "n_d",
        "n_parse_fail",
        "n_error",
    ]
    lines = [",".join(header)]
    for row in metrics.get("tables") or []:
        g = row.get("groups") or {}
        def gacc(key: str) -> tuple[str, str]:
            cell = g.get(key) or {}
            return f"{cell.get('accuracy') or 0.0:.6f}", str(cell.get("n") or 0)

        ac, nc = gacc("C")
        at, nt = gacc("T")
        ad, nd = gacc("D")
        lines.append(
            ",".join(
                [
                    str(row.get("split")),
                    str(row.get("mode")),
                    str(row.get("method")),
                    str(row.get("n")),
                    str(row.get("n_correct")),
                    f"{row.get('accuracy') or 0.0:.6f}",
                    ac,
                    nc,
                    at,
                    nt,
                    ad,
                    nd,
                    str(row.get("n_parse_fail")),
                    str(row.get("n_error")),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def fingerprint_payload(
    *,
    split: str,
    mode: str,
    num_frames: int,
    min_pixels: int,
    max_pixels: int,
    model_id: str,
    model_revision: str,
    dataset_revision: str,
    ckpt: str,
    methods: list[str],
    method_version: str,
    backbone: str,
) -> dict:
    protocol = FRAME_PROTOCOL_ORACLE if mode == "oracle" else FRAME_PROTOCOL_UNIFORM
    return {
        "protocol": protocol,
        "split": split,
        "mode": mode,
        "num_frames": int(num_frames),
        "min_pixels": int(min_pixels),
        "max_pixels": int(max_pixels),
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset_id": DATASET_ID,
        "dataset_revision": dataset_revision,
        "reljoin_ckpt": ckpt,
        "methods": list(methods),
        "method": "FrameWeave-Hybrid",
        "method_version": method_version,
        "formulation": "5.1.0",
        "backbone": backbone,
        "parser": PARSER_VERSION,
        "not_acc_gqa": True,
    }


def fingerprint_of(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def resume_completed_keys(path: Path, fingerprint: str) -> set[tuple[str, str, str, str]]:
    """Return (split, mode, question_id, method). Refuse a different fingerprint."""
    done: set[tuple[str, str, str, str]] = set()
    for rec in read_jsonl(path):
        fp = str(rec.get("run_fingerprint") or "")
        if fp and fp != fingerprint:
            raise RuntimeError(
                f"predictions at {path} have a different fingerprint {fp} != {fingerprint}"
            )
        if str(rec.get("status") or "") == "runtime_error":
            continue
        done.add(
            (
                str(rec.get("split") or ""),
                str(rec.get("mode") or ""),
                str(rec.get("question_id") or ""),
                str(rec.get("method") or ""),
            )
        )
    return done


def select_smoke_rows(rows: list[dict]) -> list[dict]:
    usable = [r for r in rows if r.get("resolved_video_path") and r.get("answer_letter")]
    picks: list[dict] = []
    seen: set[str] = set()

    def add(row: dict) -> None:
        qid = row["question_id"]
        if qid in seen:
            return
        seen.add(qid)
        picks.append(row)

    for row in usable:
        if len(row.get("spans") or []) == 1:
            add(row)
            break
    for row in usable:
        if len(row.get("spans") or []) >= 2:
            add(row)
            break
    for row in usable:
        if str(row.get("eval_type") or "") in {"CW", "CH"}:
            add(row)
            break
    for row in usable:
        if str(row.get("eval_type") or "") in {"TN", "TC"} or str(row.get("raw_type") or "") == "TP":
            add(row)
            break
    if len(picks) < 2:
        for row in usable:
            add(row)
            if len(picks) >= 4:
                break
    return picks


def code_commit(repo: Path | None = None) -> str:
    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
