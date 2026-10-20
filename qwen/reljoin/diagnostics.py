"""Answer evaluation helpers for RelJoin (backbone-agnostic)."""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd


LETTER_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)
LETTERS = "ABCDE"


def parse_choice_strict(text: str, n_choices: int = 5) -> tuple[str, bool]:
    """Return (letter or '', is_valid). Ambiguous/missing → invalid."""
    if text is None:
        return "", False
    t = str(text).strip()
    if not t:
        return "", False
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_choices]
    if len(t) == 1 and t.upper() in letters:
        return t.upper(), True
    found = LETTER_RE.findall(t)
    uniq = []
    for f in found:
        u = f.upper()
        if u in letters and u not in uniq:
            uniq.append(u)
    if len(uniq) == 1:
        return uniq[0], True
    return "", False


def accuracy_reports(rows: list[dict]) -> dict:
    """rows need: is_correct, is_valid."""
    n = len(rows)
    n_valid = sum(1 for r in rows if r.get("is_valid"))
    n_correct = sum(1 for r in rows if r.get("is_correct"))
    n_correct_valid = sum(1 for r in rows if r.get("is_correct") and r.get("is_valid"))
    return {
        "n": n,
        "strict_accuracy": n_correct / n if n else 0.0,
        "valid_accuracy": n_correct_valid / n_valid if n_valid else 0.0,
        "coverage": n_valid / n if n else 0.0,
    }


def answer_position_bias(rows: list[dict]) -> dict:
    pred = Counter()
    gt = Counter()
    for r in rows:
        p = str(r.get("parsed_prediction") or "")
        g = str(r.get("ground_truth") or "")
        if p:
            pred[p] += 1
        if g:
            gt[g] += 1
    return {"predicted": dict(pred), "ground_truth": dict(gt)}


def group_accuracy(df: pd.DataFrame, key: str) -> dict:
    out = {}
    if key not in df.columns:
        return out
    for val, g in df.groupby(key):
        n = len(g)
        out[str(val)] = {
            "n": int(n),
            "strict_accuracy": float(g["is_correct"].mean()) if n else 0.0,
        }
    return out
