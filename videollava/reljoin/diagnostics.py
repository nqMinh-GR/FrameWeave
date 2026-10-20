"""Evidence-joining diagnostics and answer evaluation helpers."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


LETTER_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)
LETTERS = "ABCDE"


def parse_choice_strict(text: str, n_choices: int = 5) -> tuple[str, bool]:
    """Return (letter or '', is_valid). Ambiguous/missing → invalid."""
    if text is None:
        return "", False
    t = str(text).strip()
    if not t:
        return "", False
    # Prefer HERBench-style if available is handled by caller; here local strict
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_choices]
    # Exact single letter
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


@torch.no_grad()
def option_loglikelihoods(
    model,
    tokenizer,
    video_tensor,
    question: str,
    choices: list[str],
    num_frames: int,
    set_context_fn,
    dtype,
) -> dict[str, float]:
    """Length-normalized log p(letter | V,q,O). Not option-content scoring."""
    from videollava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, IGNORE_INDEX
    from videollava.conversation import conv_templates
    from videollava.mm_utils import tokenizer_image_token

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    scores = {}
    for i, _c in enumerate(choices):
        if not str(_c).strip():
            continue
        letter = letters[i]
        prompt_lines = [question.strip(), ""]
        for j, ch in enumerate(choices):
            if str(ch).strip():
                prompt_lines.append(f"{letters[j]}. {ch}")
        prompt_lines += ["", "Answer with the option letter only."]
        prompt = "\n".join(prompt_lines)

        conv = conv_templates["llava_v1"].copy()
        instruction = "".join([DEFAULT_IMAGE_TOKEN] * num_frames) + "\n" + prompt
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], letter)
        full = conv.get_prompt()
        input_ids = tokenizer_image_token(
            full, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        conv_q = conv_templates["llava_v1"].copy()
        conv_q.append_message(conv_q.roles[0], instruction)
        conv_q.append_message(conv_q.roles[1], None)
        prompt_ids = tokenizer_image_token(
            conv_q.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()
        labels = input_ids.clone()
        labels[:, : prompt_ids.shape[1]] = IGNORE_INDEX

        set_context_fn()
        out = model(input_ids=input_ids, labels=labels, images=video_tensor, use_cache=False)
        shift_logits = out.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        m = shift_labels[0] != IGNORE_INDEX
        if m.sum() == 0:
            scores[letter] = float("-inf")
            continue
        tok = log_probs[0, torch.arange(shift_labels.shape[1], device=labels.device)[m], shift_labels[0, m]]
        scores[letter] = float(tok.mean().cpu())
    return scores


def joining_gain_from_scores(s_full: float, s_singles: list[float]) -> dict:
    s_single = max(s_singles) if s_singles else float("nan")
    return {
        "s_full": s_full,
        "s_single": s_single,
        "join_gain": s_full - s_single if s_singles else float("nan"),
    }


def loo_degradation(s_full: float, s_loo: list[float]) -> dict:
    if not s_loo:
        return {"loo_mean": float("nan"), "loo_degradation": float("nan"), "worst_degradation": float("nan")}
    mean_loo = float(np.mean(s_loo))
    worst = float(min(s_loo))
    return {
        "loo_mean": mean_loo,
        "loo_degradation": s_full - mean_loo,
        "worst_degradation": s_full - worst,
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


def nonempty_choices(choices: list[str]) -> list[str]:
    return [str(c).strip() for c in choices if str(c).strip()]


def _token_ids_for_text(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(x) for x in ids]


def letter_token_ids(tokenizer, n_choices: int = 5) -> dict[str, int]:
    """Single-token ids for A–E (no leading space). Matches teacher-forced letters."""
    ids: dict[str, int] = {}
    for letter in LETTERS[:n_choices]:
        toks = _token_ids_for_text(tokenizer, letter)
        if len(toks) != 1:
            raise ValueError(
                f"Letter {letter!r} tokenized to {toks} (len={len(toks)}); expected 1 token"
            )
        ids[letter] = toks[0]
    return ids


def cyclic_rotations(choices: list[str]) -> list[tuple[int, list[str]]]:
    """Left rotations: ABCDE, BCDEA, CDEAB, DEABC, EABCD."""
    n = len(choices)
    return [(k, choices[k:] + choices[:k]) for k in range(n)]


def letter_to_content(pred_letter: str, rotated_choices: list[str]) -> str:
    pred = str(pred_letter or "").strip().upper()
    if pred not in LETTERS:
        return ""
    idx = LETTERS.index(pred)
    if idx >= len(rotated_choices):
        return ""
    return str(rotated_choices[idx]).strip()


def content_consistency(chosen_contents: list[str], n_rotations: Optional[int] = None) -> float:
    """max_c #{rotations choosing content c} / n_rotations (default 5 or len)."""
    n = n_rotations if n_rotations is not None else len(chosen_contents)
    if n <= 0:
        return float("nan")
    counts = Counter(c for c in chosen_contents if c)
    if not counts:
        return 0.0
    return max(counts.values()) / float(n)


def first_position_rate(pred_letters: list[str]) -> float:
    """Fraction of runs whose predicted letter is the first presented option (A)."""
    n = len(pred_letters)
    if n <= 0:
        return float("nan")
    return sum(1 for p in pred_letters if str(p or "").strip().upper() == "A") / float(n)


def letter_logit_stats(
    next_logits: torch.Tensor,
    token_ids: dict[str, int],
    gt_letter: str,
    n_choices: int = 5,
) -> dict:
    """Softmax over A–E logits at the answer position; rank / margin / entropy."""
    letters = LETTERS[:n_choices]
    z = next_logits.detach().float().reshape(-1)
    zs = torch.stack([z[token_ids[L]] for L in letters])
    p = torch.softmax(zs, dim=0)
    entropy = float((-(p * p.clamp_min(1e-12).log())).sum())
    order = torch.argsort(zs, descending=True)
    ranked = [letters[int(i)] for i in order.tolist()]
    gt = str(gt_letter or "").strip().upper()
    if gt in ranked:
        gt_rank = ranked.index(gt) + 1
        zy = float(zs[letters.index(gt)])
    else:
        gt_rank = n_choices + 1
        zy = float("nan")
    z_a = float(zs[0])
    probs = {f"p_{L}": float(p[i]) for i, L in enumerate(letters)}
    zvals = {f"z_{L}": float(zs[i]) for i, L in enumerate(letters)}
    return {
        **probs,
        **zvals,
        "entropy": entropy,
        "margin_zA_minus_zy": (z_a - zy) if math.isfinite(zy) else float("nan"),
        "gt_rank": int(gt_rank),
        "gt_top1": bool(gt_rank == 1),
        "gt_top2": bool(gt_rank <= 2),
        "argmax_letter": letters[int(zs.argmax())],
    }


def _llava_prompt_ids(tokenizer, prompt: str, num_frames: int, continuation: Optional[str] = None):
    from videollava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from videollava.conversation import conv_templates
    from videollava.mm_utils import tokenizer_image_token

    conv = conv_templates["llava_v1"].copy()
    instruction = "".join([DEFAULT_IMAGE_TOKEN] * num_frames) + "\n" + prompt
    conv.append_message(conv.roles[0], instruction)
    conv.append_message(conv.roles[1], continuation)
    full = conv.get_prompt()
    return tokenizer_image_token(
        full, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()


@torch.no_grad()
def next_token_logits(
    model,
    tokenizer,
    video_tensor,
    prompt: str,
    num_frames: int,
    set_context_fn: Callable[[], None],
) -> torch.Tensor:
    """Answer-position logits: last prompt token predicts the first assistant token."""
    input_ids = _llava_prompt_ids(tokenizer, prompt, num_frames, continuation=None)
    set_context_fn()
    out = model(input_ids=input_ids, images=video_tensor, use_cache=False)
    if out.logits.dim() != 3 or out.logits.size(1) < 1:
        raise RuntimeError(f"Unexpected logits shape {tuple(out.logits.shape)}")
    return out.logits[0, -1]


@torch.no_grad()
def continuation_mean_logprob(
    model,
    tokenizer,
    video_tensor,
    prompt: str,
    continuation: str,
    num_frames: int,
    set_context_fn: Callable[[], None],
) -> float:
    """Length-normalized mean log p(continuation tokens | V, prompt)."""
    from videollava.constants import IGNORE_INDEX

    text = str(continuation or "").strip()
    if not text:
        return float("-inf")
    input_ids = _llava_prompt_ids(tokenizer, prompt, num_frames, continuation=text)
    prompt_ids = _llava_prompt_ids(tokenizer, prompt, num_frames, continuation=None)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = IGNORE_INDEX
    set_context_fn()
    out = model(input_ids=input_ids, labels=labels, images=video_tensor, use_cache=False)
    shift_logits = out.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    m = shift_labels[0] != IGNORE_INDEX
    if m.sum() == 0:
        return float("-inf")
    tok = log_probs[0, torch.arange(shift_labels.shape[1], device=labels.device)[m], shift_labels[0, m]]
    return float(tok.mean().cpu())


@torch.no_grad()
def option_content_loglikelihoods(
    model,
    tokenizer,
    video_tensor,
    prompt: str,
    choices: list[str],
    num_frames: int,
    set_context_fn: Callable[[], None],
) -> dict[str, float]:
    """s_c = (1/|o_c|) log p(option content | V, q, all options). Not letter NLL."""
    scores: dict[str, float] = {}
    for i, choice in enumerate(choices):
        if i >= len(LETTERS):
            break
        text = str(choice).strip()
        if not text:
            continue
        scores[LETTERS[i]] = continuation_mean_logprob(
            model,
            tokenizer,
            video_tensor,
            prompt,
            text,
            num_frames,
            set_context_fn,
        )
    return scores


def content_pred_letter(scores: dict[str, float]) -> str:
    valid = {k: v for k, v in scores.items() if v is not None and math.isfinite(v)}
    if not valid:
        return ""
    return max(valid.items(), key=lambda kv: kv[1])[0]


def _as_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return False
    return str(val).strip().lower() in {"1", "true", "yes"}


def _as_float(val, default: float = float("nan")) -> float:
    try:
        if val is None or val == "":
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def paired_letter_flips(
    df: pd.DataFrame,
    base_method: str = "vanilla",
    other_method: str = "reljoin",
    letter_col: str = "generated_letter",
    id_key: str = "question_id",
) -> dict:
    """Count A-absorbing vs A-escaping flips (paired on question_id)."""
    if letter_col not in df.columns or "method" not in df.columns:
        return {"n_paired": 0, "to_A": 0, "from_A": 0}
    base = df[df["method"] == base_method].set_index(id_key)[letter_col]
    other = df[df["method"] == other_method].set_index(id_key)[letter_col]
    common = base.index.intersection(other.index)
    to_a = 0
    from_a = 0
    for qid in common:
        b = str(base.loc[qid] or "").strip().upper()
        o = str(other.loc[qid] or "").strip().upper()
        if len(b) != 1 or len(o) != 1:
            continue
        if b in "BCDE" and o == "A":
            to_a += 1
        elif b == "A" and o in "BCDE":
            from_a += 1
    return {"n_paired": int(len(common)), "to_A": int(to_a), "from_A": int(from_a)}


def summarize_decoder_diagnostics(df: pd.DataFrame, methods: Optional[list[str]] = None) -> dict:
    """Per-method aggregates for letter logits, content scores, and permutation."""
    if df is None or df.empty:
        return {}
    work = df.copy()
    if "status" in work.columns:
        work = work[work["status"].astype(str) == "ok"]
    if methods is None:
        methods = [str(m) for m in work["method"].dropna().unique().tolist()] if "method" in work.columns else []
    out: dict = {}
    for method in methods:
        sub = work[work["method"] == method] if "method" in work.columns else work
        n = len(sub)
        if n == 0:
            out[method] = {"n": 0}
            continue
        gt = sub["ground_truth"].astype(str).str.upper() if "ground_truth" in sub.columns else pd.Series([""] * n)
        gen_col = sub["generated_letter"].astype(str).str.upper() if "generated_letter" in sub.columns else pd.Series([""] * n)
        logit_col = (
            sub["logit_argmax_letter"].astype(str).str.upper()
            if "logit_argmax_letter" in sub.columns
            else pd.Series([""] * n)
        )
        content_col = (
            sub["content_pred"].astype(str).str.upper() if "content_pred" in sub.columns else pd.Series([""] * n)
        )
        gen_ok = gen_col == gt
        logit_ok = logit_col == gt
        content_ok = content_col == gt
        p_a = sub["p_A"].map(_as_float) if "p_A" in sub.columns else pd.Series([float("nan")] * n)
        ent = sub["entropy"].map(_as_float) if "entropy" in sub.columns else pd.Series([float("nan")] * n)
        rank = sub["gt_rank"].map(_as_float) if "gt_rank" in sub.columns else pd.Series([float("nan")] * n)
        margin = (
            sub["margin_zA_minus_zy"].map(_as_float)
            if "margin_zA_minus_zy" in sub.columns
            else pd.Series([float("nan")] * n)
        )
        top1 = sub["gt_top1"].map(_as_bool) if "gt_top1" in sub.columns else pd.Series([False] * n)
        top2 = sub["gt_top2"].map(_as_bool) if "gt_top2" in sub.columns else pd.Series([False] * n)
        cons = (
            sub["content_consistency"].map(_as_float)
            if "content_consistency" in sub.columns
            else pd.Series([float("nan")] * n)
        )
        fpr = (
            sub["first_position_rate"].map(_as_float)
            if "first_position_rate" in sub.columns
            else pd.Series([float("nan")] * n)
        )
        out[method] = {
            "n": int(n),
            "mean_p_A": float(p_a.mean(skipna=True)) if n else float("nan"),
            "mean_entropy": float(ent.mean(skipna=True)) if n else float("nan"),
            "mean_gt_rank": float(rank.mean(skipna=True)) if n else float("nan"),
            "mean_margin_zA_minus_zy": float(margin.mean(skipna=True)) if n else float("nan"),
            "gt_top1_rate": float(top1.mean()) if n else float("nan"),
            "gt_top2_rate": float(top2.mean()) if n else float("nan"),
            "letter_acc": float(gen_ok.mean()) if n else float("nan"),
            "logit_acc": float(logit_ok.mean()) if n else float("nan"),
            "content_acc": float(content_ok.mean()) if n else float("nan"),
            "mean_content_consistency": float(cons.mean(skipna=True)) if n else float("nan"),
            "mean_first_position_rate": float(fpr.mean(skipna=True)) if n else float("nan"),
        }
    flips = {}
    if "vanilla" in methods:
        for other in methods:
            if other == "vanilla":
                continue
            flips[f"vanilla_vs_{other}"] = paired_letter_flips(work, "vanilla", other)
    out["_flips"] = flips
    out["_verdicts"] = decoder_verdicts(out)
    return out


def decoder_verdicts(summary: dict) -> list[dict]:
    """Five-row interpretation matrix from input.md (representation vs letter bias)."""
    v = summary.get("vanilla") or {}
    d = summary.get("dtr") or {}
    r = summary.get("reljoin") or {}
    nr = summary.get("reljoin_norelation") or {}

    def _f(block, key):
        return _as_float((block or {}).get(key))

    rows = []
    if r and v:
        letter_down = _f(r, "letter_acc") < _f(v, "letter_acc")
        content_up = _f(r, "content_acc") > _f(v, "content_acc")
        content_down = _f(r, "content_acc") < _f(v, "content_acc")
        rows.append(
            {
                "claim": "RelJoin letter acc down, content score up",
                "holds": bool(letter_down and content_up),
                "conclusion": "Representation useful; frozen decoder has position bias",
                "reljoin_letter_acc": _f(r, "letter_acc"),
                "vanilla_letter_acc": _f(v, "letter_acc"),
                "reljoin_content_acc": _f(r, "content_acc"),
                "vanilla_content_acc": _f(v, "content_acc"),
            }
        )
        rows.append(
            {
                "claim": "RelJoin down on both letter and content",
                "holds": bool(letter_down and content_down),
                "conclusion": "Relation block is actually harmful",
                "reljoin_letter_acc": _f(r, "letter_acc"),
                "reljoin_content_acc": _f(r, "content_acc"),
            }
        )
    if d and v:
        content_up = _f(d, "content_acc") > _f(v, "content_acc")
        p_a_down = _f(d, "mean_p_A") < _f(v, "mean_p_A")
        rows.append(
            {
                "claim": "DTR raises content score",
                "holds": bool(content_up),
                "conclusion": "DTR improves evidence access",
                "dtr_content_acc": _f(d, "content_acc"),
                "vanilla_content_acc": _f(v, "content_acc"),
            }
        )
        rows.append(
            {
                "claim": "DTR only reduces A; content score does not rise",
                "holds": bool(p_a_down and not content_up),
                "conclusion": "DTR mostly fixes calibration",
                "dtr_mean_p_A": _f(d, "mean_p_A"),
                "vanilla_mean_p_A": _f(v, "mean_p_A"),
                "dtr_content_acc": _f(d, "content_acc"),
            }
        )
    if nr and r:
        nr_better_letter = _f(nr, "letter_acc") > _f(r, "letter_acc")
        nr_better_content = _f(nr, "content_acc") > _f(r, "content_acc")
        rows.append(
            {
                "claim": "NoRelation better than RelJoin on letter and content",
                "holds": bool(nr_better_letter and nr_better_content),
                "conclusion": "Pairwise relational computation is failing / collapsing",
                "norelation_letter_acc": _f(nr, "letter_acc"),
                "reljoin_letter_acc": _f(r, "letter_acc"),
                "norelation_content_acc": _f(nr, "content_acc"),
                "reljoin_content_acc": _f(r, "content_acc"),
            }
        )
    return rows


def perm_payload_from_run(pred_letters: list[str], chosen_contents: list[str], n_rot: int) -> dict:
    return {
        "perm_pred_letters_json": json.dumps(pred_letters, ensure_ascii=False),
        "perm_contents_json": json.dumps(chosen_contents, ensure_ascii=False),
        "content_consistency": content_consistency(chosen_contents, n_rotations=n_rot),
        "first_position_rate": first_position_rate(pred_letters),
    }
