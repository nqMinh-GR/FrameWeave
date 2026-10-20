"""Training objectives for FrameWeave Hybrid v6.0 (V5.1) on Video-LLaVA.

Primary objective is per-sample logsumexp of candidate CE and Vanilla-relative
correction CE, then a batch mean. Letter scores are raw logits at the first
answer position. Do not use HF outputs.loss (fp16 Llama CE overflows).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import IGNORE_INDEX

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def answer_loss_from_outputs(outputs) -> torch.Tensor:
    """HF CE on labels already aligned with the multimodal sequence."""
    loss = getattr(outputs, "loss", None)
    if loss is None:
        raise RuntimeError(
            "Model returned no loss. Pass labels= into the forward so the LM computes "
            "cross-entropy on the multimodal label sequence."
        )
    return loss


def expanded_answer_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, int]:
    """fp32 length-averaged CE on already-aligned labels. Returns (loss, n_label_tokens)."""
    n_ans = int((labels != ignore_index).sum().item())
    loss = answer_token_nll(logits.float(), labels, ignore_index=ignore_index)
    return loss, n_ans


def answer_token_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Length-averaged NLL over non-ignored answer tokens."""
    if logits.size(1) != labels.size(1):
        raise ValueError(
            f"logits length {logits.size(1)} != labels length {labels.size(1)}. "
            "For Video-LLaVA use hook-expanded last_labels."
        )
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index, reduction="none")
    mask = flat_labels != ignore_index
    if mask.sum() == 0:
        return logits.new_zeros(())
    return loss[mask].mean()


def letter_token_ids(tokenizer, n_choices: int) -> torch.Tensor:
    """Token id of each option letter A.. (first token if the letter splits)."""
    ids: list[int] = []
    unk = int(getattr(tokenizer, "unk_token_id", 0) or 0)
    for i in range(int(n_choices)):
        toks = tokenizer.encode(LETTERS[i], add_special_tokens=False)
        ids.append(int(toks[0]) if toks else unk)
    return torch.tensor(ids, dtype=torch.long)


def first_answer_logit_row(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Vocab logits that predict the first non-ignored answer token. Shape [B, V]."""
    if logits.size(1) != labels.size(1):
        raise ValueError(
            f"logits length {logits.size(1)} != labels length {labels.size(1)}."
        )
    mask = labels != ignore_index
    has = mask.any(dim=1)
    first = mask.to(dtype=torch.long).argmax(dim=1)
    pred_pos = (first - 1).clamp(min=0)
    bidx = torch.arange(logits.size(0), device=logits.device)
    rows = logits.float()[bidx, pred_pos]
    return torch.where(has.unsqueeze(-1), rows, torch.zeros_like(rows))


def candidate_letter_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    letter_ids: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """z_m = logit(letter_m) at the first answer position. Shape [B, M]."""
    rows = first_answer_logit_row(logits, labels, ignore_index=ignore_index)
    ids = letter_ids.to(device=rows.device, dtype=torch.long)
    return rows.index_select(dim=-1, index=ids)


def reljoin_v5_loss(
    z_rel: torch.Tensor,
    z_van: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample logsumexp(L_cand, L_corr) - log 2.

    Returns (loss_each [B], loss_cand [B], loss_corr [B]). Combine per sample
    before the batch mean. Vanilla scores must already be stopgrad.
    """
    loss_cand = F.cross_entropy(z_rel.float(), target, reduction="none")
    loss_corr = F.cross_entropy((z_rel.float() - z_van.float().detach()), target, reduction="none")
    stacked = torch.stack([loss_cand, loss_corr], dim=-1)
    loss_each = torch.logsumexp(stacked, dim=-1) - math.log(2.0)
    return loss_each, loss_cand, loss_corr


def batch_reljoin_v5_loss(
    z_rel: torch.Tensor,
    z_van: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch mean of the v5 sample loss."""
    loss_each, loss_cand, loss_corr = reljoin_v5_loss(z_rel, z_van, target)
    return loss_each.mean(), loss_cand.mean(), loss_corr.mean()
