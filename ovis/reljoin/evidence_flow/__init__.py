"""TSO evidence-flow figure extraction. Does not modify checkpoints or eval logs."""

from .geometry import run_merge_order_self_test, verified_token_boxes
from .mine import rank_tso_candidates

__all__ = [
    "rank_tso_candidates",
    "run_merge_order_self_test",
    "verified_token_boxes",
]
