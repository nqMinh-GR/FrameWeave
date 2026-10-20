"""Qualitative dumps for FrameWeave Hybrid v6. Inactive unless a session is open."""

from .collector import FrameVizSession, active_session, note_delivered_tokens

__all__ = [
    "FrameVizSession",
    "active_session",
    "note_delivered_tokens",
]
