"""Evaluation metrics (skeleton)."""
from __future__ import annotations
from typing import Sequence


def chord_accuracy(pred: Sequence[str], ref: Sequence[str]) -> float:
    if not ref:
        return 0.0
    n = min(len(pred), len(ref))
    if n == 0:
        return 0.0
    correct = sum(1 for i in range(n) if pred[i] == ref[i])
    return correct / n


def voice_leading_penalty(chords: Sequence[str]) -> float:
    # Placeholder: return 0.0 (no penalty) until a real metric is implemented
    return 0.0
