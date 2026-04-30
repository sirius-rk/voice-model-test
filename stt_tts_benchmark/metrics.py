from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    normalized = text.lower().strip()
    normalized = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _levenshtein(source: list[str], target: list[str]) -> int:
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous = list(range(len(target) + 1))
    for row_index, source_item in enumerate(source, start=1):
        current = [row_index]
        for col_index, target_item in enumerate(target, start=1):
            substitution_cost = 0 if source_item == target_item else 1
            current.append(
                min(
                    previous[col_index] + 1,
                    current[col_index - 1] + 1,
                    previous[col_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)
    reference_words = normalized_reference.split()
    hypothesis_words = normalized_hypothesis.split()
    if not reference_words:
        return None
    return _levenshtein(reference_words, hypothesis_words) / len(reference_words)


def character_error_rate(reference: str, hypothesis: str) -> float | None:
    normalized_reference = normalize_text(reference).replace(" ", "")
    normalized_hypothesis = normalize_text(hypothesis).replace(" ", "")
    if not normalized_reference:
        return None
    return _levenshtein(list(normalized_reference), list(normalized_hypothesis)) / len(normalized_reference)

