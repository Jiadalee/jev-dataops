"""Standalone reference-only reward for exported verl jobs; no external calls or code evaluation."""

import unicodedata


MAX_REWARD_CHARACTERS = 131072


def _normalize(value):
    if not isinstance(value, str) or len(value) > MAX_REWARD_CHARACTERS:
        return None
    # Preserve case, punctuation and numbers: this is exact answer matching,
    # not a semantic judge or an executable mathematical/code checker.
    return " ".join(unicodedata.normalize("NFC", value).split())


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """verl's keyword callback; compare generated text only with an explicit reference.

    Missing, empty, oversized or non-text references/responses receive zero reward.
    Metadata, JEV scores and model-produced instructions never change the algorithm.
    """
    expected = _normalize(ground_truth)
    actual = _normalize(solution_str)
    score = float(bool(expected) and actual is not None and actual == expected)
    return {"score": score, "acc": score}
