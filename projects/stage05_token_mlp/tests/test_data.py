from pathlib import Path
import sys

import pytest

STAGE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(STAGE_ROOT))
sys.modules.pop("data", None)

from data import build_context_targets

def test_contexts_are_left_padded_and_include_eos_target():
    contexts, targets = build_context_targets(
        [3, 4], context_size=2, bos_id=1, eos_id=2
    )

    assert contexts.tolist() == [[1, 1], [1, 3], [3, 4]]
    assert targets.tolist() == [3, 4, 2]
    assert contexts.shape == (3, 2)
    assert targets.shape == (3,)


def test_context_size_three_and_input_is_not_modified():
    token_ids = [5, 6, 7]
    original = token_ids.copy()

    contexts, targets = build_context_targets(
        token_ids, context_size=3, bos_id=1, eos_id=2
    )

    assert contexts.tolist() == [
        [1, 1, 1],
        [1, 1, 5],
        [1, 5, 6],
        [5, 6, 7],
    ]
    assert targets.tolist() == [5, 6, 7, 2]
    assert token_ids == original


def test_empty_sequence_still_predicts_eos_from_bos_context():
    contexts, targets = build_context_targets(
        [], context_size=2, bos_id=1, eos_id=2
    )

    assert contexts.tolist() == [[1, 1]]
    assert targets.tolist() == [2]
    assert contexts.shape == (1, 2)
    assert targets.shape == (1,)
