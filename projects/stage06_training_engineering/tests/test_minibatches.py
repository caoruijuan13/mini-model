import numpy as np
import pytest

from data import DeterministicBatcher


def _batcher(*, batch_size=4, seed=11):
    return DeterministicBatcher(
        np.arange(20).reshape(10, 2),
        np.arange(10),
        batch_size=batch_size,
        seed=seed,
    )


def test_minibatches_cover_each_example_once_per_epoch():
    batcher = _batcher()
    seen = []
    for step in range(3):
        _, batch_targets, epoch, batch_index = batcher.batch_for_step(step)
        assert epoch == 0
        assert batch_index == step
        seen.extend(batch_targets.tolist())
    assert sorted(seen) == list(range(10))


def test_batch_selection_is_reproducible_from_global_step():
    first = _batcher(batch_size=3, seed=7).batch_for_step(5)
    second = _batcher(batch_size=3, seed=7).batch_for_step(5)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2:] == second[2:] == (1, 1)


def test_batch_input_validation():
    with pytest.raises(ValueError, match="equal length"):
        DeterministicBatcher(
            np.zeros((2, 1)), np.zeros(1), batch_size=1, seed=1
        )


def test_permutation_is_built_once_per_epoch():
    class CountingBatcher(DeterministicBatcher):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.build_count = 0

        def _build_permutation(self, epoch):
            self.build_count += 1
            return super()._build_permutation(epoch)

    batcher = CountingBatcher(
        np.arange(20).reshape(10, 2), np.arange(10), batch_size=4, seed=11
    )
    for step in range(batcher.per_epoch):
        batcher.batch_for_step(step)
    assert batcher.build_count == 1

    batcher.batch_for_step(batcher.per_epoch)
    assert batcher.build_count == 2
