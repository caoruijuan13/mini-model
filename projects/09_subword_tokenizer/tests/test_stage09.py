"""Active scaffold checks; unskip BPE acceptance tests as TODOs are filled."""

from importlib import import_module

import pytest


bpe = import_module("projects.09_subword_tokenizer.bpe")
data = import_module("projects.09_subword_tokenizer.data")
comparison = import_module("projects.09_subword_tokenizer.compare")
stage07_data = import_module("projects.07_char_transformer.data")


def test_config_has_explicit_limits():
    assert bpe.BPEConfig(max_merges=0).max_merges == 0
    with pytest.raises(ValueError, match="max_merges"):
        bpe.BPEConfig(max_merges=-1)
    with pytest.raises(ValueError, match="min_pair_frequency"):
        bpe.BPEConfig(min_pair_frequency=0)


def test_raw_text_splits_match_stage07():
    expected = stage07_data.make_text_splits(stage07_data.load_corpus())
    assert data.prepare_text_splits() == expected
    assert "".join(expected) == stage07_data.load_corpus()


def test_tokenizer_state_has_base_vocab_and_ordered_merge_slots():
    tokenizer = bpe.BPETokenizer(
        base_tokens=("<UNK>", "<BOS>", "<EOS>", "a"),
        tokens=("<UNK>", "<BOS>", "<EOS>", "a"),
        merges=(),
    )
    assert tokenizer.vocab_size == 4
    assert tokenizer.merges == ()
    assert tokenizer.FORMAT_VERSION == 1


def test_tokenizer_init_accepts_ordered_merges():
    tokenizer = bpe.BPETokenizer(
        base_tokens=("<UNK>", "<BOS>", "<EOS>", "a", "b"),
        tokens=("<UNK>", "<BOS>", "<EOS>", "a", "b", "ab", "abab"),
        merges=((3, 4), (5, 5)),
    )
    assert tokenizer.vocab_size == 7


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"tokens": ("<UNK>", "<BOS>", "<EOS>", "a", "a")}, "unique"),
        ({"tokens": ("<UNK>", "<BOS>", "<EOS>", "b")}, "base_tokens"),
        ({"bos_token": "<UNK>"}, "distinct"),
        ({"unk_token": "<MISSING>"}, "base_tokens"),
        ({"tokens": ("<UNK>", "<BOS>", "<EOS>", "a", "ab")}, "exactly one"),
        ({"tokens": ("<UNK>", "<BOS>", "<EOS>", "a", "aa"), "merges": ((4, 3),)}, "previously defined"),
        ({"tokens": ("<UNK>", "<BOS>", "<EOS>", "a", "bb"), "merges": ((3, 3),)}, "concatenation"),
    ],
)
def test_tokenizer_init_rejects_inconsistent_state(changes, message):
    values = {
        "base_tokens": ("<UNK>", "<BOS>", "<EOS>", "a"),
        "tokens": ("<UNK>", "<BOS>", "<EOS>", "a"),
        "merges": (),
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        bpe.BPETokenizer(**values)


def test_comparison_fits_only_train_text_and_uses_same_lines(tmp_path, monkeypatch):
    corpus_text = "abcdefghij" * 4
    path = tmp_path / "corpus.txt"
    path.write_text(corpus_text, encoding="utf-8")
    observed = []

    class FakeBPE:
        vocab_size = 15

        @classmethod
        def train(cls, train_text, *, config):
            observed.append(train_text)
            return cls()

        def encode(self, text):
            return list(text)

    monkeypatch.setattr(comparison, "BPETokenizer", FakeBPE)
    report = comparison.compare_tokenizers(corpus_path=path)
    train_text, _, _ = data.prepare_text_splits(path)
    assert observed == [train_text]
    assert report["train"]["char_token_count"] == report["train"]["bpe_token_count"]
    assert report["valid"]["character_count"] == len(corpus_text) // 10
    assert report["test"]["character_count"] == len(corpus_text) // 10


def test_pair_count_includes_overlapping_occurrences():
    assert bpe.count_pairs([3, 4, 3, 4]) == {(3, 4): 2, (4, 3): 1}
    assert bpe.count_pairs([3, 3, 3]) == {(3, 3): 2}


def test_pair_selection_uses_frequency_then_id_tie_break():
    counts = {(4, 3): 2, (3, 5): 2, (3, 4): 2}
    assert bpe.select_best_pair(counts, min_frequency=2) == (3, 4)
    assert bpe.select_best_pair(counts, min_frequency=3) is None

    counts = {(4, 4): 3, (3, 3): 2}
    assert bpe.select_best_pair(counts, min_frequency=1) == (4, 4)

def test_merge_pair_is_left_to_right_and_non_overlapping():
    assert bpe.merge_pair([3, 3, 3], (3, 3), 5) == [5, 3]
    assert bpe.merge_pair([3, 4, 3, 4], (3, 4), 5) == [5, 5]


def test_abab_learning_records_ordered_merges():
    tokenizer = bpe.BPETokenizer.train(
        "abab", config=bpe.BPEConfig(max_merges=2, min_pair_frequency=1)
    )
    assert tokenizer.base_tokens == ("<UNK>", "<BOS>", "<EOS>", "a", "b")
    assert tokenizer.merges == ((3, 4), (5, 5))
    assert tokenizer.tokens == tokenizer.base_tokens + ("ab", "abab")


def test_encode_uses_merge_rank_and_decode_round_trips_known_text():
    tokenizer = bpe.BPETokenizer.train(
        "abab", config=bpe.BPEConfig(max_merges=2, min_pair_frequency=1)
    )
    assert tokenizer.encode("ababab") == [6, 5]
    assert tokenizer.decode([6, 5]) == "ababab"


def test_unknown_and_special_token_boundaries():
    tokenizer = bpe.BPETokenizer.train("abab")
    assert tokenizer.encode("z") == [0]
    with pytest.raises(ValueError, match="unknown"):
        tokenizer.encode("z", on_unknown="error")
    ids = tokenizer.encode("ab", add_bos=True, add_eos=True)
    assert ids[0] == 1 and ids[-1] == 2
    assert tokenizer.decode(ids, skip_special_tokens=True) == "ab"


@pytest.mark.parametrize("policy", ["ignore", "invalid"])
def test_encode_rejects_unsupported_unknown_policy_for_known_text(policy):
    tokenizer = bpe.BPETokenizer.train("abab")
    with pytest.raises(ValueError, match="on_unknown"):
        tokenizer.encode("ab", on_unknown=policy)


@pytest.mark.parametrize("token_id", [True, False, 1.0])
def test_decode_rejects_noninteger_token_ids(token_id):
    tokenizer = bpe.BPETokenizer.train("abab")
    with pytest.raises(TypeError, match="integer"):
        tokenizer.decode([token_id])


def test_save_load_preserves_ids(tmp_path):
    tokenizer = bpe.BPETokenizer.train(
        "abab", config=bpe.BPEConfig(max_merges=2, min_pair_frequency=1)
    )
    path = tmp_path / "bpe.pt"
    tokenizer.save(path)
    restored = bpe.BPETokenizer.load(path)
    assert restored.tokens == tokenizer.tokens
    assert restored.merges == tokenizer.merges
    assert restored.encode("ababab") == tokenizer.encode("ababab")


@pytest.mark.parametrize(
    ("format_field", "incompatible_value"),
    [("FORMAT_NAME", "other-bpe-format"), ("FORMAT_VERSION", 2)],
)
def test_load_rejects_incompatible_format(tmp_path, monkeypatch, format_field, incompatible_value):
    tokenizer = bpe.BPETokenizer.train("abab")
    path = tmp_path / "bpe.pt"
    tokenizer.save(path)
    monkeypatch.setattr(bpe.BPETokenizer, format_field, incompatible_value)
    with pytest.raises(ValueError):
        bpe.BPETokenizer.load(path)
