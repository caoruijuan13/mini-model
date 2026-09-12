import json
from pathlib import Path
import sys

import pytest

from projects.tokenization import CharTokenizer

STAGE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(STAGE_ROOT))
from demo import make_bigram_examples


def test_vocabulary_is_deterministic_and_round_trip_works():
    tokenizer = CharTokenizer.from_text("cabca")

    assert tokenizer.tokens[:3] == ("<UNK>", "<BOS>", "<EOS>")
    assert tokenizer.tokens[3:] == ("a", "b", "c")
    assert tokenizer.vocab_size == 6
    assert tokenizer.decode(tokenizer.encode("cab")) == "cab"
    with pytest.raises(TypeError):
        tokenizer.token_to_id["new"] = tokenizer.vocab_size


def test_special_tokens_and_empty_text_are_explicit():
    tokenizer = CharTokenizer.from_text("ab")

    token_ids = tokenizer.encode("ab", add_bos=True, add_eos=True)
    assert token_ids[0] == tokenizer.token_to_id["<BOS>"]
    assert token_ids[-1] == tokenizer.token_to_id["<EOS>"]
    assert tokenizer.decode(token_ids) == "<BOS>ab<EOS>"
    assert tokenizer.decode(token_ids, skip_special_tokens=True) == "ab"
    assert tokenizer.encode("") == []


def test_unknown_character_policy_uses_training_vocabulary_only():
    tokenizer = CharTokenizer.from_text("aaab")

    assert "c" not in tokenizer.token_to_id
    assert tokenizer.encode("ac") == [
        tokenizer.token_to_id["a"],
        tokenizer.token_to_id["<UNK>"],
    ]
    with pytest.raises(ValueError, match="unknown character"):
        tokenizer.encode("ac", on_unknown="error")


def test_save_and_load_preserve_complete_contract(tmp_path):
    tokenizer = CharTokenizer.from_text("We hold")
    path = tmp_path / "tokenizer.json"

    tokenizer.save(path)
    loaded = CharTokenizer.load(path)

    assert loaded == tokenizer
    assert loaded.token_to_id == tokenizer.token_to_id
    assert loaded.encode("We", add_bos=True, add_eos=True) == tokenizer.encode(
        "We", add_bos=True, add_eos=True
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "mini-model-char-tokenizer"
    assert payload["version"] == 1


def test_invalid_ids_and_bigram_examples_have_clear_boundaries():
    tokenizer = CharTokenizer.from_text("abc")
    ids = tokenizer.encode("abc")

    assert make_bigram_examples(ids) == list(zip(ids, ids[1:]))
    with pytest.raises(ValueError, match="outside vocabulary"):
        tokenizer.decode([tokenizer.vocab_size])
    with pytest.raises(TypeError, match="must be integers"):
        tokenizer.decode([True])
