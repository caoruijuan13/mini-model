"""Stage 04 demonstration: text -> token IDs -> text."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from projects.tokenization import CharTokenizer
from model import TokenBigram


CORPUS_PATH = PROJECT_ROOT / "data" / "declaration_excerpt.txt"
DEFAULT_TOKENIZER_PATH = Path(__file__).with_name("artifacts") / "char_tokenizer.json"


def make_text_splits(text: str) -> tuple[str, str, str]:
    """Use the existing 80/10/10 sequential split for a fair stage comparison."""
    if len(text) < 30:
        raise ValueError("text must contain at least 30 characters")
    train_end = int(len(text) * 0.8)
    valid_end = int(len(text) * 0.9)
    return text[:train_end], text[train_end:valid_end], text[valid_end:]


def make_bigram_examples(token_ids: list[int]) -> list[tuple[int, int]]:
    """Show that model examples are built after tokenization."""
    return list(zip(token_ids, token_ids[1:]))


def run_demo(
    tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
) -> dict[str, int | str | bool]:
    corpus = CORPUS_PATH.read_text(encoding="utf-8").rstrip("\n")
    train, valid, test = make_text_splits(corpus)

    # The vocabulary is intentionally built from training text only.
    tokenizer = CharTokenizer.from_text(train)
    train_ids = tokenizer.encode(train, add_bos=True, add_eos=True)
    valid_ids = tokenizer.encode(valid, add_bos=True, add_eos=True)
    test_ids = tokenizer.encode(test, add_bos=True, add_eos=True)
    valid_unknown_tokens = sum(
        character not in tokenizer.token_to_id for character in valid
    )
    test_unknown_tokens = sum(
        character not in tokenizer.token_to_id for character in test
    )

    tokenizer.save(tokenizer_path)
    loaded = CharTokenizer.load(tokenizer_path)
    encoded_sample = "We hold"
    encoded_sample_ids = loaded.encode(
        encoded_sample, add_bos=True, add_eos=True
    )

    model = TokenBigram(loaded.vocab_size)
    model.fit(train_ids)
    generated_ids = model.sample(
        start_id=loaded.token_to_id["<BOS>"],
        length=100,
        seed=7,
        eos_id=loaded.token_to_id["<EOS>"],
    )
    generated_text = loaded.decode(generated_ids, skip_special_tokens=True)

    return {
        "vocab_size": loaded.vocab_size,
        "train_tokens": len(train_ids),
        "valid_tokens": len(valid_ids),
        "test_tokens": len(test_ids),
        "valid_unknown_tokens": valid_unknown_tokens,
        "test_unknown_tokens": test_unknown_tokens,
        "train_bigram_examples": len(make_bigram_examples(train_ids)),
        "encoded_sample": encoded_sample,
        "encoded_sample_ids": str(encoded_sample_ids),
        "round_trip_ok": loaded.decode(
            encoded_sample_ids, skip_special_tokens=True
        ) == encoded_sample,
        "generated_tokens": len(generated_ids),
        "generated_text": generated_text,
    }


if __name__ == "__main__":
    for key, value in run_demo().items():
        print(f"{key} = {value}")
