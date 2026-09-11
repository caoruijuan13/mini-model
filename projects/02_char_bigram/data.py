"""阶段二：字符语料和顺序切分。"""


def make_char_splits(text: str) -> tuple[str, str, str]:
    if len(text) < 30:
        raise ValueError("text must contain at least 30 characters")
    a, b = int(len(text) * 0.8), int(len(text) * 0.9)
    return text[:a], text[a:b], text[b:]


DEFAULT_CORPUS = (
    "small models are useful because every prediction can be inspected. "
    "we train, validate, explain, and then build a tiny language model. "
    "reproducible experiments make failures understandable. "
) * 8
