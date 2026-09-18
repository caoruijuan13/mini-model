"""阶段三：真实字符语料和顺序切分。"""


def make_char_splits(text: str) -> tuple[str, str, str]:
    if len(text) < 30:
        raise ValueError("text must contain at least 30 characters")
    a, b = int(len(text) * 0.8), int(len(text) * 0.9)
    return text[:a], text[a:b], text[b:]


# 美国《独立宣言》的公共领域英文节选。
REAL_CORPUS = (
    "We hold these truths to be self-evident, that all men are created equal, "
    "that they are endowed by their Creator with certain unalienable Rights, "
    "that among these are Life, Liberty and the pursuit of Happiness. "
    "That to secure these rights, Governments are instituted among Men, "
    "deriving their just powers from the consent of the governed. "
    "That whenever any Form of Government becomes destructive of these ends, "
    "it is the Right of the People to alter or to abolish it, and to institute "
    "new Government, laying its foundation on such principles and organizing "
    "its powers in such form, as to them shall seem most likely to effect their "
    "Safety and Happiness."
)
