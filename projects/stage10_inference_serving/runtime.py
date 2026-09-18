""" runtime about inference serving """

from projects.tokenization.char_tokenizer import CharTokenizer
from projects.stage08_pytorch_transformer.model import TorchCharTransformer


class InferenceRuntime:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def load(cls, path):
        model, tokenizer = TorchCharTransformer.load_with_tokenizer(
    "projects/stage08_pytorch_transformer/artifacts/char_transformer.pt"
)
