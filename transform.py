from typing import List, Text, Any

import torch

from mixin import NameMixIn
from tokenizer import Tokenizer
from vocab import Vocabulary


class ToTokens(NameMixIn):
    """
    Transforms text (usually sentence) into list of tokens based on tokenizer.
    """
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __repr__(self):
        return f"{self.name}(\n" \
            f"  tokenizer={self.tokenizer}\n)"

    def __call__(self, text: Text) -> List[Text]:
        return self.tokenizer.tokenize(text)


class ToIndices(NameMixIn):
    """
    Transforms word tokens into indices based on vocabulary.
    """
    def __init__(self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary

    def __repr__(self):
        return f"{self.name}(vocabulary={self.vocabulary})"

    def __call__(self, tokens: List[Text]) -> List[int]:
        return self.vocabulary.token2idx(tokens)


class ToTensor(NameMixIn):
    """
    Transforms given data to `torch.Tensor` of specific `dtype`.
    """
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __repr__(self):
        return f"{self.name}(dtype={self.dtype})"

    def __call__(self, seq: List[Any]):
        return torch.tensor(seq, dtype=self.dtype)
