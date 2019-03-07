import torch


class ToIndices:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, tokens):
        return self.vocabulary.token2idx(tokens)


class ToTokens:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text):
        return self.tokenizer.tokenize(text)


class ToTensor:
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, seq):
        return torch.tensor(seq, dtype=self.dtype)
