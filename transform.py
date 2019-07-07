import torch


class ToIndices:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def __repr__(self):
        return "{}(vocabulary={})".format(self.name, self.vocabulary)

    def __call__(self, tokens):
        return self.vocabulary.token2idx(tokens)

    @property
    def name(self):
        return self.__class__.__name__


class ToTokens:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __repr__(self):
        return "{}(tokenizer={})".format(self.name, self.tokenizer)

    def __call__(self, text):
        return self.tokenizer.tokenize(text)

    @property
    def name(self):
        return self.__class__.__name__


class ToTensor:
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __repr__(self):
        return "{}(dtype={})".format(self.name, self.dtype)

    def __call__(self, seq):
        return torch.tensor(seq, dtype=self.dtype)

    @property
    def name(self):
        return self.__class__.__name__
