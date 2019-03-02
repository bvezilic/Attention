from collections import Iterable

from tokenizer import Tokenizer


class Vocabulary:
    PAD_TOKEN = "[PAD]"
    END_TOKEN = "[EOS]"
    UNKNOWN_TOKEN = "[UNK]"

    def __init__(self, token2idx_dict=None, idx2token_dict=None):
        assert (token2idx_dict is not None and idx2token_dict is None) or \
               (token2idx_dict is None and idx2token_dict is not None), "Only one dictionary must be provided"

        self.token2idx_dict = {}
        self.idx2token_dict = {}

        if token2idx_dict is not None:
            self.token2idx_dict.update(token2idx_dict)
            self.idx2token_dict.update({idx: token for token, idx in self.token2idx_dict.items()})

        if idx2token_dict is not None:
            self.idx2token_dict.update(idx2token_dict)
            self.token2idx_dict.update({token: idx for idx, token in self.idx2token_dict.items()})

    def __repr__(self):
        return "{}: token2idx_dict={} tokens, idx2token_dict={} tokens".format(
            self.__class__.__name__, len(self.token2idx_dict), len(self.idx2token_dict))

    def token2idx(self, tokens):
        if isinstance(tokens, Iterable):
            return [self.token2idx_dict.get(token) for token in tokens]
        elif isinstance(tokens, str):
            return self.token2idx_dict.get(tokens)
        else:
            raise TypeError("Tokens must be of type str or iterable! Given '{}'".format(type(tokens)))

    def idx2token(self, idx):
        if isinstance(idx, Iterable):
            return [self.idx2token_dict.get(i) for i in idx]
        elif isinstance(idx, int):
            return self.idx2token_dict.get(idx)
        else:
            raise TypeError("Tokens must be of type int or iterable! Given '{}'".format(type(idx)))

    def save(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for token in self.token2idx_dict:
                f.write(token + "\n")

    @classmethod
    def from_file(cls, filepath):
        token2idx_dict = {}
        with open(filepath, "r") as f:
            for idx, token in enumerate(f):
                token2idx_dict[token] = idx

        return cls(idx2token_dict=token2idx_dict)

    @classmethod
    def from_texts(cls, texts, do_lower=True):
        tokenizer = Tokenizer(do_lower=do_lower)
        tokens = []
        for text in texts:
            tokens.extend(tokenizer(text))

        return cls.from_tokens(tokens=tokens)

    @classmethod
    def from_tokens(cls, tokens):
        token2idx_dict = {cls.PAD_TOKEN: 0, cls.END_TOKEN: 1, cls.UNKNOWN_TOKEN: 2}
        idx = 3
        for token in tokens:
            if token not in token2idx_dict:
                token2idx_dict[token] = idx
                idx += 1

        return cls(token2idx_dict=token2idx_dict)


if __name__ == "__main__":
    # Create english and french vocabulary from the dataset
    from data import NMTDataset

    dataset = NMTDataset("./dataset/fra.txt")

    english_texts = dataset.train_data.iloc[:, 0]
    french_texts = dataset.train_data.iloc[:, 1]

    english_vocab = Vocabulary.from_texts(texts=english_texts)
    french_vocab = Vocabulary.from_texts(texts=french_texts)

    english_vocab.save("./dataset/eng_vocab.txt")
    french_vocab.save("./dataset/fra_vocab.txt")
