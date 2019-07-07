from collections import Iterable, namedtuple

from tokenizer import Tokenizer

Token = namedtuple('Token', ["name", "idx"])


class Vocabulary:
    pad_token = Token(name="[PAD]", idx=0)
    end_token = Token(name="[EOS]", idx=1)
    unknown_token = Token(name="[UNK]", idx=2)

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
        return "{}(token2idx_dict={}, idx2token_dict={})".format(
            self.name, len(self.token2idx_dict), len(self.idx2token_dict))

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def size(self):
        return len(self.token2idx_dict)

    def token2idx(self, tokens):
        if isinstance(tokens, Iterable):
            return [self.token2idx_dict.get(token, self.unknown_token.idx) for token in tokens]
        elif isinstance(tokens, str):
            return self.token2idx_dict.get(tokens, self.unknown_token.idx)
        else:
            raise TypeError("Tokens must be of type str or iterable! Given '{}'".format(type(tokens)))

    def idx2token(self, idx):
        if isinstance(idx, Iterable):
            return [self.idx2token_dict.get(i, self.unknown_token.name) for i in idx]
        elif isinstance(idx, int):
            return self.idx2token_dict.get(idx, self.unknown_token.name)
        else:
            raise TypeError("Tokens must be of type int or iterable! Given '{}'".format(type(idx)))

    def save(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for token in self.token2idx_dict:
                f.write(token + "\n")

    @classmethod
    def from_file(cls, filepath):
        idx2token_dict = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, token in enumerate(f):
                idx2token_dict[idx] = token.strip()

        return cls(idx2token_dict=idx2token_dict)

    @classmethod
    def from_texts(cls, texts, do_lower=True):
        tokenizer = Tokenizer(do_lower=do_lower)
        tokens = []
        for text in texts:
            tokens.extend(tokenizer.tokenize(text))

        return cls.from_tokens(tokens=tokens)

    @classmethod
    def from_tokens(cls, tokens):
        token2idx_dict = {cls.pad_token.name: cls.pad_token.idx,
                          cls.end_token.name: cls.end_token.idx,
                          cls.unknown_token.name: cls.unknown_token.idx}
        idx = 3
        for token in tokens:
            if token not in token2idx_dict:
                token2idx_dict[token] = idx
                idx += 1

        return cls(token2idx_dict=token2idx_dict)


def create_vocabs(filepath, eng_vocab, fra_vocab):
    """Create vocabularies from English and French texts."""
    from data import NMTDataset

    dataset = NMTDataset(filepath)

    english_texts = dataset.train_data.iloc[:, 0]
    french_texts = dataset.train_data.iloc[:, 1]

    english_vocab = Vocabulary.from_texts(texts=english_texts)
    french_vocab = Vocabulary.from_texts(texts=french_texts)

    english_vocab.save(eng_vocab)
    french_vocab.save(fra_vocab)


if __name__ == "__main__":
    create_vocabs(filepath="./dataset/fra.txt",
                  eng_vocab="./dataset/eng_vocab.txt",
                  fra_vocab="./dataset/fra_vocab.txt")
