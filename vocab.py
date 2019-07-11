from collections import Iterable
from typing import Dict
from typing import Union, List, Text

from mixin import NameMixIn
from tokenizer import Token, Tokenizer


class Vocabulary(NameMixIn):
    pad_token = Token(name="[PAD]", idx=0)
    end_token = Token(name="[EOS]", idx=1)
    unknown_token = Token(name="[UNK]", idx=2)

    def __init__(self, token2idx_dict: Dict = None, idx2token_dict: Dict = None):
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
        return f"{self.name}(token2idx_dict={len(self.token2idx_dict)}, idx2token_dict={len(self.idx2token_dict)}\n)"

    @property
    def size(self) -> int:
        return len(self.token2idx_dict)

    def token2idx(self, tokens: Union[Iterable, List, Text]) -> Union[List[int], int]:
        """
        Retrieve tokens index from vocabulary. If not found, an unknown token index is provided instead.
        """
        if isinstance(tokens, Iterable):
            return [self.token2idx_dict.get(token, self.unknown_token.idx) for token in tokens]
        elif isinstance(tokens, str):
            return self.token2idx_dict.get(tokens, self.unknown_token.idx)
        else:
            raise TypeError("Tokens must be of type str or iterable! Given '{}'".format(type(tokens)))

    def idx2token(self, idx: Union[Iterable, List, int]) -> Union[List[Text], Text]:
        """
        Retrieve tokens from the vocabulary based on index. If out of bound, an unknown token name is provided instead.
        """
        if isinstance(idx, Iterable):
            return [self.idx2token_dict.get(i, self.unknown_token.name) for i in idx]
        elif isinstance(idx, int):
            return self.idx2token_dict.get(idx, self.unknown_token.name)
        else:
            raise TypeError("Tokens must be of type int or iterable! Given '{}'".format(type(idx)))

    def save(self, filepath: Text) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            for token in self.token2idx_dict:
                f.write(token + "\n")

    @classmethod
    def from_file(cls, filepath: Text) -> "Vocabulary":
        idx2token_dict = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, token in enumerate(f):
                idx2token_dict[idx] = token.strip()

        return cls(idx2token_dict=idx2token_dict)

    @classmethod
    def from_texts(cls, texts: List[Text], do_lower: bool = True) -> "Vocabulary":
        tokenizer = Tokenizer(do_lower=do_lower)
        tokens = []
        for text in texts:
            tokens.extend(tokenizer.tokenize(text))

        return cls.from_tokens(tokens=tokens)

    @classmethod
    def from_tokens(cls, tokens: List[Text]) -> "Vocabulary":
        token2idx_dict = {cls.pad_token.name: cls.pad_token.idx,
                          cls.end_token.name: cls.end_token.idx,
                          cls.unknown_token.name: cls.unknown_token.idx}
        idx = 3  # indices 0,1,2 are reserved
        for token in tokens:
            if token not in token2idx_dict:
                token2idx_dict[token] = idx
                idx += 1

        return cls(token2idx_dict=token2idx_dict)


def create_vocabs(filepath: Text, src_vocab: Text, tar_vocab: Text) -> None:
    """
    Create vocabularies from `NMTDataset`.

    Args:
        filepath: Path to txt file containing Translation. Used to initialise `NMTDataset`.
        src_vocab: Path where to save source language vocabulary
        tar_vocab: Path where to save target language vocabulary
    """
    from data import NMTDataset

    dataset = NMTDataset(filepath)

    english_texts = dataset.train_data.iloc[:, 0]
    french_texts = dataset.train_data.iloc[:, 1]

    english_vocab = Vocabulary.from_texts(texts=english_texts)
    french_vocab = Vocabulary.from_texts(texts=french_texts)

    english_vocab.save(src_vocab)
    french_vocab.save(tar_vocab)


if __name__ == "__main__":
    create_vocabs(filepath="./dataset/fra.txt",
                  src_vocab="./dataset/eng_vocab.txt",
                  tar_vocab="./dataset/fra_vocab.txt")
