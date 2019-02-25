import re
import pandas as pd
from torch.utils.data import Dataset
from collections import Iterable


class NMTDataset(Dataset):
    def __init__(self, filepath, transform=None, tokenizer=None):
        self.filepath = filepath
        self.transform = transform
        self.tokenizer = tokenizer or Tokenizer()

        # Load the data as pd.DataFrame
        self.data = pd.read_csv(self.filepath, sep="\t")

        # Create vocabularies for source and target language
        self.src_vocab = Vocabulary([self.tokenizer(text) for text in self.data.iloc[:, 0]])
        self.tar_vocab = Vocabulary(self.data.iloc[:, 1])

        # Create train data

        self.train_data = None

    def create_vocabulary(self, data):
        tokens = []
        for text in data:
            tokens.extend(self.tokenizer(text))

        return Vocabulary(tokens)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, i):
        sample = self.train_data[i]

        if self.transform:
            sample = self.transform(sample)

        return sample


class Vocabulary:
    def __init__(self, tokens, pad_token="[PAD]", end_token="[EOS]", unknown_token="[UNK]"):
        self.pad_token = pad_token
        self.end_token = end_token
        self.unkown_token = unknown_token

        self.token2idx_dict = {self.pad_token: 0, self.end_token: 1, self.unkown_token: 2}
        # Update the vocabulary
        idx = 3
        for token in tokens:
            if token not in self.token2idx_dict:
                self.token2idx_dict[token] = idx
                idx += 1
        self.idx2token_dict = {idx: token for token, idx in self.token2idx_dict.items()}

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


class Tokenizer:
    def __init__(self, do_lower=True):
        self.do_lower = do_lower
        self.pattern = r" +"  # Split over multiple spaces

    def __call__(self, text):
        if self.do_lower:
            text = text.lower()

        tokens = re.split(self.pattern, text)
        return tokens
