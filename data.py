from typing import Text, Callable, List, Tuple, Any, Dict

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from mixin import NameMixIn
from tokenizer import Tokenizer


class NMTDataset(Dataset, NameMixIn):
    """
    Storage class for Nature Language Translation data set.
    """
    def __init__(self,
                 filepath: Text,
                 src_language: Text = "English",
                 tar_language: Text = "French",
                 src_transform: Callable = None,
                 tar_transform: Callable = None):
        self.filepath = filepath
        self.src_lang = src_language
        self.tar_lang = tar_language
        self.src_transform = src_transform
        self.tar_transform = tar_transform

        # Load the data as pd.DataFrame
        self.train_data = pd.read_csv(self.filepath, sep="\t", names=[src_language, tar_language])

    def __len__(self):
        return len(self.train_data)

    def __repr__(self):
        return f"{self.name}(\n" \
            f"  filepath={self.filepath},\n" \
            f"  src_lang={self.src_lang},\n"\
            f"  tar_lang={self.tar_lang},\n" \
            f"  src_transform={self.src_transform},\n" \
            f"  tar_transform={self.tar_transform}\n)"

    def __getitem__(self, i):
        src, tar = self.train_data.iloc[i]

        if self.src_transform:
            src = self.src_transform(src)

        if self.tar_transform:
            tar = self.tar_transform(tar)

        return src, tar

    def max_sequence_len(self, tokenizer: Tokenizer) -> Dict[Text, int]:
        """
        Computes max sequence length of the data set for both source and target language.
        > Depending on the size of the data set, tokenization might take some time.

        Args:
            tokenizer: Tokenizer with func `tokenize`.

        Returns:
            seq_max_lengths: Dictionary with max sequence length for both source and target language. Dictionary key
                correspond to language names.
        """
        print("Converting text into tokens...")
        df = self.train_data.applymap(tokenizer.tokenize)
        seq_max_lengths = df.applymap(len).max().to_dict()

        return seq_max_lengths

    @staticmethod
    def collate_fn(batch: List[Tuple[Any]], padding_value: int = 0) -> (torch.Tensor, torch.Tensor):
        """
        Pads items (inputs and targets) in batch to the length of maximum sequence.

        Args:
            batch: List of tuples [(input1, target1), (input2, target2), ...]
            padding_value: (Optional) Defaults to 0.

        Returns:
            inputs: Tensor of shape [batch_size, max_seq_length_of_inputs]
            targets: Tensor of shape [batch_size, max_seq_length_of_targets]
        """
        inputs, targets = zip(*batch)

        inputs = pad_sequence(inputs, batch_first=True, padding_value=padding_value)
        targets = pad_sequence(targets, batch_first=True, padding_value=padding_value)

        return inputs, targets


if __name__ == "__main__":
    dataset = NMTDataset("./dataset/fra.txt")
    print(dataset)
    print(dataset.train_data.head())
