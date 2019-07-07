import pandas as pd

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class NMTDataset(Dataset):
    def __init__(self, filepath, src_language="English", tar_language="French", src_transform=None, tar_transform=None):
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
        return "{}(filepath={}, src_transform={}, tar_transform={})".format(
            self.name, self.filepath, self.src_transform, self.tar_transform)

    def __getitem__(self, i):
        src, tar = self.train_data.iloc[i]

        if self.src_transform:
            src = self.src_transform(src)

        if self.tar_transform:
            tar = self.tar_transform(tar)

        return src, tar

    @property
    def name(self):
        return self.__class__.__name__

    @staticmethod
    def collate_fn(batch):
        """
        Pads the input and target tensor to `padding_value=0`
        """
        inp, tar = zip(*batch)

        # padding
        padding_value = 0
        inp = pad_sequence(inp, batch_first=True, padding_value=padding_value)
        tar = pad_sequence(tar, batch_first=True, padding_value=padding_value)

        return inp, tar


if __name__ == "__main__":
    dataset = NMTDataset("./dataset/fra.txt")
    print(dataset)
    print(dataset.train_data.head())
