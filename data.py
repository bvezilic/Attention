import pandas as pd

from torch.utils.data import Dataset


class NMTDataset(Dataset):
    def __init__(self, filepath, src_vocab=None, tar_vocab=None, transform=None):
        self.filepath = filepath
        self.src_vocab = src_vocab
        self.tar_vocab = tar_vocab
        self.transform = transform

        # Load the data as pd.DataFrame
        self.train_data = pd.read_csv(self.filepath, sep="\t", names=["English", "French"])

    def __len__(self):
        return len(self.train_data)

    def __repr__(self):
        return "{}: filepath='{}', transform={}".format(
            self.__class__.__name__, self.filepath, [func.__name__ for func in self.transform] if self.transform else [])

    def __getitem__(self, i):
        src, tar = self.train_data.iloc[i]

        if self.transform:
            src = self.transform(src)
            tar = self.transform(tar)

        if self.src_vocab:
            src = self.src_vocab.token2idx(src)

        if self.tar_vocab:
            tar = self.tar_vocab.token2idx(tar)

        return src, tar


if __name__ == "__main__":
    dataset = NMTDataset("./dataset/fra.txt")
    print(dataset)
    print(dataset.train_data.head())
