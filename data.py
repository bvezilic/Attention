import pandas as pd

from torch.utils.data import Dataset


class NMTDataset(Dataset):
    def __init__(self, filepath, src_transform=None, tar_transform=None):
        self.filepath = filepath
        self.src_transform = src_transform
        self.tar_transform = tar_transform

        # Load the data as pd.DataFrame
        self.train_data = pd.read_csv(self.filepath, sep="\t", names=["English", "French"])

    def __len__(self):
        return len(self.train_data)

    def __repr__(self):
        return "{}(filepath={}, src_transform={}, tar_transform={})".format(
            self.__class__.__name__, self.filepath, self.src_transform, self.tar_transform)

    def __getitem__(self, i):
        src, tar = self.train_data.iloc[i]

        if self.src_transform:
            src = self.src_transform(src)

        if self.tar_transform:
            tar = self.tar_transform(tar)

        return src, tar


if __name__ == "__main__":
    dataset = NMTDataset("./dataset/fra.txt")
    print(dataset)
    print(dataset.train_data.head())
