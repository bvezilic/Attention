import pandas as pd

from torch.utils.data import Dataset


class NMTDataset(Dataset):
    def __init__(self, filepath, transform=None):
        self.filepath = filepath
        self.transform = transform

        # Load the data as pd.DataFrame
        self.train_data = pd.read_csv(self.filepath, sep="\t", names=["English", "French"])

    def __len__(self):
        return len(self.train_data)

    def __repr__(self):
        return "{}: filepath='{}', transform={}".format(
            self.__class__.__name__, self.filepath, [func.__name__ for func in self.transform] if self.transform else [])

    def __getitem__(self, i):
        sample = self.train_data[i]

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    dataset = NMTDataset("./dataset/fra.txt")
    print(dataset)
    print(dataset.train_data.head())
