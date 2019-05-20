import pandas as pd

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


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

    @staticmethod
    def collate_fn(batch, padding_value=0):
        """
        Pads the input and target tensor
        :param batch: list of tuples [(input1, target1), (input2, target2), ...]
        :param padding_value: int (default=0)
        :return: inputs: Tensor
                 targets: Tensor
        """
        inputs, targets = zip(*batch)

        # padding
        inputs = pad_sequence(inputs, batch_first=True, padding_value=padding_value)
        targets = pad_sequence(targets, batch_first=True, padding_value=padding_value)

        return inputs, targets


if __name__ == "__main__":
    dataset = NMTDataset("./dataset/fra.txt")
    print(dataset)
    print(dataset.train_data.head())
