from torchtext.data import Field, LabelField
from torchtext.datasets import SST


def load_SST(root_dir):
    TEXT = Field(lower=True, tokenize="toktok", eos_token="<eos>")
    LABEL = LabelField()

    train, val, test = SST.splits(TEXT, LABEL, root=root_dir)

    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    return train, val, test, TEXT, LABEL