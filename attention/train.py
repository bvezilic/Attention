from torch.utils.data import DataLoader, RandomSampler

from data import NMTDataset
from tokenizer import Tokenizer
from vocab import Vocabulary
from attention.model import Seq2Seq


class Trainer:
    def __init__(self, dataset, model, optimizer, device):
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.data_loader = DataLoader(dataset, sampler=RandomSampler(dataset))

    def train(self, epochs):
        losses = []

        for epoch in range(epochs):
            print("Epoch: {}/{}".format(epoch+1, epochs))

            for src, tar in self.data_loader:
                src = src.to(self.device)
                tar = tar.to(self.device)


if __name__ == "__main__":
    # Load the vocabulary
    eng_vocab = Vocabulary.from_file("../dataset/eng_vocab.txt")
    fra_vocab = Vocabulary.from_file("../dataset/fra_vocab.txt")

    # Initialize the dataset
    dataset = NMTDataset("../dataset/fra.txt",
                         src_vocab=eng_vocab,
                         tar_vocab=fra_vocab,
                         transform=Tokenizer(end_token=Vocabulary.END["token"]))

    # Initialize the model
    model = Seq2Seq(src_vocab_size=src_vocab.size)
