import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from data import NMTDataset
from tokenizer import Tokenizer
from transform import ToTokens, ToIndices, ToTensor
from vocab import Vocabulary
from .model import Seq2Seq


class Trainer:
    def __init__(self, dataset, model, optimizer, criterion, device):
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.data_loader = DataLoader(dataset, shuffle=True)

    def train(self, epochs):
        losses = []

        for epoch in range(epochs):
            print("Epoch: {}/{}".format(epoch+1, epochs))

            running_loss = 0
            for src, tar in self.data_loader:
                src = src.to(self.device)
                tar = tar.to(self.device)

                preds, attn_weights = self.model(src)
                loss = self.criterion(preds, tar)

                loss.backward()
                self.optimizer.step()

                running_loss += loss

                print("test")

        return losses


def train():
    # Load the vocabulary
    eng_vocab = Vocabulary.from_file("./dataset/eng_vocab.txt")
    fra_vocab = Vocabulary.from_file("./dataset/fra_vocab.txt")

    # Initialize the dataset
    dataset = NMTDataset("./dataset/fra.txt",
                         src_transform=Compose([ToTokens(Tokenizer()), ToIndices(eng_vocab), ToTensor(torch.long)]),
                         tar_transform=Compose([ToTokens(Tokenizer()), ToIndices(fra_vocab), ToTensor(torch.long)]))
    print(dataset)
    # Initialize the model
    model = Seq2Seq(enc_vocab_size=eng_vocab.size,
                    dec_vocab_size=fra_vocab.size,
                    enc_hidden_size=512,
                    dec_hidden_size=512,
                    output_size=fra_vocab.size,
                    embedding_size=300,
                    attn_size=512)

    # Initialize the optimizer
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Initialize the loss criterion
    criterion = nn.CrossEntropyLoss()

    # Initialize the training and run training loop
    trainer = Trainer(dataset=dataset,
                      model=model,
                      optimizer=optimizer,
                      criterion=criterion,
                      device="cpu")
    trainer.train(20)


if __name__ == "__main__":
    train()
