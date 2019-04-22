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
            for i, (src, tar) in enumerate(self.data_loader):
                self.optimizer.zero_grad()

                src = src.to(self.device)  # [B, T]
                tar = tar.to(self.device)  # [B, T]

                logits, attn_weights = self.model(src)
                # logits[T, B=1, dec_vocab_size]
                # attn_weights[T, B=1, enc_outputs-time_steps]

                if logits.size(0) >= tar.size(1):
                    preds = logits[:tar.size(1)]
                else:
                    preds = torch.zeros(tar.size(1), preds.size()[1:])
                    preds[:logits.size(0)] = logits

                loss = self.criterion(preds.reshape(-1, preds.size(2)), tar.reshape(-1))

                loss.backward()
                self.optimizer.step()

                running_loss += loss
                print("Batch loss {}/{}: {}".format(i+1, len(self.data_loader), loss))

            print("Epoch loss: {}".format(running_loss/len(self.data_loader)))

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
                    attn_vec_size=512)
    model.to("cuda")

    # Initialize the optimizer
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Initialize the loss criterion
    criterion = nn.CrossEntropyLoss()

    # Initialize the training and run training loop
    trainer = Trainer(dataset=dataset,
                      model=model,
                      optimizer=optimizer,
                      criterion=criterion,
                      device="cuda")
    trainer.train(20)


if __name__ == "__main__":
    train()
