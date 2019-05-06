import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from data import NMTDataset
from tokenizer import Tokenizer
from transform import ToTokens, ToIndices, ToTensor
from vocab import Vocabulary
from model import Seq2Seq
from utils import read_params


class Trainer:
    def __init__(self, dataset, model, optimizer, criterion, batch_size, device):
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.batch_size = batch_size

        # Initialize train data loader
        self.data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=dataset.collate_fn)

        # Set model to given device
        self.model.to(device)
        self.model.device = device

    def train(self, epochs):
        losses = []

        for epoch in range(epochs):
            print("Epoch: {}/{}".format(epoch+1, epochs))

            running_loss = 0
            for i, (inp, tar) in enumerate(self.data_loader):
                self.optimizer.zero_grad()

                inp = inp.to(self.device)  # [B, T]
                tar = tar.to(self.device)  # [B, T]

                logits, attn_weights = self.model(inp)
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
    eng_vocab = Vocabulary.from_file(args.src_vocab)
    fra_vocab = Vocabulary.from_file(args.dst_vocab)

    # Initialize the dataset
    dataset = NMTDataset(args.data,
                         src_transform=Compose([ToTokens(Tokenizer()), ToIndices(eng_vocab), ToTensor(torch.long)]),
                         tar_transform=Compose([ToTokens(Tokenizer()), ToIndices(fra_vocab), ToTensor(torch.long)]))
    print(dataset)
    # Initialize the model
    model = Seq2Seq(enc_vocab_size=eng_vocab.size,
                    dec_vocab_size=fra_vocab.size,
                    hidden_size=model_params["hidden_size"],
                    embedding_dim=model_params["embedding_dim"],
                    output_size=fra_vocab.size,
                    attn_vec_size=model_params["attn_vec_size"])

    # Initialize the optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Initialize the loss criterion
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # Initialize the training and run training loop
    trainer = Trainer(dataset=dataset,
                      model=model,
                      optimizer=optimizer,
                      criterion=criterion,
                      batch_size=args.batch_size,
                      device=args.device)
    trainer.train(20)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../dataset/fra.txt",
                        help="Path to train data set.")
    parser.add_argument("--src_vocab", type=str, default="../dataset/eng_vocab.txt",
                        help="Path to source vocabulary")
    parser.add_argument("--dst_vocab", type=str, default="../dataset/fra_vocab.txt",
                        help="Path to destination vocabulary")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of samples per batch")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--model_params", type=str, default="../config/global_attention.json",
                        help="Path to json config file of model parameters")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="Run train on cuda/cpu")

    args = parser.parse_args()
    model_params = read_params(args.model_params)

    # Validate if cuda is available
    if args.device == "cuda" and torch.cuda.is_available() is False:
        raise ValueError("Set to use cuda, but cuda is not available!")

    train()
