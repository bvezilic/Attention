import os
import os.path as osp
import sys
from typing import Text, Callable

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import Compose

from base import History
from data import NMTDataset
from global_attention.model import Seq2Seq
from metrics import Metric, BLEUMetric
from mixin import NameMixIn
from tokenizer import Tokenizer
from transform import ToTokens, ToIndices, ToTensor, ToWords
from utils import read_params
from vocab import Vocabulary

# Set random seed
torch.random.manual_seed(47)


class Trainer(NameMixIn):
    def __init__(self,
                 dataset: NMTDataset,
                 model: Seq2Seq,
                 optimizer: Optimizer,
                 criterion: Callable,
                 metric: Metric,
                 batch_size: int,
                 device: Text,
                 save_dir: Text,
                 padding_idx: int = 0,
                 test_split: float = 0.1):
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric
        self.batch_size = batch_size
        self.device = device
        self.save_dir = save_dir
        self.padding_idx = padding_idx
        self.test_split = test_split

        # Set training and test data set
        if 1. > self.test_split > 0.:
            self.train_dataset, self.test_dataset = self.train_test_split()
        else:
            self.train_dataset = self.dataset
            self.test_dataset = None

        # Initialize train data loader
        self.train_loader = DataLoader(self.train_dataset,
                                       shuffle=True,
                                       batch_size=batch_size,
                                       collate_fn=lambda x: dataset.collate_fn(x, padding_value=0))
        if self.test_dataset is not None:
            self.test_loader = DataLoader(self.test_dataset,
                                          shuffle=True,
                                          batch_size=batch_size,
                                          collate_fn=lambda x: dataset.collate_fn(x, padding_value=0))
        else:
            self.test_loader = None

        # Set model to device
        self.model.to(device)
        self.model.device = device

    def __repr__(self):
        return f"{self.name}:\n" \
            f"dataset={self.dataset}\n" \
            f"model={self.model}\n" \
            f"optimizer={self.optimizer}\n" \
            f"criterion={self.criterion}\n" \
            f"device={self.device}\n" \
            f"batch_size={self.batch_size}"

    def train(self, epochs: int) -> History:
        """
        Runs training for number of `epochs`.

        Returns:
            losses: List of losses for each epoch
        """
        self.model.train()  # Set model in train mode
        train_history = History()

        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}/{epochs}")

            running_loss = 0
            running_score = 0
            for i, (inputs, targets) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                inputs = inputs.to(self.device)  # [B, T]
                targets = targets.to(self.device)  # [B, T]
                mask_ids = (inputs != self.padding_idx)  # Create mask where word_idx=1 and pad=0

                # Obtain predictions
                output = self.model(inputs=inputs,
                                    mask_ids=mask_ids,
                                    targets=targets)

                # Compute cross-entropy loss
                loss = self.criterion(input=output["logits"].reshape(-1, output["logits"].size(2)),
                                      target=targets.reshape(-1))
                # Compute gradients
                loss.backward()

                # Update weights
                self.optimizer.step()

                # Compute metric score
                score = self.metric.score(y_pred=output["predictions"], y_true=targets)

                running_loss += loss.item()
                running_score += score
                print(f"BATCH {i + 1}/{len(self.train_loader)} - loss: {loss:.4f} - score: {score:.4f}")

            train_loss = running_loss / len(self.train_loader)
            train_score = running_score / len(self.train_loader)
            print(f"TRAIN - loss: {train_loss:.4f} - score: {train_score:.4f}")

            # Run evaluation on test set if available
            (test_loss, test_score) = self.eval() if self.test_dataset else (None, None)

            # Update training history
            train_history.update(train_loss=train_loss,
                                 train_score=train_score,
                                 test_loss=test_loss,
                                 test_score=test_score)

            # Save model to `save_dir`
            self.save_model(epoch_num=epoch, epoch_loss=train_loss, epoch_score=train_score)

        return train_history

    def eval(self) -> (float, float):
        """
        Run forward pass for test dataset and computes metric score
        """
        self.model.eval()  # Set model to evaluation

        with torch.no_grad():
            running_loss = 0
            running_score = 0
            for i, (inputs, targets) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)  # [B, T]
                targets = targets.to(self.device)  # [B, T]
                mask_ids = (inputs != 0)  # Create mask where word_idx=1 and pad=0

                # Obtain predictions
                output = self.model(inputs=inputs,
                                    mask_ids=mask_ids,
                                    targets=targets)

                # Compute cross-entropy loss
                loss = self.criterion(input=output["logits"].reshape(-1, output["logits"].size(2)),
                                      target=targets.reshape(-1))

                # Compute metric score
                score = self.metric.score(y_pred=output["predictions"], y_true=targets)

                running_loss += loss.item()
                running_score += score

            epoch_loss = running_loss / len(self.test_loader)
            metric_score = running_score / len(self.test_loader)
            print(f"TEST - loss: {epoch_loss:.4f} - score: {metric_score:.4f}")

            self.model.train()  # Return model to train mode

            return epoch_loss, metric_score

    def train_test_split(self) -> (Subset, Subset):
        """
        Randomly splits dataset into 'train_subset' and 'test_subset'
        """
        test_size = int(len(self.dataset) * self.test_split)
        train_size = len(self.dataset) - test_size

        train_subset, test_subset = random_split(self.dataset, lengths=[train_size, test_size])

        return train_subset, test_subset

    def save_model(self, epoch_num: int, epoch_loss: float, epoch_score: float) -> None:
        """
        Saves model to given directory. If directory doesn't exist, one will be created.
        """
        if not osp.exists(self.save_dir):
            print(f"Creating directory on path '{osp.abspath(self.save_dir)}'...")
            os.makedirs(self.save_dir)

        save_path = osp.join(self.save_dir, f"seq2seq_ep:{epoch_num}-loss:{epoch_loss:.4f}-score:{epoch_score:.4f}.pt")
        self.model.save(save_path)


def train():
    # Load the vocabulary
    src_vocab = Vocabulary.from_file(args.src_vocab)
    tar_vocab = Vocabulary.from_file(args.dst_vocab)

    # Initialize the data set
    dataset = NMTDataset(args.data,
                         src_transform=Compose([ToTokens(tokenizer=Tokenizer(end_token=src_vocab.end_token)),
                                                ToIndices(vocabulary=src_vocab),
                                                ToTensor(dtype=torch.long)]),
                         tar_transform=Compose([ToTokens(tokenizer=Tokenizer(end_token=tar_vocab.end_token)),
                                                ToIndices(vocabulary=tar_vocab),
                                                ToTensor(dtype=torch.long)]))

    # Initialize the model
    model = Seq2Seq(enc_vocab_size=src_vocab.size,
                    dec_vocab_size=tar_vocab.size,
                    hidden_size=model_params["hidden_size"],
                    embedding_dim=model_params["embedding_dim"],
                    attn_vec_size=model_params["attn_vec_size"])

    # Initialize the optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Initialize the loss criterion
    criterion = nn.CrossEntropyLoss(ignore_index=tar_vocab.pad_token.idx)

    # Initialize the score metric
    metric = BLEUMetric(transforms=ToWords(tar_vocab))

    # Initialize the training and run training loop
    trainer = Trainer(dataset=dataset,
                      model=model,
                      optimizer=optimizer,
                      criterion=criterion,
                      metric=metric,
                      batch_size=args.batch_size,
                      device=args.device,
                      save_dir=args.save_dir,
                      padding_idx=src_vocab.pad_token.idx)
    history = trainer.train(args.epochs)
    history.save("./train_log.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../dataset/fra.txt",
                        help="Path to train data set.")
    parser.add_argument("--src_vocab", type=str, default="../dataset/eng_vocab.txt",
                        help="Path to source vocabulary")
    parser.add_argument("--dst_vocab", type=str, default="../dataset/fra_vocab.txt",
                        help="Path to destination vocabulary")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of epochs to run training")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Number of samples per batch")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--save_dir", type=str, default="./trained_models",
                        help="Directory in which to save model")
    parser.add_argument("--model_params", type=str, default="../config/global_attention.json",
                        help="Path to json config file of model parameters")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="Run train on cuda/cpu")

    args = parser.parse_args()

    # Read model params from file
    model_params = read_params(args.model_params)

    # Validate if cuda is available
    if args.device == "cuda" and torch.cuda.is_available() is False:
        sys.stderr("Set to use cuda, but cuda is not available!")
        sys.stderr("Setting device to use CPU instead of GPU!")
        args.device = "cpu"

    train()
