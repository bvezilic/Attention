import logging
from typing import List, Text, Callable

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from data import NMTDataset
from model import Seq2Seq
from tokenizer import Tokenizer
from transform import ToTokens, ToIndices, ToTensor
from utils import read_params
from vocab import Vocabulary

logger = logging.getLogger(__file__)


class Trainer:
    def __init__(self,
                 dataset: NMTDataset,
                 model: Seq2Seq,
                 optimizer: Optimizer,
                 criterion: Callable,
                 batch_size: int,
                 device: Text):
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.batch_size = batch_size

        # Initialize train data loader
        self.data_loader = DataLoader(dataset,
                                      shuffle=True,
                                      batch_size=batch_size,
                                      collate_fn=lambda x: dataset.collate_fn(x, padding_value=0))

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

    @property
    def name(self) -> Text:
        return self.__class__.__name__

    def train(self, epochs: int) -> List[float]:
        """
        Runs training for number of `epochs`.

        Returns:
            losses: List of losses for each epoch
        """
        losses = []

        for epoch in range(epochs):
            logger.info("Epoch: {}/{}".format(epoch+1, epochs))

            running_loss = 0
            for i, (inputs, targets) in enumerate(self.data_loader):
                self.optimizer.zero_grad()

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
                # Compute gradients
                loss.backward()

                # Update weights
                self.optimizer.step()

                running_loss += loss
                logger.info("Batch loss {}/{}: {:.4f}".format(i+1, len(self.data_loader), loss))

            logger.info("Epoch loss: {}".format(running_loss/len(self.data_loader)))

        return losses


def train():
    # Load the vocabulary
    eng_vocab = Vocabulary.from_file(args.src_vocab)
    fra_vocab = Vocabulary.from_file(args.dst_vocab)

    # Initialize the data set
    dataset = NMTDataset(args.data,
                         src_transform=Compose([ToTokens(tokenizer=Tokenizer(end_token=eng_vocab.end_token)),
                                                ToIndices(vocabulary=eng_vocab),
                                                ToTensor(dtype=torch.long)]),
                         tar_transform=Compose([ToTokens(tokenizer=Tokenizer(end_token=fra_vocab.end_token)),
                                                ToIndices(vocabulary=fra_vocab),
                                                ToTensor(dtype=torch.long)]))

    # Initialize the model
    model = Seq2Seq(enc_vocab_size=eng_vocab.size,
                    dec_vocab_size=fra_vocab.size,
                    hidden_size=model_params["hidden_size"],
                    embedding_dim=model_params["embedding_dim"],
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
    parser.add_argument("--debug", action="store_true",
                        help="Set logger to debugging level")

    args = parser.parse_args()

    # Read model params from file
    model_params = read_params(args.model_params)

    # Set debugging level if provided
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Validate if cuda is available
    if args.device == "cuda" and torch.cuda.is_available() is False:
        logger.warning("Set to use cuda, but cuda is not available!")
        logger.warning("Setting device to use CPU instead of GPU!")
        args.device = "cpu"

    train()
