import json
import os
import sys
from typing import Text, List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from mixin import NameMixIn


class BaseModel(nn.Module):
    """
    Tracks objects attributes and provides methods for model serialization/deserialization
    """

    @property
    def attribute_names(self) -> List[Text]:
        raise NotImplementedError

    @property
    def params_dict(self) -> Dict[Text, Any]:
        if not hasattr(self, 'attribute_names'):
            sys.stderr("No 'attribute_names' provided, saving/loading model might be unavailable.")
            return {}

        return {attr: getattr(self, attr) for attr in self.attribute_names}

    def save(self, path: Text) -> None:
        """Saves model parameters and weights.
        """
        torch.save({
            "model_params": self.params_dict,
            "state_dict": self.state_dict()
        }, path)

    @classmethod
    def load(cls, path: Text) -> "nn.Module":
        """Restore model and load weights.
        """
        path_ = os.path.abspath(path)
        restore = torch.load(path_)

        model = cls(**restore["model_params"])
        model.load_state_dict(restore["state_dict"])
        model.eval()

        return model


class History(NameMixIn):
    """
    Container class for storing losses/scores during training.
    """

    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.train_scores = []
        self.test_scores = []

    def __repr__(self):
        return f"{self.name}:\n" \
               f"train_losses={len(self.train_losses)}\n" \
               f"test_losses={len(self.test_losses)}\n" \
               f"train_scores={len(self.train_scores)}\n" \
               f"test_scores={len(self.test_scores)}"

    def update(self,
               train_loss: float = None,
               train_score: float = None,
               test_loss: float = None,
               test_score: float = None) -> None:
        """Updates storage attributes for train and test loss/score.
        """
        self.add_to_list(self.train_losses, train_loss)
        self.add_to_list(self.train_scores, train_score)
        self.add_to_list(self.test_losses, test_loss)
        self.add_to_list(self.test_scores, test_score)

    def save(self, path: Text):
        """Save as json object on given path.
        """
        abs_path = os.path.abspath(path)

        with open(abs_path, "w") as fp:
            print(f"Saving history train on '{abs_path}'")
            json.dump(vars(self), fp)

    def plot(self):
        """Plots two graphs:

        1. Obtained train/test loss during the training for each epoch
        2. Obtained train/test score during the training for each epoch
        """
        fig, ax = plt.subplots(2, 1)

        x = np.arange(len(self.train_losses))
        ax[0].plot(x, self.train_losses, label="train")
        ax[0].plot(x, self.test_losses, label="test")
        ax[0].set_title('Loss')
        ax[0].legend()

        x = np.arange(len(self.train_scores))
        ax[1].plot(x, self.train_scores, label="train")
        ax[1].plot(x, self.test_scores, label="test")
        ax[1].set_title('Score')
        ax[1].legend()

        plt.show()

    @classmethod
    def from_json(cls, path: Text) -> "History":
        abs_path = os.path.abspath(path)

        with open(abs_path, "r") as fp:
            print(f"Loading train history from '{abs_path}'")
            restore = json.load(fp)

            return cls.from_dict(dictionary=restore)

    @classmethod
    def from_dict(cls, dictionary) -> "History":
        history = cls()
        history.train_losses = dictionary.get("train_losses") or []
        history.train_scores = dictionary.get("train_scores") or []
        history.test_losses = dictionary.get("test_losses") or []
        history.test_scores = dictionary.get("test_scores") or []

        return history

    @staticmethod
    def add_to_list(l: List[float], value: float) -> None:
        """Append value to list if value is not None
        """
        if value is not None:
            l.append(value)
