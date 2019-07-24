import os
import sys
from typing import Text, List, Dict, Any

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
        torch.save({
            "model_params": self.params_dict,
            "state_dict": self.state_dict()
        }, path)

    @classmethod
    def load(cls, path: Text) -> "nn.Module":
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
        self.add_to_list(self.train_losses, train_loss)
        self.add_to_list(self.train_scores, train_score)
        self.add_to_list(self.test_losses, test_loss)
        self.add_to_list(self.test_scores, test_score)

    @staticmethod
    def add_to_list(l: List[float], value: float) -> None:
        """
        Append value to list if value is not None
        """
        if value is not None:
            l.append(value)
