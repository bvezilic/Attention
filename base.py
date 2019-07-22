import os
from typing import Text, List, Dict, Any
import sys
import torch
from torch import nn


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


class History:
    """
    Container class for storing losses/scores during training.
    """
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.train_scores = []
        self.test_scores = []

    def update(self,
               train_loss: float,
               train_score: float,
               test_loss: float = None,
               test_score: float = None):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)

        if test_score is None:
            self.train_scores.append(train_score)
        if test_loss is None:
            self.test_scores.append(test_score)
