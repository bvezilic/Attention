import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu

from mixin import NameMixIn
from utils import filter_tokens


class Metric(NameMixIn):
    def score(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        raise NotImplementedError


class AccuracyMetric(Metric):
    def __init__(self):
        super().__init__()

    def score(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Computes accuracy for two tensors of the same shape.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Tensors 'y_true'={y_true.size()} and 'y_pred'={y_pred.size()} must be of the same shape!")

        correct = (y_true == y_pred).sum()
        total = y_true.shape.numel()

        return (correct / total).item()


class BLEUMetric(Metric):
    def __init__(self, transforms):
        super().__init__()

        self.transforms = transforms

    def score(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Computes BLEU metric using NLTK.

        Args:
            y_true (torch.Tensor): Tensor containing references
                shape=[batch_size, seq_length]
            y_pred (torch.Tensor): Tensor containing hypothesis
                shape=[batch_size, seq_length]

        Returns:
            score: Computed BLEU metric
        """
        y_true_shape = y_true.shape
        y_pred_shape = y_pred.shape

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Tensors 'y_true'={y_true.size()} and 'y_pred'={y_pred.size()} must be of the same shape!")

        if self.transforms:
            # Convert tensor to list
            y_pred = y_pred.view(-1).tolist()
            y_true = y_true.view(-1).tolist()

            # Transform data and convert to numpy array
            y_pred = np.array(self.transforms(y_pred))
            y_true = np.array(self.transforms(y_true))

            # Reshape back to original shape
            y_pred = y_pred.reshape(y_pred_shape)
            y_true = y_true.reshape(y_true_shape)

        # Convert to list to match NLTK bleu input
        hypotheses = y_pred.tolist()
        references = np.expand_dims(y_true, axis=1).tolist()

        # Filter out special tokens (PAD, EOS, UNK)
        hypotheses = filter_tokens(hypotheses)
        references = filter_tokens(references)

        score = corpus_bleu(list_of_references=references, hypotheses=hypotheses)

        return score
