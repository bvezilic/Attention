import string
from typing import Union, Text, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import Compose

from global_attention.model import Seq2Seq
from tokenizer import Tokenizer
from transform import ToTokens, ToTensor, ToIndices, ToWords
from utils import filter_tokens
from vocab import Vocabulary, Token


class Translation:
    """
    Stores translation predictions and provides visualization of attention weights.
    """

    def __init__(self,
                 input_tokens: List[Text],
                 pred_tokens: List[Text],
                 attn_weights: np.ndarray,
                 end_token: Token):
        self.input_tokens = input_tokens
        self.pred_tokens = pred_tokens
        self.attn_weights = attn_weights
        self.end_token = end_token
        self.eos_index = self.pred_tokens.index(end_token.name)

    def __repr__(self):
        return self.text

    @property
    def text(self) -> Text:
        text = ""
        for i, token in enumerate(self.pred_tokens[:self.eos_index]):
            if any(char in set(string.punctuation) for char in token) or i == 0:
                text += "" + token
            else:
                text += " " + token
        return text

    @property
    def attention_matrix(self) -> np.ndarray:
        if self.eos_index is None:
            return self.attn_weights
        else:
            return self.attn_weights[:, :self.eos_index]

    def plot_attention(self) -> None:
        """Plots heatmap of attention weights.
        """
        trans_words = self.pred_tokens[:self.eos_index]

        return plt.matshow(self.attention_matrix,
                           xticklabels=trans_words,
                           yticklabels=self.input_tokens,
                           vmin=0,
                           vmax=1,
                           annot=True,
                           fmt=".2f")


class Predictor:
    """
    Translate text or list of texts from source to target language based on pre-trained Seq2Seq model.
    """

    def __init__(self,
                 model: Seq2Seq,
                 tokenizer: Tokenizer,
                 src_vocab: Vocabulary,
                 tar_vocab: Vocabulary,
                 device: Text = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.src_vocab = src_vocab
        self.tar_vocab = tar_vocab
        self.device = device

        # Set model to given device
        self.model.to(device)
        self.model.device = self.device
        self.model.eval()

        # Transforms
        self.src_transform = Compose([ToTokens(tokenizer=self.tokenizer),
                                      ToIndices(vocabulary=self.src_vocab),
                                      ToTensor(dtype=torch.long)])
        self.tar_transform = Compose([ToWords(vocabulary=tar_vocab)])

    def __call__(self, texts: Union[Text, List[Text]]) -> Union[Translation, List[Translation]]:
        """Runs inference on single line of text or list of texts. Forward pass is done per example not in batches.

        Args:
            texts: Single sentence or list of sentences for translation.

        Returns:
            Single or multiple dict objects like:
        """
        if isinstance(texts, str):
            return self._predict_text(texts)
        elif isinstance(texts, list):
            return [self._predict_text(text) for text in texts]
        else:
            raise ValueError(f"Expected object of type 'str' or 'list' got {type(texts)}")

    def _predict_text(self, text: Text) -> Union[Translation, List[Translation]]:
        """Run inference on single line of text.

        Args:
            text (str): Sentence to translate

        Returns:
            translation (Translation):
        """
        with torch.no_grad():
            # Convert text to model input
            inputs = self.src_transform(text)
            inputs = inputs.unsqueeze(0)  # Add batch_size
            inputs = inputs.to(self.device)

            # Create mask for padded tokens
            mask_ids = (inputs != self.src_vocab.pad_token.idx)

            # Obtain predictions
            output = self.model(inputs=inputs, mask_ids=mask_ids)

            # Process pred_tokens and attn weights
            pred_tokens = self.tar_transform(output["predictions"][0].tolist())
            text_tokens = filter_tokens(pred_tokens)
            input_tokens = self.src_transform.transforms[0](text)

            # Convert attn_weights to numpy array
            attn_weights = output["attn_weights"].squeeze(0).t().numpy()

            return Translation(input_tokens=input_tokens,
                               pred_tokens=pred_tokens,
                               attn_weights=attn_weights,
                               end_token=self.tar_vocab.end_token)


if __name__ == '__main__':
    from global_attention.model import Seq2Seq

    # Load the vocabulary
    src_vocab = Vocabulary.from_file("../dataset/eng_vocab.txt")
    tar_vocab = Vocabulary.from_file("../dataset/fra_vocab.txt")

    # Initialize the model
    model = Seq2Seq.load(
        "/home/bane/code/Attention/global_attention/trained_models/seq2seq_ep:9-loss:1.3504-score:0.3818.pt")

    predict = Predictor(model, Tokenizer(end_token=src_vocab.end_token), src_vocab, tar_vocab)
    t = predict("Does this work?")

    print(t)
