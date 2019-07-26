from typing import Union, Text, List, Any, Dict

import torch
from torchvision.transforms import Compose

from tokenizer import Tokenizer
from transform import ToTokens, ToTensor, ToIndices, ToWords
from model import Seq2Seq
from utils import filter_tokens
from vocab import Vocabulary


class Predictor:
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

    def __call__(self, texts: Union[Text, List[Text]]) -> Union[Dict, List[Dict]]:
        if isinstance(texts, str):
            return self._predict_text(texts)
        elif isinstance(texts, list):
            return [self._predict_text(text) for text in texts]
        else:
            raise ValueError(f"Expected object of type 'str' or 'list' got {type(texts)}")

    def _predict_text(self, text: Text) -> Dict[Text, Any]:
        """

        Args:
            text:

        Returns:

        """
        with torch.no_grad():
            # Convert text to model input
            inputs = self.src_transform(text)
            inputs = inputs.unsqueeze(0)  # Add batch_size
            inputs = inputs.to(self.device)

            # Create mask for padded tokens
            mask_ids = (inputs != src_vocab.pad_token.idx)

            # Obtain predictions
            output = self.model(inputs=inputs, mask_ids=mask_ids)

            # Process prediction and attn weights
            prediction = self.tar_transform(output["predictions"][0].tolist())
            prediction = filter_tokens(prediction)

            # Convert attn_weights to numpy array
            attn_weights = output["attn_weights"].squeeze(0).numpy()

            return {
                "prediction": prediction,
                "attn_weights": attn_weights
            }


if __name__ == '__main__':
    from global_attention.model import Seq2Seq

    # Load the vocabulary
    src_vocab = Vocabulary.from_file("../dataset/eng_vocab.txt")
    tar_vocab = Vocabulary.from_file("../dataset/fra_vocab.txt")

    # Initialize the model
    model = Seq2Seq(enc_vocab_size=src_vocab.size,
                    dec_vocab_size=tar_vocab.size,
                    hidden_size=128,
                    embedding_dim=100,
                    attn_vec_size=128)

    predict = Predictor(model, Tokenizer(end_token=src_vocab.end_token), src_vocab, tar_vocab)
    translation = predict("Does this work?")

    print(translation)
