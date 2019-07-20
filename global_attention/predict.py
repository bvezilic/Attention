import torch
from torchvision.transforms import Compose

from tokenizer import Tokenizer
from transform import ToTokens, ToTensor, ToIndices


class Predictor:
    def __init__(self, model, tokenizer, src_vocab, tar_vocab, device):
        self.model = model
        self.tokenizer = tokenizer
        self.src_vocab = src_vocab
        self.tar_vocab = tar_vocab
        self.device = device

        # Set model to given device
        self.model.to(device)

        # Transforms
        self.transforms = Compose([ToTokens(Tokenizer()),
                                   ToIndices(src_vocab),
                                   ToTensor()])
        self.tar_transform = Compose([ToTo])

    def __call__(self, inputs):
        pass

    def _predict_text(self, text):
        x = self.transforms(text)

        # Set x to device
        x = x.to(self.device)
        x = x.unsqueeze(0)  # Add batch_size
        mask_ids = torch.ones_like(x)

        # Make predictions
        output = self.model(x)

    def _prepare_inputs(self):
        return Compose
