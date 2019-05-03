import torch
import json
from .model import Seq2Seq


def save_model(path: str, model: Seq2Seq):
    torch.save({
        "model_params": {
            "enc_vocab_size": model.enc_vocab_size,
            "dec_vocab_size": model.dec_vocab_size,
            "enc_hidden_size": model.enc_hidden_size,
            "dec_hidden_size": model.dec_hidden_size,
            "output_size": model.output_size,
            "embedding_size": model.embedding_size,
            "attn_size": model.attn_size
        },
        "state_dict": model.state_dict()
    }, path)


def load_model(path):
    restore = torch.load(path)

    model = Seq2Seq(**restore["model_params"])
    model.load_state_dict(restore["state_dict"])
    model.eval()

    return model


def read_params(path):
    with open(path, "r") as f:
        return json.load(f)

