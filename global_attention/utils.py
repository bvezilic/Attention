import json
from typing import Text, Dict, Any

import torch

from model import Seq2Seq


def save_model(path: Text, model: Seq2Seq) -> None:
    torch.save({
        "model_params": {
            "enc_vocab_size": model.enc_vocab_size,
            "dec_vocab_size": model.dec_vocab_size,
            "hidden_size": model.hidden_size,
            "embedding_dim": model.embedding_dim,
            "attn_vec_size": model.attn_size
        },
        "state_dict": model.state_dict()
    }, path)


def load_model(path: Text) -> Seq2Seq:
    restore = torch.load(path)

    model = Seq2Seq(**restore["model_params"])
    model.load_state_dict(restore["state_dict"])
    model.eval()

    return model


def read_params(path: Text) -> Dict[Text, Any]:
    with open(path, "r") as f:
        return json.load(f)

