import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_sequence


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)

    def forward(self, inputs):
        x_packed = pack_sequence(inputs)
        x = self.gru(x_packed)
        x_paded = pad_sequence(x)

        return x_paded


class AttentionLayer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, inputs):
        return None


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

    def forward(self, inputs):
        pass
