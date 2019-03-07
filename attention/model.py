import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, num_embeddings, hidden_size, embedding_size):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_size)
        self.gru = nn.GRU(input_size=embedding_size,
                          hidden_size=hidden_size,
                          batch_first=True)

    def forward(self, inputs):
        emb = self.embedding(inputs)
        outputs, h_n = self.gru(emb)

        return outputs, h_n


class AttentionLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.attn = nn.Linear(input_size, output_size, bias=False)

    def forward(self, encoder_outputs):
        # [T, B, enc_hidden_size]
        attn_encoded = self.attn(encoder_outputs)


class Decoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, hidden_size, output_size):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, inputs):
        pass


class Seq2Seq(nn.Module):
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_hidden_size, dec_hidden_size, output_size, embedding_size,
                 attn_size):
        super().__init__()
        self.enc_vocab_size = enc_hidden_size
        self.dec_vocab_size = dec_vocab_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.output_size = output_size
        self.attn_size = attn_size

        self.encoder = Encoder(num_embeddings=enc_vocab_size,
                               embedding_size=embedding_size,
                               hidden_size=enc_hidden_size)

        self.attn_layer = AttentionLayer(input_size=embedding_size, output_size=attn_size)

        self.decoder = Decoder(num_embeddings=dec_vocab_size,
                               embedding_size=embedding_size,
                               hidden_size=dec_hidden_size,
                               output_size=embedding_size)

    def forward(self, *input):
        pass
