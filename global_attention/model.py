import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, ht, encoder_outputs):
        # Compute the score between output hidden_state of the decoder and each output hidden state of the encoder
        # h_n[1, 1, 512] x enc_outputs[1, T, 512].T
        score = torch.bmm(encoder_outputs, ht.transpose(2, 1))
        # score[1, T, 1]
        attn_weights = F.softmax(score, dim=1)
        # attn_weights[1, T, 1] x h_n[1, 1, 512]
        context_vec = torch.sum(attn_weights * ht, dim=1)
        # context_vec[1, 1, 512]

        return context_vec, attn_weights


class Decoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, hidden_size, output_size):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_size,
                                      padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_size,
                          hidden_size=hidden_size)
        self.attn = AttentionLayer(input_size=2*hidden_size,
                                   output_size=512)
        self.fc = nn.Sequential(
            nn.Linear(self.attn.input_size, num_embeddings),
            nn.Softmax()
        )

    def forward(self, x, hidden, encoder_outputs):

        emb = self.embedding(inputs)
        outputs, h_n = self.gru(emb)

        # pass hidden state to attention layer
        attn_vec = self.attn(h_n, encoder_outputs)

        return self.fc(attn_vec)


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
        self.decoder = Decoder(num_embeddings=dec_vocab_size,
                               embedding_size=embedding_size,
                               hidden_size=dec_hidden_size,
                               output_size=embedding_size)

    def forward(self, inputs, targets=None):
        encoder_outputs, enc_hidden = self.encoder(inputs)


        start = torch.zeros(1, 1, self.decoder)
        outputs = self.decoder(encoder_outputs, targets)

        return outputs
