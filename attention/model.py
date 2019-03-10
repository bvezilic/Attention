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

        self.attn = nn.Sequential(
            nn.Linear(input_size, output_size, bias=False),
            nn.Tanh()
        )

    def forward(self, h_n, encoder_outputs):
        # h_n[1, 1, 512] x enc_outputs[1, T, 512]
        score = torch.bmm(h_n, encoder_outputs.transpose(2, 1))
        attn_weights = F.softmax(score, dim=2)
        # attn_weights[1, 1, T] x h_n[1, 1, 512]
        c_t = (attn_weights.transpose(2, 1) * h_n).sum(dim=2)  # weighted sum

        x = torch.cat((c_t, h_n), dim=2)
        attn_vec = self.attn(x)

        return attn_vec


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

    def forward(self, inputs, encoder_outputs):
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
        encoder_outputs, h_n = self.encoder(inputs)
        next_word = self.decoder(torch.zeros(1, 1, dtype=torch.long), encoder_outputs)

        if targets:
            pass

        return next_word
