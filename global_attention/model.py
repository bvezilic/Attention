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
    def __init__(self):
        super().__init__()

    def forward(self, ht, encoder_outputs):
        """
        :param ht: shape [num_layers=1, batch_size, dec_hidden_size]
        :param encoder_outputs: [batch_size, time_steps, enc_hidden_size]
        """
        # Compute the score between output hidden_state of the decoder and each output hidden state of the encoder
        # Before bmm encoder_outputs[batch_size, time-steps, enc_hidden_size] x ht_[batch_size, dec_hidden_size, num_layers=1]
        score = torch.bmm(encoder_outputs, ht.permute(1, 2, 0))  # Dot product is better for global attention

        # Apply softmax function on scores
        # score[batch_size, time-steps, 1]
        attn_weights = F.softmax(score, dim=1)

        # Compute weighted sum (in paper it's average)
        # attn_weights[batch_size, time_steps, 1] x encoder_outputs[batch_size, time_steps, enc_hidden_state]
        context_vec = torch.sum(attn_weights * encoder_outputs, dim=1)

        # attn_weights[batch_size, time_steps, 1]
        attn_weights = attn_weights.squeeze(2)

        # context_vec[batch_size, enc_hidden_state]
        # attn_weights[batch_size, time_steps]
        return context_vec, attn_weights


class Decoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, hidden_size, attn_vec_size):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attn_vec_size = attn_vec_size

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_size,
                                      padding_idx=0)

        self.gru = nn.GRU(input_size=embedding_size + attn_vec_size,
                          hidden_size=hidden_size,
                          batch_first=True)

        self.attn = AttentionLayer()

        self.fc_attnvec = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.attn_vec_size),
            nn.Tanh()
        )

        self.fc_classifier = nn.Linear(self.attn_vec_size, num_embeddings)

    def forward(self, word_ids, attn_vec, enc_outputs, prev_hidden):
        """
        :param word_ids: shape (batch_size,)
        :param attn_vec: shape (batch_size, attn_size)
        :param enc_outputs: (time_step, batch_size, enc_hidden_size)
        :param prev_hidden: (batch_size, dec_hidden_size)
        :return:
        """
        # Convert index `x` into embedding
        x_emb = self.embedding(word_ids)  # x_emb[batch_size, embedding_size]

        # Concatenate embedding and previous attention vector (ht~)
        word_ids = torch.cat((x_emb, attn_vec), dim=1)  # x[batch_size, embedding_size + attn_size]

        # Obtain hidden state ht
        _, ht = self.gru(word_ids.unsqueeze(0), prev_hidden)  # ht[time_step=1, batch_size, dec_hidden_size]

        # Pass ht to attention layer
        context_vec, attn_weights = self.attn(ht, enc_outputs)  # context_vec[batch_size, enc_hidden_state]
                                                                # attn_weights[batch_size, time_steps, 1]
        # Compute attention vector (ht~)
        merged = torch.cat((context_vec, ht.squeeze(0)), dim=1)  # Remove the num_layers dimension
        attn_vec = self.fc_attnvec(merged)

        # Obtain predictions
        preds = self.fc_classifier(attn_vec)

        return preds, attn_vec, attn_weights, ht


class Seq2Seq(nn.Module):
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_hidden_size, dec_hidden_size, output_size, embedding_size,
                 attn_vec_size):
        super().__init__()
        self.enc_vocab_size = enc_hidden_size
        self.dec_vocab_size = dec_vocab_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.output_size = output_size
        self.attn_vec_size = attn_vec_size
        self.max_sequence = 100

        self.encoder = Encoder(num_embeddings=enc_vocab_size,
                               embedding_size=embedding_size,
                               hidden_size=enc_hidden_size)
        self.decoder = Decoder(num_embeddings=dec_vocab_size,
                               embedding_size=embedding_size,
                               hidden_size=dec_hidden_size,
                               attn_vec_size=attn_vec_size)

    def forward(self, inputs, targets=None):
        batch_size = inputs.size(0)
        enc_outputs, enc_ht = self.encoder(inputs)

        current_ids = torch.zeros(batch_size, dtype=torch.long, device="cuda")  # Start index <PAD>
        current_attn_vec = torch.zeros(batch_size, self.decoder.attn_vec_size, dtype=torch.float, device="cuda")  # Start attn vector
        current_ht = enc_ht  # Take encoder hidden state for initial state for decoder

        attn_weights_list = []
        predictions = []
        num_steps = 0
        while current_ids != 1 and num_steps < self.max_sequence:
            preds, attn_vec, attn_weights, ht = self.decoder(word_ids=current_ids,
                                                             attn_vec=current_attn_vec,
                                                             enc_outputs=enc_outputs,
                                                             prev_hidden=current_ht)
            attn_weights_list.append(attn_weights)
            predictions.append(preds)

            current_ids = torch.argmax(preds, dim=1)
            current_attn_vec = attn_vec
            current_ht = ht

            num_steps += 1

        predictions = torch.stack(predictions)  # shape[time-step, batch_size, dec_vocab_size]
        attn_weights_list = torch.stack(attn_weights_list)  # shape[time-step, batch_size, enc_outputs]

        return predictions, attn_weights_list