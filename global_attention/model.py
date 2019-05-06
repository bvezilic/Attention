import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, num_embeddings, hidden_size, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.embedding_size = embedding_dim

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_size,
                          batch_first=True)

    def forward(self, inputs):
        """
        :param inputs: [batch_size, time_step]
        :return: outputs: [batch_size, time_step, hidden_size]
                 h_n: [num_layers=1, batch_size, hidden_size]
        """
        emb = self.embedding(inputs)  # emb[B, T, emb_dim]
        outputs, h_n = self.gru(emb)  # outputs [B, T, h_size], h_n[num_layers=1, B, h_size]

        return outputs, h_n


class AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ht, encoder_outputs, mask_ids):
        """
        :param ht: shape [num_layers=1, batch_size, hidden_size]
        :param encoder_outputs: [batch_size, time_step, hidden_size]
        :param mask_ids: [batch_size, time_step]
        :return: context_vec: [batch_size, hidden_size]
                 attn_weights: [batch_size, time_step]
        """
        # Compute the score between output hidden_state of the decoder and each output hidden state of the encoder
        # Before bmm encoder_outputs[B, T, h_size] x ht_[B, h_size, num_layers=1]
        # Dot product is better for global attention
        score = torch.bmm(encoder_outputs, ht.permute(1, 2, 0))  # score[B, T, 1]

        # Assign inf negative score to padded elements
        mask_ids = mask_ids.unsqueeze(2)  # [B, T] -> [B, T, 1]
        n_inf = float("-inf")
        score.masked_fill_(~mask_ids, n_inf)

        # Apply softmax function on scores
        attn_weights = F.softmax(score, dim=1)  # attn_weights[B, T, 1]

        # Compute weighted sum (in paper it's average)
        # attn_weights[B, T, 1] * encoder_outputs[B, T, h_size]
        context_vec = torch.sum(attn_weights * encoder_outputs, dim=1)  # context_vec[B, h_size]
        attn_weights = attn_weights.squeeze(2)  # [B, T, 1] -> [B, T]

        return context_vec, attn_weights


class Decoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, attn_vec_size):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_dim
        self.hidden_size = hidden_size
        self.attn_vec_size = attn_vec_size

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      padding_idx=0)

        self.gru = nn.GRU(input_size=embedding_dim + attn_vec_size,
                          hidden_size=hidden_size,
                          batch_first=True)

        self.attn_layer = AttentionLayer()

        self.fc_attn_vec = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.attn_vec_size),
            nn.Tanh()
        )

        self.fc_classifier = nn.Linear(self.attn_vec_size, num_embeddings)

    def forward(self, word_ids, attn_vec, enc_outputs, prev_hidden, mask_ids):
        """
        :param word_ids: [batch_size]
        :param attn_vec: [batch_size, attn_vec_size]
        :param enc_outputs: [batch_size, time_step, hidden_size]
        :param prev_hidden: [num_layers=1, batch_size, hidden_size]
        :param mask_ids: [batch_size, time_step]
        """
        # Convert words into embeddings
        words_emb = self.embedding(word_ids)  # words_emb[B, emb_dim]

        # Concatenate embedding and previous attention vector (ht~)
        x = torch.cat((words_emb, attn_vec), dim=1)  # x[B, emb_dim + attn_size]

        # Obtain hidden state ht
        x = x.unsqueeze(1)  # [B, T=1, emb_dim + attn_size]
        _, ht = self.gru(x, prev_hidden)  # ht[num_layers=1, B, h_size]

        # Obtain context vector and attention weights
        context_vec, attn_weights = self.attn_layer(ht, enc_outputs, mask_ids)  # context_vec[B, h_size]
                                                                                # attn_weights[B, T, 1]
        # Compute attention vector (ht~)
        ht = ht.squeeze(0)  # [num_layers=1, B, h_size] -> [B, h_size]
        merged = torch.cat((context_vec, ht), dim=1)  # merged[B, 2 * h_size]
        attn_vec = self.fc_attn_vec(merged)  # attn_vec[B, attn_size]

        # Obtain predictions (no activation function)
        logits = self.fc_classifier(attn_vec)  # logits[B, vocab_size]

        return logits, attn_vec, attn_weights, ht


class Seq2Seq(nn.Module):
    def __init__(self,
                 enc_vocab_size,
                 dec_vocab_size,
                 hidden_size,
                 embedding_dim,
                 output_size,
                 attn_vec_size,
                 device="cpu"):
        super().__init__()
        self.enc_vocab_size = enc_vocab_size
        self.dec_vocab_size = dec_vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.output_size = output_size
        self.attn_vec_size = attn_vec_size
        self.max_sequence = 512
        self.device = device

        self.encoder = Encoder(num_embeddings=enc_vocab_size,
                               embedding_dim=embedding_dim,
                               hidden_size=hidden_size)
        self.decoder = Decoder(num_embeddings=dec_vocab_size,
                               embedding_dim=embedding_dim,
                               hidden_size=hidden_size,
                               attn_vec_size=attn_vec_size)

    def forward(self, inputs, targets=None):
        """
        :param inputs: [batch_size, time_step]
        :param targets: (Optional) [batch_size, time_step]
        """
        batch_size = inputs.size(0)
        mask_ids = (inputs != 0)  # Create mask where word_idx=1 and pad=0

        # Obtain outputs from encoder
        enc_outputs, enc_ht = self.encoder(inputs)

        # Initialize tensors for 1st time-step
        current_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)  # Start index <PAD>
        current_attn_vec = torch.zeros(batch_size, self.attn_vec_size, dtype=torch.float, device=self.device)  # Start attn vector
        current_ht = enc_ht  # Take encoder hidden state for initial state for decoder

        attn_weights_list = []
        predictions = []
        num_steps = 0
        while num_steps < self.max_sequence:
            preds, attn_vec, attn_weights, ht = self.decoder(word_ids=current_ids,
                                                             attn_vec=current_attn_vec,
                                                             enc_outputs=enc_outputs,
                                                             prev_hidden=current_ht,
                                                             mask_ids=mask_ids)
            attn_weights_list.append(attn_weights)
            predictions.append(preds)

            current_ids = torch.argmax(preds, dim=1)
            current_attn_vec = attn_vec
            current_ht = ht

            num_steps += 1

        predictions = torch.stack(predictions)  # shape[time-step, batch_size, dec_vocab_size]
        attn_weights_list = torch.stack(attn_weights_list)  # shape[time-step, batch_size, enc_outputs]

        return predictions, attn_weights_list
