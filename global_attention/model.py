from typing import List, Text

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


class Encoder(nn.Module):
    def __init__(self, num_embeddings: int, hidden_size: int, embedding_dim: int):
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
        Args:
            inputs: Words indices from source (language) vocabulary
                shape=[batch_size, time_steps]

        Returns:
            outputs: Output (hidden states) from RNN at all time steps
                shape=[batch_size, time_steps, hidden_size]
            h_n: Last hidden state at `max_time_steps`
                shape=[num_layers=1, batch_size, hidden_size]
        """
        emb = self.embedding(inputs)        # emb[B, T, emb_dim]
        outputs, h_n = self.gru(emb)        # outputs[B, T, h_size]
                                            # h_n[num_layers=1, B, h_size]
        return outputs, h_n


class AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ht, encoder_outputs, mask_ids):
        """
        Args:
            ht: Hidden state of decoder RNN
                shape=[num_layers=1, batch_size, hidden_size]
            encoder_outputs: Output (hidden states) from encoder RNN at all time steps
                shape=[batch_size, time_step, hidden_size]
            mask_ids: Boolean tensor for masking padded elements
                shape=[batch_size, time_step]

        Returns:
            contex_vec: Sum of attention weights and encoder outputs
                shape=[batch_size, hidden_size]
            attn_weights: Softmax score between decoder hidden state and encoder outputs
                shape=[batch_size, time_step]
        """
        # Before bmm encoder_outputs[B, T, h_size] @ ht_[B, h_size, num_layers=1]
        # Dot product is better for global attention
        score = torch.bmm(encoder_outputs, ht.permute(1, 2, 0))         # score[B, T, 1]

        # Assign inf negative score to padded elements
        mask_ids = mask_ids.unsqueeze(2)                                # mask_ids[B, T] -> [B, T, 1]
        n_inf = float("-inf")
        score.masked_fill_(~mask_ids, n_inf)                            # score[B, T, 1]

        # Apply softmax function on scores
        attn_weights = F.softmax(score, dim=1)                          # attn_weights[B, T, 1]

        # Compute weighted sum (in paper it's average)
        # attn_weights[B, T, 1] * encoder_outputs[B, T, h_size]
        context_vec = torch.sum(attn_weights * encoder_outputs, dim=1)  # context_vec[B, h_size]
        attn_weights = attn_weights.squeeze(2)                          # attn_weights[B, T, 1] -> [B, T]

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
        Args:
            word_ids: Words indices from target (language) vocabulary
                shape=[batch_size]
            attn_vec: Attention vector from previous time-step
                shape=[batch_size, attn_vec_size]
            enc_outputs: Output of encoder RNN
                shape=[batch_size, time_steps, hidden_size]
            prev_hidden: Previous hidden state of the decoder RNN
                shape=[num_layers=1, batch_size, hidden_size]
            mask_ids: Boolean tensor for masking padded elements
                shape=[batch_size, time_step]

        Returns:
            output:
                logits: Classification output from decoder at specific time-step
                    shape=[batch_size, vocab_size]
                attn_vec: Computed attention vector from context vector and hidden state of decoder RNN
                    shape=[batch_size, attn_size]
                attn_weights: Attention weights to encoder outputs
                    shape=[batch_size, time_steps, 1]
                hidden_state: Output hidden state of decoder RNN
                    shape=[num_layers=1, batch_size, hidden_size]
        """
        # Convert words into embeddings
        words_emb = self.embedding(word_ids)                                    # words_emb[B, emb_dim]

        # Concatenate embedding and previous attention vector (ht~)
        x = torch.cat((words_emb, attn_vec), dim=1)                             # x[B, emb_dim + attn_size]

        # Add time_step dimension
        x = x.unsqueeze(1)                                                      # x[B, T=1, emb_dim + attn_size]

        # Obtain hidden state ht
        _, ht = self.gru(x, prev_hidden)                                        # ht[num_layers=1, B, h_size]

        # Obtain context vector and attention weights
        context_vec, attn_weights = self.attn_layer(ht, enc_outputs, mask_ids)  # context_vec[B, h_size]
                                                                                # attn_weights[B, T]
        # Compute attention vector (ht~)
        ht_ = ht.squeeze(0)                                                     # [1, B, h_size] -> [B, h_size]
        merged = torch.cat((context_vec, ht_), dim=1)                           # merged[B, 2 * h_size]
        attn_vec = self.fc_attn_vec(merged)                                     # attn_vec[B, attn_size]

        # Obtain predictions (no activation function)
        logits = self.fc_classifier(attn_vec)                                   # logits[B, vocab_size]

        return {
            "logits": logits,
            "attn_vec": attn_vec,
            "attn_weights": attn_weights,
            "hidden_state": ht
        }


class Seq2Seq(BaseModel):
    def __init__(self,
                 enc_vocab_size,
                 dec_vocab_size,
                 hidden_size,
                 embedding_dim,
                 attn_vec_size,
                 max_len=128,
                 device="cpu"):
        super().__init__()
        self.enc_vocab_size = enc_vocab_size  # Output size of encoder
        self.dec_vocab_size = dec_vocab_size  # Output size of decoder
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.attn_vec_size = attn_vec_size
        self.max_len = max_len
        self.device = device

        self.encoder = Encoder(num_embeddings=enc_vocab_size,
                               embedding_dim=embedding_dim,
                               hidden_size=hidden_size)
        self.decoder = Decoder(num_embeddings=dec_vocab_size,
                               embedding_dim=embedding_dim,
                               hidden_size=hidden_size,
                               attn_vec_size=attn_vec_size)

    @property
    def attribute_names(self) -> List[Text]:
        return ["enc_vocab_size",
                "dec_vocab_size",
                "hidden_size",
                "embedding_dim",
                "attn_vec_size",
                "max_len",
                "device"]

    def forward(self, inputs, mask_ids, targets=None):
        """
        Args:
            inputs: Word indices from source vocabulary
                shape=[batch_size, max_time_steps_inputs]
            mask_ids: Tensor with shape of inputs for variable sized sequences
                shape=[batch_size, max_time_steps_inputs]
            targets: (Optional) Word indices from target vocabulary. Used for teacher training.
                shape=[batch_size, max_time_steps_targets]
        Returns:
            output:
                logits: Raw models predictions
                    shape=[batch_size, dec_time_steps, tar_vocab_size]
                attn_weights: Attention score
                    shape=[batch_size, dec_time_steps, enc_time_steps]
                predictions: Word indices predictions
                    shape=[batch_size, dec_time_steps]
        """
        batch_size = inputs.size(0)

        # Obtain outputs from encoder
        (enc_outputs, enc_ht) = self.encoder(inputs)

        # Initialize zero-vectors inputs for 1-time step for decoder
        (input_ids, input_attn_vec) = self._init_start_input(batch_size=batch_size)
        input_ht = enc_ht  # Take encoder hidden state for initial state for decoder

        # Consolidate all inputs for prediction
        decoder_inputs = {
            "input_ids": input_ids,
            "input_attn_vec": input_attn_vec,
            "input_ht": input_ht,
            "mask_ids": mask_ids,
            "enc_outputs": enc_outputs
        }

        if targets is not None:
            output = self.prediction_with_teacher(targets, **decoder_inputs)
        else:
            output = self.prediction_without_teacher(**decoder_inputs)

        # Modify output
        output["logits"] = torch.stack(output["logits"])                    # logits[T, B, dec_vocab_size]
        output["logits"] = output["logits"].transpose(1, 0)                 # logits[T, B, *] -> [B, T, *]
        output["attn_weights"] = torch.stack(output["attn_weights"])        # attn_weights[T, B, enc_outputs]
        output["attn_weights"] = output["attn_weights"].transpose(1, 0)     # attn_weights[T, B, *] -> [B, T, *]
        output["predictions"] = torch.argmax(output["logits"], dim=2)       # predictions[B, T]

        return output

    def _init_start_input(self, batch_size):
        zero_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)  # Start index <PAD>
        zero_attn_vec = torch.zeros(batch_size, self.decoder.attn_vec_size, dtype=torch.float, device=self.device)

        return zero_ids, zero_attn_vec

    def prediction_with_teacher(self, targets, **kwargs):
        """
        Run prediction where at each time step an input from targets is taken as new input.

        Args:
            targets: Target word indices from target vocabulary.
                shape=[batch_size, max_time_steps_targets]
            kwargs:
                input_ids: Current input word indices from target vocabulary.
                    shape=[batch_size]
                input_attn_vec: Previously computed attention vector
                    shape=[batch_size, attention_size]
                input_ht: Previous computed hidden state from decoder RNN
                    shape=[num_layers=1, batch_size, hidden_size]
                mask_ids: Tensor with shape of inputs (of encoder) for variable sized sequences
                    shape=[batch_size, max_time_steps_inputs]
                enc_outputs: Output of encoder RNN
                    shape=[batch_size, time_steps, hidden_size]

        Returns:
            output:
                attn_weights: List of tensors[batch_size, enc_max_time_steps] of size 'dec_max_time_steps'
                logits: List of tensors[batch_size, target_vocab_size] of size 'dec_max_time_steps'
        """
        # Initialize output for seq2seq
        output = {
            "attn_weights": [],
            "logits": []
        }

        # Retrieve input tensors
        input_ids = kwargs.get("input_ids")
        input_attn_vec = kwargs.get("input_attn_vec")
        input_ht = kwargs.get("input_ht")
        mask_ids = kwargs.get("mask_ids")
        enc_outputs = kwargs.get("enc_outputs")

        # Iterate over time-steps
        for i in range(targets.size(1)):
            dec_output = self.decoder(word_ids=input_ids,
                                      attn_vec=input_attn_vec,
                                      prev_hidden=input_ht,
                                      mask_ids=mask_ids,
                                      enc_outputs=enc_outputs)

            # Store decoder outputs (attention weights and raw predictions)
            output["attn_weights"].append(dec_output["attn_weights"])
            output["logits"].append(dec_output["logits"])

            # Update inputs for next time step
            input_ids = targets[:, i]
            input_attn_vec = dec_output["attn_vec"]
            input_ht = dec_output["hidden_state"]

        return output

    def prediction_without_teacher(self, **kwargs):
        # Initialize output for seq2seq
        output = {
            "attn_weights": [],
            "logits": []
        }

        # Retrieve input tensors
        input_ids = kwargs.get("input_ids")
        input_attn_vec = kwargs.get("input_attn_vec")
        input_ht = kwargs.get("input_ht")
        mask_ids = kwargs.get("mask_ids")
        enc_outputs = kwargs.get("enc_outputs")

        step = 0
        # Iterate over time-steps
        while step < self.max_len:
            dec_output = self.decoder(word_ids=input_ids,
                                      attn_vec=input_attn_vec,
                                      prev_hidden=input_ht,
                                      mask_ids=mask_ids,
                                      enc_outputs=enc_outputs)

            # Store decoder outputs (attention weights and raw predictions)
            output["attn_weights"].append(dec_output["attn_weights"])
            output["logits"].append(dec_output["logits"])

            # Update inputs for next time step
            input_ids = dec_output["logits"].argmax(-1)
            input_attn_vec = dec_output["attn_vec"]
            input_ht = dec_output["hidden_state"]

            # Increment to next step
            step += 1

        return output
