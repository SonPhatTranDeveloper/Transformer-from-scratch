"""
Author: Son Phat Tran
This file contains the code implementation for the model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.positional_encoding import PositionalEncoding
from layers.decoder import DecoderBlock

from config import SEQ_LENGTH, D_MODEL, NUM_HEAD, P_DROPOUT


class LanguageModel(nn.Module):
    def __init__(self, n_vocab: int,
                 n_layer: int,
                 seq_length: int = SEQ_LENGTH,
                 d_model: int = D_MODEL,
                 n_head: int = NUM_HEAD,
                 p_dropout: float = P_DROPOUT):
        """
        Initialise the Language Model
        :param seq_length: the length of the sequence
        :param d_model: the size of the model
        :param n_head: the number of heads
        """
        # Call parent's constructor
        super().__init__()

        # Create embedding layer
        self.embedding = nn.Embedding(n_vocab, d_model)

        # Create positional encoding layer
        self.positional_encoding = PositionalEncoding(seq_length=seq_length, d_model=d_model)

        # Create Decoder blocks
        self.decoder_blocks = nn.Sequential(
            *[DecoderBlock(d_model, n_head, p_dropout) for _ in range(n_layer)]
        )

        # Create final layer norm
        self.layer_norm = nn.LayerNorm(d_model)

        # Predict next token layer
        self.projection = nn.Linear(d_model, n_vocab)

        # Use the same matrix for embedding and projection
        self.embedding.weight = self.projection.weight

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        """
        x is the input vector of size (batch_size, seq_length)
        :param x: the input vector
        :param targets: the target values of size (batch_size, seq_length)
        :return: The loss value
        """
        # Get the size of x
        batch_size, seq_length = x.shape

        # Pass through the embedding layer
        # x has size (batch_size, seq_length, d_model)
        x = self.embedding(x)

        # Pass through the positional encoding layer
        x = self.positional_encoding(x)

        # Pass through decoder blocks
        # x has size (batch_size, seq_length, d_model)
        x = self.decoder_blocks(x)

        # Pass through layer norm
        x = self.layer_norm(x)

        # Pass through language model layer
        x = self.projection(x)

        # Reshape x to get (batch_size * seq_length, n_vocab)
        if targets is not None:
            outputs = x.view(batch_size * seq_length, -1)
            targets = targets.view(batch_size * seq_length)
            loss = F.cross_entropy(outputs, targets)
            return loss, x
        else:
            return None, x
