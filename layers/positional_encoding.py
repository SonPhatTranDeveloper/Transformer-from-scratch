"""
Author: Son Phat Tran
This file contains the code implementation for Positional Encoding of Transformers
"""
import math

import torch
import torch.nn as nn

from config import SEQ_LENGTH, D_MODEL


class PositionalEncoding(nn.Module):
    def __init__(self, seq_length: int = SEQ_LENGTH, d_model: int = D_MODEL) -> None:
        # Call parent's constructor
        super().__init__()

        # Remember the sequence length and model size
        self.seq_length = seq_length
        self.d_model = d_model

        # Create positional encoding matrix
        pe = torch.zeros(size=(seq_length, d_model))

        # Create the position row vector
        # Position is a vector of size (seq_length): (0, 1, 2, ..., seq_length - 1)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)

        # Create the dividing terms
        # div_term is a vector of size (d_model/2): (10,000^(0/d_model), 10,000^(2/d_model), 10,000^(4/d_model), ...)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10_000) / d_model))

        # Create the pe matrix
        # and un-squeeze to get (1, seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register Positional Encoding matrix as buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        X is a matrix of tensor (batch_size, seq_length, d_model)
        :param x: the input tensor
        :return: x + positional encoding
        """
        return x + self.pe[:, :x.shape[1], :]
