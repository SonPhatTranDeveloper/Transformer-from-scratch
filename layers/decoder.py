"""
Author: Son Phat Tran
This file contains the code implementation for the Encoder block
"""
import torch
import torch.nn as nn

from layers.attention import MultiHeadAttention
from layers.feed_forward import FeedForward

from config import D_MODEL, NUM_HEAD, P_DROPOUT


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int = D_MODEL, num_head: int = NUM_HEAD, p_dropout: float = P_DROPOUT) -> None:
        """
        Initialise the Encoder block
        :param d_model: the size of the embedding
        :param num_head: the number of attention head
        """
        # Call parent's constructor
        super().__init__()

        # Get the size of the attention head
        d_head = d_model // num_head

        # Create layer norm
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        # Create multi-head attention
        self.attention = MultiHeadAttention(d_model, num_head, d_head, p_dropout)

        # Create feed-forward layer
        self.feed_forward = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is the input tensor of size (batch_size, seq_length, d_model)
        :param x: the input vector
        :return: output vector of size (batch_size, seq_length, d_model)
        """
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x
