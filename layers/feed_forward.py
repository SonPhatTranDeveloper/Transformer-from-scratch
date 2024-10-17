"""
Author: Son Phat Tran
This file contains the code implementation for the FeedForward layer in Decoder block
"""
import torch
import torch.nn as nn

from config import D_MODEL, P_DROPOUT


class FeedForward(nn.Module):
    def __init__(self, d_model: int = D_MODEL, p_dropout: float = P_DROPOUT) -> None:
        """
        Initialise the feedforward network in the Decoder block
        :param d_model: the size of the model
        :param d_hidden: the size of the hidden layer
        """
        # Call parent's constructor
        super().__init__()

        # Create two linear layers
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p_dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is the tensor of shape (batch_size, seq_length, d_model)
        :param x: the input tensor
        :return: the output tensor
        """
        return self.feed_forward(x)
