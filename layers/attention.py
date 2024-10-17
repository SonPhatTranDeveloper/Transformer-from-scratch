"""
Author: Son Phat Tran
This file contains the code implementation for single-head and multi-head attention
"""
import torch
import torch.nn as nn

from config import SEQ_LENGTH


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_head: int, p_dropout: float) -> None:
        """
        Initialise the single attention head
        :param d_model: size of the model
        :param d_head: size of the head
        """
        # Call parent's constructor
        super().__init__()

        # Remember the model size and head size
        self.d_model = d_model
        self.d_head = d_head

        # Create key, query, value linear layer
        # to map the input
        self.key = nn.Linear(d_model, d_head, bias=False)
        self.query = nn.Linear(d_model, d_head, bias=False)
        self.value = nn.Linear(d_model, d_head, bias=False)

        # Create the upper triangular matrix
        upper_triangular_matrix = torch.tril(torch.ones(SEQ_LENGTH, SEQ_LENGTH))
        self.register_buffer("upper_triangular", upper_triangular_matrix)

        # Create drop-out layer
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is the input tensor of size (batch_size, seq_length, d_model)
        :param x: the input tensor
        :return: output of size (batch_size, seq_length, d_head)
        """
        # Get the dimension of x
        batch_size, seq_length, d_model = x.shape

        # Create the query, key, value tensor
        # query (batch_size, seq_length, d_head)
        # key (batch_size, seq_length, d_head)
        # value (batch_size, seq_length, d_head)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Calculate the weight matrix
        # weight has size (batch_size, seq_length, seq_length)
        weights = query @ key.transpose(-2, -1) * (self.d_head ** -0.5)

        # Mask the weights with upper triangular matrix
        weights = weights.masked_fill(self.upper_triangular[:seq_length, :seq_length] == 0, float('-inf'))

        # Perform softmax to normalise the weight
        weights = torch.softmax(weights, dim=-1)

        # Dropout
        weights = self.dropout(weights)

        # Perform matrix multiplication with the key
        # the result has size (batch_size, seq_length, d_head)
        return weights @ value


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_head: int, d_head: int, p_dropout: float) -> None:
        """
        Initialise the slower version of multi-headed attention block
        :param d_model: the model's embedding size
        :param num_head: the number of attention head
        :param d_head: the size of each head
        """
        # Call parent's constructor
        super().__init__()

        # Remember the number of heads and head size
        self.num_head = num_head
        self.d_head = d_head

        # Initialise Attention block
        self.heads = nn.ModuleList(
            [SingleHeadAttention(d_model, d_head, p_dropout) for _ in range(num_head)]
        )

        # Initialise the projection and dropout
        self.projection = nn.Linear(d_head * num_head, d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is the input vector of size (batch_size, seq_length, d_model)
        :param x: the input vector
        :return: tensor of size (batch_size, seq_length, d_model)
        """
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

