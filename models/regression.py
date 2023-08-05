# -*- coding: utf-8 -*-

"""
This module defines a simple MLP model for regression.

Classes:
    - py:class:`Regression`: A simple MLP model for regression.
"""

__author__ = "Mir Sazzat Hossain"

import torch
import torch.nn as nn


class Regression(nn.Module):
    """A simple linear regression model."""

    def __init__(self, emb_size: int, out_size: int) -> None:
        """
        Initialize the Regression class.

        :param emb_size: embedding size
        :type emb_size: int
        :param out_size: output size
        :type out_size: int
        """
        super(Regression, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, out_size),
            nn.ReLU()
        )
        self.init_params()

    def init_params(self) -> None:
        """Initialize parameters."""
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, embds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param embds: embeddings of shape (batch_size, emb_size)
        :type embds: torch.Tensor

        :return: logits of shape (batch_size, out_size)
        :rtype: torch.Tensor
        """
        logits = self.layer(embds)
        return logits
