# -*- coding: utf-8 -*-

"""This module contains tests for the Regression model."""

__author__ = "Mir Sazzat Hossain"

import unittest

import torch

from models.regression import Regression


class TestRegression(unittest.TestCase):
    """Test the Regression class."""

    def setUp(self) -> None:
        """Set up the test."""

        self.emb_size = 30
        self.out_size = 30
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.regression = Regression(
            self.emb_size, self.out_size)
        self.regression.to(self.device)

    def test_forward(self) -> None:
        """Test the forward method."""
        embds = torch.rand(95, self.emb_size).to(self.device)

        logits = self.regression(embds)

        self.assertEqual(logits.shape, (95, self.out_size))


if __name__ == "__main__":
    unittest.main()
