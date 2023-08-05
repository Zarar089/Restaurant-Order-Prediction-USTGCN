# -*- coding: utf-8 -*-

"""This module contains tests for the SPTempGNN class."""

__author__ = "Mir Sazzat Hossain"


import random
import unittest

import torch

from models.gnn import CombinedGNN


class TestCombinedGNN(unittest.TestCase):
    """Test the CombinedGNN class."""

    def setUp(self) -> None:
        """Set up the test."""

        self.out_size = 30
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.adj_matrix = torch.randint(0, 2, (95, 95)).float().to(self.device)
        self.start_time = 1
        self.num_gnn_layers = 3
        self.num_timestamps = 1
        self.num_days = 30
        self.total_data = 74
        self.total_nodes = 95

        self.combined_gnn = CombinedGNN(
            self.out_size,
            self.adj_matrix,
            self.device,
            self.start_time,
            self.num_gnn_layers,
            self.num_timestamps,
            self.num_days,
        )
        self.combined_gnn.to(self.device)

    def test_forward(self) -> None:
        """Test the forward method."""
        historical_raw_features = torch.rand(
            self.total_data, self.num_timestamps,
            self.total_nodes, self.num_days).to(self.device)

        historical_raw_features = historical_raw_features[random.randint(
            0, self.total_data-1)]

        embds = self.combined_gnn(historical_raw_features)

        self.assertEqual(embds.shape, (self.total_nodes, self.out_size))


if __name__ == '__main__':
    unittest.main()
