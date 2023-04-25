# -*- coding: utf-8 -*-

"""This module contains tests for the SPTempGNN class."""

___author___ = "Mir Sazzat Hossain"

import random
import unittest

import torch

from models.gnn import SPTempGNN


class TestSPTempGNN(unittest.TestCase):
    """Test the SPTempGNN class."""

    def setUp(self) -> None:
        """Setup the test."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.D_temporal = torch.eye(95, 95).to(self.device)
        self.A_temporal = torch.randint(0, 2, (95, 95)).float().to(self.device)
        self.num_timestamps = 1
        self.out_size = 30
        self.total_nodes = 95
        self.num_days = 30
        self.total_data = 74

        self.sp_temp_gnn = SPTempGNN(
            self.D_temporal,
            self.A_temporal,
            self.num_timestamps,
            self.out_size,
            self.total_nodes,
            self.device,
        )

    def test_forward(self) -> None:
        """Test the forward method."""
        historical_raw_features = torch.rand(
            self.total_data, self.num_timestamps,
            self.total_nodes, self.num_days).to(self.device)

        historical_raw_features = historical_raw_features[random.randint(
            0, self.total_data-1)]

        historical_raw_features = \
            historical_raw_features[:, :, :self.num_days].view(
                self.total_nodes, self.num_days)
        output = self.sp_temp_gnn(historical_raw_features)

        self.assertEqual(output.shape, (self.total_nodes, self.out_size),
                         msg="Test failed for SPTempGNN!")


if __name__ == '__main__':
    unittest.main()
