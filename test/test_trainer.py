# -*- coding: utf-8 -*-

"""This module contains tests for the trainer."""

__author__ = "Mir Sazzat Hossain"

import os
import unittest

import torch

from models.trainer import GNNTrainer
from utils.data import DataCenter, DataLoader


class TestTrainer(unittest.TestCase):
    """Test the trainer."""

    def setUp(self):
        """Setup the test."""

        self.adj_path = 'data/processed/food_adj.csv'
        self.content_path = 'data/processed/order_matrix.csv'
        self.num_days = 30
        self.pred_len = 7
        self.train_start = 1
        self.train_end = 80
        self.test_start = 80
        self.test_end = 126
        self.data_center = DataCenter()
        self.data_loader = DataLoader(
            self.adj_path,
            self.content_path,
            self.num_days,
            self.pred_len,
            self.train_end,
            self.test_end,
        )
        self.adj_matrix = self.data_center.load_adj(self.adj_path)
        self.train_data, self.train_label = self.data_center.load_data(
            self.content_path,
            self.train_start,
            self.train_end,
            self.num_days,
            self.pred_len,
        )
        self.test_data, self.test_label = self.data_center.load_data(
            self.content_path,
            self.test_start,
            self.test_end,
            self.num_days,
            self.pred_len,
        )
        self.num_gnn_layers = 3
        self.epochs = 5
        self.learning_rate = 0.001
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.work_dir = os.getcwd()

        self.trainer = GNNTrainer(
            self.train_data,
            self.train_label,
            self.test_data,
            self.test_label,
            self.adj_matrix,
            self.num_gnn_layers,
            self.epochs,
            self.learning_rate,
            self.device,
            self.work_dir
        )

    def test_train(self):
        """Test the train method."""
        self.trainer.train()

    def test_test(self):
        """Test the test method."""
        # get the last run directory
        run_dir = os.path.join(
            self.work_dir,
            'logs',
            sorted(
                os.listdir(os.path.join(self.work_dir, 'logs')),
                reverse=True
            )[0]
        )
        self.trainer.test(run_dir)


if __name__ == "__main__":
    unittest.main()
