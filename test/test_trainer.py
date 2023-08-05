# -*- coding: utf-8 -*-

"""This module contains tests for the trainer."""

__author__ = "Mir Sazzat Hossain"

import unittest

import torch

from models.trainer import GNNTrainer
from utils.config import load_config
from utils.data import DataCenter, DataLoader


class TestTrainer(unittest.TestCase):
    """Test the trainer."""

    def setUp(self):
        """Set up the test."""
        self.config = load_config("ustgcn")
        self.adj_path = self.config["data_params"]["adj_path"]
        self.content_path = self.config["data_params"]["content_path"]
        self.num_days = self.config["model_params"]["num_days"]
        self.pred_len = self.config["model_params"]["pred_len"]
        self.train_start = self.config["model_params"]["train_start"]
        self.train_end = self.config["model_params"]["train_end"]
        self.test_start = self.config["model_params"]["test_start"]
        self.test_end = self.config["model_params"]["test_end"]
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
        self.num_gnn_layers = self.config["exp_params"]["num_gnn_layers"]
        self.epochs = self.config["exp_params"]["epochs"]
        self.learning_rate = self.config["exp_params"]["learning_rate"]
        self.device = torch.device(
            self.config["exp_params"]["device"]
        )
        self.work_dir = self.config["logging_params"]["work_dir"]
        self.batch_size = self.config["exp_params"]["batch_size"]

        self.trainer = GNNTrainer(
            self.train_data,
            self.train_label,
            self.test_data,
            self.test_label,
            self.adj_matrix,
            self.num_gnn_layers,
            self.epochs,
            self.learning_rate,
            self.batch_size,
            self.device,
            self.work_dir
        )

    def test_train(self):
        """Test the train method."""
        self.trainer.train()

    def test_test(self):
        """Test the test method."""
        # get the last run directory
        run_dir = self.config["logging_params"]["last_saved_model"]
        self.trainer.test(run_dir)


if __name__ == "__main__":
    unittest.main()
