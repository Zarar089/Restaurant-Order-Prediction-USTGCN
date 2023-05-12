# -*- coding: utf-8 -*-

"""This module contains tests for the Dataloader."""

__author__ = "Mir Sazzat Hossain"

import unittest

from utils.data import DataCenter, DataLoader


class TestDataLoader(unittest.TestCase):
    """Test the DataLoader class."""

    def setUp(self) -> None:
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

    def test_data_center(self) -> None:
        """Test the DataCenter class."""
        adj = self.data_center.load_adj(self.adj_path)
        self.assertEqual(adj.shape, (95, 95))

        train_data, train_label = self.data_center.load_data(
            self.content_path,
            1,
            self.train_end,
            self.num_days,
            self.pred_len,
        )
        self.assertEqual(len(train_data), 74)
        self.assertEqual(train_data[0].shape, (1, 95, 30))
        self.assertEqual(len(train_label), 74)
        self.assertEqual(train_label[0].shape, (95, 7))

        test_data, test_label = self.data_center.load_data(
            self.content_path,
            self.test_start,
            self.test_end,
            self.num_days,
            self.pred_len,
        )
        self.assertEqual(len(test_data), 11)
        self.assertEqual(test_data[0].shape, (1, 95, 30))
        self.assertEqual(len(test_label), 11)
        self.assertEqual(test_label[0].shape, (95, 7))

    def test_data_loader(self) -> None:
        """Test the DataLoader class."""
        train_data, train_label, test_data, test_label, adj = \
            self.data_loader.load_data()
        self.assertEqual(adj.shape, (95, 95))
        self.assertEqual(len(train_data), 74)
        self.assertEqual(train_data[0].shape, (1, 95, 30))
        self.assertEqual(len(train_label), 74)
        self.assertEqual(train_label[0].shape, (95, 7))
        self.assertEqual(len(test_data), 11)
        self.assertEqual(test_data[0].shape, (1, 95, 30))
        self.assertEqual(len(test_label), 11)
        self.assertEqual(test_label[0].shape, (95, 7))


if __name__ == '__main__':
    unittest.main()
