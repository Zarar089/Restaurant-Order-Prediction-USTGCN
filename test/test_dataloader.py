# -*- coding: utf-8 -*-

"""This module contains tests for the Dataloader."""

__author__ = "Mir Sazzat Hossain"

import unittest

from utils.data import DataCenter, DataLoader


class TestDataLoader(unittest.TestCase):
    """Test the DataLoader class."""

    def setUp(self) -> None:
        """Set up the test."""
        self.adj_path = 'data/processed/co_occurrence_matrix.csv'
        self.content_path = 'data/processed/order_matrix_new.csv'
        self.date_dict_path = 'data/processed/dates_dict.pkl'
        self.num_days = 30
        self.pred_len = 7
        self.train_start = 1
        self.train_end = 552
        self.test_start = 552
        self.test_end = 690
        self.data_center = DataCenter()
        self.data_loader = DataLoader(
            self.adj_path,
            self.content_path,
            self.date_dict_path,
            self.num_days,
            self.pred_len,
            self.train_end,
            self.test_end,
        )

    def check_data(self, data: list, label: list, test: bool = False) -> None:
        """Check the data and label."""
        self.assertEqual(len(data), 546) if not test else \
            self.assertEqual(len(data), 103)
        self.assertEqual(data[0].shape, (1, 128, 30*7))
        self.assertEqual(len(label), 546) if not test else \
            self.assertEqual(len(label), 103)
        self.assertEqual(label[0].shape, (128, 7))

    def test_data_center(self) -> None:
        """Test the DataCenter class."""
        adj = self.data_center.load_adj(self.adj_path)
        self.assertEqual(adj.shape, (128, 128))

        train_data, train_label = self.data_center.load_data(
            self.content_path,
            self.date_dict_path,
            1,
            self.train_end,
            self.num_days,
            self.pred_len,
        )
        self.check_data(train_data, train_label, test=False)

        test_data, test_label = self.data_center.load_data(
            self.content_path,
            self.date_dict_path,
            self.test_start,
            self.test_end,
            self.num_days,
            self.pred_len,
        )
        self.check_data(test_data, test_label, test=True)

    def test_data_loader(self) -> None:
        """Test the DataLoader class."""
        train_data, train_label, test_data, test_label, adj = \
            self.data_loader.load_data()
        self.assertEqual(adj.shape, (128, 128))
        self.check_data(train_data, train_label, test=False)
        self.check_data(test_data, test_label, test=True)


if __name__ == '__main__':
    unittest.main()
