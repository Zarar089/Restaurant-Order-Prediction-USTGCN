# _*_ coding: utf-8 -*-
"""
Converts the data to the format required by the model.

Classes:
    - :py:class:`Data` converts the data to the format required by the model.
"""

__author__ = "Mir Sazzat Hossain"

import numpy as np
import pandas as pd


class DataCenter(object):
    """DataCenter class for loading and preprocessing data."""

    def load_adj(self, adj_path: str) -> np.ndarray:
        """
        Load adjacency matrix from adj_path.

        :param adj_path: path to adjacency matrix
        :type adj_path: str

        :return: adjacency matrix
        :rtype: numpy.ndarray

        :raises FileNotFoundError: if adj_path is not found
        """
        try:
            adj = pd.read_csv(adj_path, header=None).values
        except FileNotFoundError:
            print(f'File {adj_path} not found')
            exit(1)

        return adj

    def load_data(
        self,
        content_path: str,
        start_day: int,
        end_day: int,
        num_days: int,
        pred_len: int,
    ) -> tuple:
        """
        Load data from adj_path and content_path.

        :param adj_path: path to adjacency matrix
        :type adj_path: str
        :param content_path: path to content matrix
        :type content_path: str
        :param start_day: start day
        :type start_day: int
        :param end_day: end day
        :type end_day: int
        :param num_days: number of days
        :type num_days: int
        :param pred_len: prediction length
        :type pred_len: int

        :return: adjacency matrix, content matrix
        :rtype: numpy.ndarray, numpy.ndarray

        :raises FileNotFoundError: if adj_path or content_path is not found
        """
        try:
            content = pd.read_csv(content_path, header=None).values
        except FileNotFoundError:
            print(f'File {content_path} not found')
            exit(1)

        content = content.T

        timestamp_data = []
        label_data = []

        start_day = start_day - 1

        for i in range(start_day, end_day+1-pred_len):
            a = content[:, i:i+num_days]
            if a.shape[1] < num_days:
                continue
            a = a.reshape(1, a.shape[0], a.shape[1])
            timestamp_data.append(a)
            label_data.append(content[:, i+num_days:i+num_days+pred_len])

        return timestamp_data, label_data


class DataLoader(object):
    """DataLoader class for loading and preprocessing data."""

    def __init__(
        self,
        adj_path: str,
        content_path: str,
        num_days: int,
        pred_len: int,
        train_end: int,
        test_end: int,
    ) -> None:
        """
        Initialize the DataLoader class.

        :param adj_path: path to adjacency matrix
        :type adj_path: str
        :param content_path: path to content matrix
        :type content_path: str
        :param num_days: number of days
        :type num_days: int
        :param pred_len: prediction length
        :type pred_len: int
        :param train_end: end of training data
        :type train_end: int
        :param test_end: end of testing data
        :type test_end: int
        """
        super(DataLoader, self).__init__()
        self.data_center = DataCenter()

        self.train_start = 1
        self.train_end = train_end
        self.test_start = self.train_end
        self.test_end = test_end
        self.adj_path = adj_path
        self.content_path = content_path
        self.num_days = num_days
        self.pred_len = pred_len

    def load_data(self) -> tuple:
        """
        Load data from adj_path and content_path.

        :return: adjacency matrix, train data & label, test & label
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray

        :raises FileNotFoundError: if adj_path or content_path is not found
        """
        adj = self.data_center.load_adj(self.adj_path)

        train_data, train_label = self.data_center.load_data(
            self.content_path,
            self.train_start,
            self.train_end,
            self.num_days,
            self.pred_len,
        )

        test_data, test_label = self.data_center.load_data(
            self.content_path,
            self.test_start,
            self.test_end,
            self.num_days,
            self.pred_len,
        )

        return train_data, train_label, test_data, test_label, adj
