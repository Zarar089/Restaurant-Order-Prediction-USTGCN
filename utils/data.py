# _*_ coding: utf-8 -*-
"""
Converts the data to the format required by the model.

Classes:
    - :py:class:`Data` converts the data to the format required by the model.
"""

__author__ = "Mir Sazzat Hossain"

import pickle

import numpy as np
import pandas as pd
import torch


class DataCenter(object):
    """DataCenter class for loading and preprocessing data."""

    @staticmethod
    def load_adj(adj_path: str) -> np.ndarray:
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

    @staticmethod
    def load_data(
        content_path: str,
        date_file: str,
        start_day: int,
        end_day: int,
        num_days: int,
        pred_len: int,
        stride: int = 1,
    ) -> tuple:
        """
        Load data from adj_path and content_path.

        :param content_path: path to content matrix
        :type content_path: str
        :param date_file: date file path
        :type date_file: str
        :param start_day: start day
        :type start_day: int
        :param end_day: end day
        :type end_day: int
        :param num_days: number of days
        :type num_days: int
        :param pred_len: prediction length
        :type pred_len: int
        :param stride: stride
        :type stride: int

        :return: adjacency matrix, content matrix
        :rtype: numpy.ndarray, numpy.ndarray

        :raises FileNotFoundError: if adj_path or content_path is not found
        """
        try:
            content = pd.read_csv(content_path, header=None).values
        except FileNotFoundError:
            print(f'File {content_path} not found')
            exit(1)

        try:
            with open(date_file, 'rb') as f:
                dates_dict = pickle.load(f)
                dates_dict = list(dates_dict.keys())
        except FileNotFoundError:
            print(f'File {date_file} not found')
            exit(1)

        # covert dates to index based on day of week
        # 1 is Monday, 7 is Sunday
        dates_dict = pd.to_datetime(dates_dict)
        date_indices = dates_dict.dayofweek

        content = content.T

        timestamp_data = []
        label_data = []

        start_day = start_day - 1

        for i in range(start_day, end_day+1-pred_len, stride):
            data = content[:, i:i+num_days]

            # shape of data is (num_nodes, num_days)
            # and in data[node, day] the number of that item ordered on that
            # day is stored. We need to add the day of week as a feature
            # to the data. So, we need to add a 7 vector instead of a scalar
            # to each node. make it (num_nodes, num_days*7)
            # e.g. if date index is 1, then [0, data[node, day], 0, 0, 0, 0, 0]

            new_data = np.zeros((data.shape[0], data.shape[1]*7))
            for j in range(data.shape[0]):
                for k in range(data.shape[1]):
                    new_data[j, k*7+date_indices[i+k]] = data[j, k]

            data = new_data

            label = content[:, i+num_days:i+num_days+pred_len]
            if data.shape[1] < num_days or label.shape[1] < pred_len:
                continue
            data = data.reshape(1, data.shape[0], data.shape[1])
            timestamp_data.append(data)
            label_data.append(label)

        # convert to torch tensor
        timestamp_data = torch.from_numpy(
            np.array(timestamp_data, dtype=np.float32))
        label_data = torch.from_numpy(
            np.array(label_data, dtype=np.float32))

        return timestamp_data, label_data


class DataLoader(object):
    """DataLoader class for loading and preprocessing data."""

    def __init__(
        self,
        adj_path: str,
        content_path: str,
        date_dict_path: str,
        num_days: int,
        pred_len: int,
        train_end: int,
        test_end: int,
        train_stride: int = 1,
        test_stride: int = 1,
    ) -> None:
        """
        Initialize the DataLoader class.

        :param adj_path: path to adjacency matrix
        :type adj_path: str
        :param content_path: path to content matrix
        :type content_path: str
        :param date_dict_path: path to date dictionary
        :type date_dict_path: str
        :param num_days: number of days
        :type num_days: int
        :param pred_len: prediction length
        :type pred_len: int
        :param train_end: end of training data
        :type train_end: int
        :param test_end: end of testing data
        :type test_end: int
        :param train_stride: stride for training data
        :type train_stride: int
        :param test_stride: stride for testing data
        :type test_stride: int
        """
        super(DataLoader, self).__init__()
        self.data_center = DataCenter()

        self.train_start = 1
        self.train_end = train_end
        self.test_start = self.train_end
        self.test_end = test_end
        self.adj_path = adj_path
        self.content_path = content_path
        self.date_path = date_dict_path
        self.num_days = num_days
        self.pred_len = pred_len
        self.train_stride = train_stride
        self.test_stride = test_stride

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
            self.date_path,
            self.train_start,
            self.train_end,
            self.num_days,
            self.pred_len,
            self.train_stride,
        )

        test_data, test_label = self.data_center.load_data(
            self.content_path,
            self.date_path,
            self.test_start,
            self.test_end,
            self.num_days,
            self.pred_len,
            self.test_stride,
        )

        return train_data, train_label, test_data, test_label, adj
