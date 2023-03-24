# _*_ coding: utf-8 -*-
"""
Converts the data to the format required by the model.

Classes:
    - :py:class:`Data` converts the data to the format required by the model.
"""

__author__ = "Mir Sazzat Hossain"

import pandas as pd


class DataCenter(object):
    """DataCenter class for loading and preprocessing data."""

    def __init__(self, config):
        """
        Initialize the DataCenter class.

        :param config: Config class object
        :type config: Config
        """
        super(DataCenter, self).__init__()
        self.config = config

    def load_adj(self, adj_path):
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

    def load_data(self, content_path):
        """
        Load data from adj_path and content_path.

        :param adj_path: path to adjacency matrix
        :type adj_path: str
        :param content_path: path to content matrix
        :type content_path: str

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
        num_nodes = content.shape[0]
        num_timestamps = content.shape[1]
