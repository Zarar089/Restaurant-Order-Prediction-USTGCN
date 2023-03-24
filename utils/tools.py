# _*_ coding: utf-8 -*-
"""
This module defines some useful functions.

Functions:
    - :py:function:`rmse_loss` calculates root mean squared error loss.
    - :py:function:`mape` calculates mean absolute percentage error.
"""

__author__ = "Mir Sazzat Hossain"

import numpy as np
import torch


def rmse_loss(y_true, y_pred):
    """
    Calculate root mean squared error loss.

    :param y_true: true values of shape (batch_size, out_size)
    :type y_true: numpy.ndarray
    :param y_pred: predicted values of shape (batch_size, out_size)
    :type y_pred: numpy.ndarray

    :return: RMSE loss
    :rtype: float
    """
    y_true = torch.FloatTensor(y_true)
    y_pred = torch.FloatTensor(y_pred)

    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


def mape(y_true, y_pred):
    """
    Calculate mean absolute percentage error.

    :param y_true: true values of shape (batch_size, out_size)
    :type y_true: numpy.ndarray
    :param y_pred: predicted values of shape (batch_size, out_size)
    :type y_pred: numpy.ndarray
    :return: MAPE
    :rtype: float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
