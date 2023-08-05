# -*- coding: utf-8 -*-

"""This module contains tests for the evaluation tools."""

__author__ = "Mir Sazzat Hossain"

import unittest

from utils.tools import mae, mape, rmse


class TestTools(unittest.TestCase):
    """Test the evaluation tools."""

    def setUp(self):
        """Set up the test."""
        self.y_true = [1, 2, 3, 4, 5]
        self.y_pred = [1.2, 2.2, 3.2, 4.2, 5.2]

    def test_mae(self):
        """Test the mean absolute error."""
        self.assertAlmostEqual(mae(self.y_true, self.y_pred), 0.2)

    def test_mape(self):
        """Test the mean absolute percentage error."""
        self.assertAlmostEqual(
            mape(self.y_true, self.y_pred), 9.13, delta=0.01)

    def test_rmse(self):
        """Test the root mean squared error."""
        self.assertAlmostEqual(rmse(self.y_true, self.y_pred), 0.2)


if __name__ == '__main__':
    unittest.main()
