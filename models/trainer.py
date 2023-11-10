# -*- coding: utf-8 -*-

"""
This module a Trainer class for training and evaluating the GNN model.

Classes:
    - py:class:`GNNTrainer`: Trainer class for GNN model.
"""

__author__ = "Mir Sazzat Hossain"

import math
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.gnn import CombinedGNN
from models.regression import Regression
from utils.tools import calculate_foodwise_errors, mae, mape, rmse




class GNNTrainer(object):
    """GNN trainer."""

    def __init__(
            self,
            train_data: torch.Tensor,
            train_labels: torch.Tensor,
            test_data: torch.Tensor,
            test_labels: torch.Tensor,
            adj_matrix: np.ndarray,
            num_gnn_layers: int,
            epochs: int,
            learning_rate: float,
            batch_size: int,
            device: torch.device,
            work_dir: str,
            dish_dict_path: str,
            dates_dict_path: str,
    ) -> None:
        """
        Initialize the GNNTrainer class.

        :param train_data: training data
        :type train_data: torch.Tensor
        :param train_labels: training labels
        :type train_labels: torch.Tensor
        :param test_data: testing data
        :type test_data: torch.Tensor
        :param test_labels: testing labels
        :type test_labels: torch.Tensor
        :param adj_matrix: adjacency matrix
        :type adj_matrix: torch.Tensor
        :param num_gnn_layers: number of GNN layers
        :type num_gnn_layers: int
        :param epochs: number of epochs
        :type epochs: int
        :param learning_rate: learning rate
        :type learning_rate: float
        :param batch_size: batch size
        :type batch_size: int
        :param device: device
        :type device: str
        :param work_dir: working directory
        :type work_dir: str
        :param dish_dict_path: path to the dish dictionary
        :type dish_dict_path: str
        :param dates_dict_path: path to the dates dictionary
        :type dates_dict_path: str
        """
        super(GNNTrainer, self).__init__()
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.adj_matrix = adj_matrix
        self.input_size = self.train_data.shape[-1]
        self.output_size = self.input_size  # may change
        self.num_gnn_layers = num_gnn_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.num_timestamps = self.train_data.shape[1]  # 1 time step each day
        self.pred_len = self.train_labels.shape[-1]  # 7 days
        self.work_dir = work_dir
        self.dish_dict_path = dish_dict_path
        self.dates_dict_path = dates_dict_path

        self.all_nodes = [i for i in range(self.adj_matrix.shape[0])]
        self.node_batch_size = batch_size

        self.train_data = torch.Tensor(self.train_data).to(self.device)
        self.train_labels = torch.Tensor(
            self.train_labels).to(self.device)
        self.test_data = torch.Tensor(self.test_data).to(self.device)
        self.test_labels = torch.Tensor(self.test_labels).to(self.device)
        self.adj_matrix = torch.Tensor(self.adj_matrix).to(self.device)
        self.all_nodes = torch.LongTensor(self.all_nodes).to(self.device)

        self.time_stamp_model = CombinedGNN(
            self.output_size,
            self.adj_matrix,
            self.device,
            1,
            self.num_gnn_layers,
            self.num_timestamps,
            self.input_size
        )

        self.regression_model = Regression(
            self.input_size * self.num_timestamps,
            self.pred_len
        )

        self.time_stamp_model.to(self.device)
        self.regression_model.to(self.device)

        self.log_dir = None
        self.run_version = None
        self.writer = None
        self.run_version = None
        self.stats_df = pd.DataFrame(
            columns=['Epoch', 'Train Loss', 'Validation Loss', 'RMSE Validation', 'MAE Validation',
                     'MAPE Validation', 'Min_RMSE Evaluation', 'Min_MAE Evaluation', 'Min_MAPE Evaluation'])

    def initiate_writer(self) -> None:
        """Initiate the writer."""
        self.log_dir = self.work_dir + "/logs"
        self.run_version = 0

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        else:
            self.run_version = len(os.listdir(self.log_dir))

        self.log_dir = os.path.join(self.log_dir, f"run_{self.run_version}")
        self.writer = SummaryWriter(self.log_dir)

    def train(self) -> None:
        """Train the model."""
        self.initiate_writer()

        min_rmse = float("Inf")
        min_mae = float("Inf")
        min_mape = float("Inf")
        best_test = float("Inf")

        train_loss = torch.tensor(0.0).to(self.device)
        loop = tqdm(range(1, self.epochs))
        for epoch in loop:
            total_timestamp = len(self.train_data)
            indices = torch.randperm(total_timestamp)
            stats = [epoch]

            for index in indices:
                data = self.train_data[index]
                labels = self.train_labels[index]

                models = [self.time_stamp_model, self.regression_model]
                parameters = []
                for model in models:
                    for param in model.parameters():
                        if param.requires_grad:
                            parameters.append(param)

                optimizer = torch.optim.Adam(
                    parameters, lr=self.learning_rate, weight_decay=0)

                optimizer.zero_grad()
                for model in models:
                    model.zero_grad()

                num_node_batches = math.ceil(
                    len(self.all_nodes) / self.node_batch_size)

                node_batch_loss = torch.tensor(0.0).to(self.device)
                for batch in range(num_node_batches):
                    nodes_in_batch = self.all_nodes[
                                     batch * self.node_batch_size:(batch + 1) *
                                                                  self.node_batch_size
                                     ]
                    nodes_in_batch = nodes_in_batch.view(
                        nodes_in_batch.shape[0], 1)
                    labels_in_batch = labels[nodes_in_batch]
                    labels_in_batch = labels_in_batch.view(
                        len(nodes_in_batch), self.pred_len
                    )
                    embeddings = self.time_stamp_model(data)
                    logits = self.regression_model(embeddings)
                    loss = torch.nn.MSELoss()(logits, labels_in_batch)
                    node_batch_loss += loss / (len(nodes_in_batch))

                train_loss += node_batch_loss.item()

                node_batch_loss.backward()
                for model in models:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

                optimizer.zero_grad()
                for model in models:
                    model.zero_grad()

            train_loss = train_loss / len(indices)
            if epoch <= 24 and epoch % 8 == 0:
                self.learning_rate = self.learning_rate / 2
            else:
                self.learning_rate = 0.0001

            loop.set_description(f"Epoch {epoch}/{self.epochs - 1}")
            loop.set_postfix(loss=train_loss.item())

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            stats.append(train_loss.cpu().detach().numpy())

            labels, pred, _eval_loss,_ = self.evaluate()
            _rmse = rmse(labels, pred)
            _mae = mae(labels, pred)
            _mape = mape(labels, pred)

            self.writer.add_scalar("Loss/validation", _eval_loss, epoch)
            stats.append(_eval_loss.cpu().detach().numpy())
            self.writer.add_scalar("RMSE/validation", _rmse, epoch)
            stats.append(_rmse)
            self.writer.add_scalar("MAE/validation", _mae, epoch)
            stats.append(_mae)
            self.writer.add_scalar("MAPE/validation", _mape, epoch)
            stats.append(_mape)

            if _eval_loss < best_test:
                best_test = _eval_loss
                self.save_model()

            min_rmse = min(min_rmse, _rmse)
            min_mae = min(min_mae, _mae)
            min_mape = min(min_mape, _mape)

            self.writer.add_scalar("Evaluation/Min_RMSE", min_rmse, epoch)
            stats.append(min_rmse)
            self.writer.add_scalar("Evaluation/Min_MAE", min_mae, epoch)
            stats.append(min_mae)
            self.writer.add_scalar("Evaluation/Min_MAPE", min_mape, epoch)
            stats.append(min_mape)

            self.stats_df.loc[len(self.stats_df.index)] = stats

        self.writer.close()

        self.stats_df.to_csv(self.log_dir + "/loss_stats.csv", index=False)

    def evaluate(self) -> tuple:
        """
        Evaluate the model.

        :return: tuple of labels, predictions and loss
        :rtype: tuple
        """
        pred = []
        labels = []
        stats= []
        total_timestamp = len(self.test_data)
        indices = torch.randperm(total_timestamp)

        print(self.test_data.shape)

        total_loss = torch.tensor(0.0).to(self.device)
        for index in indices:
            data = self.test_data[index]
            label = self.test_labels[index]

            models = [self.time_stamp_model, self.regression_model]
            parameters = []

            for model in models:
                for param in model.parameters():
                    if param.requires_grad:
                        param.requires_grad = False
                        parameters.append(param)
            embading = self.time_stamp_model(data)
            input_embeddings = embading
            logits = self.regression_model(embading)
            stats.append(self.__get_distribution_stats(input_embeddings))
            loss = torch.nn.MSELoss()(logits, label)
            loss = loss / len(self.all_nodes)
            total_loss += loss.item()
            labels = labels + label.detach().tolist()
            pred = pred + logits.detach().tolist()

            for param in parameters:
                param.requires_grad = True

        total_loss = total_loss / len(indices)

        #for stat in stats:
        #    print(stat)

        return labels, pred, total_loss, stats

    def test(
            self,
            test_start: int,
            model_path: str = None,
            num_days: int = 30,
    ) -> None:
        """
        Test the model.

        :param test_start: start timestamp of the test data
        :type test_start: int
        :param model_path: path to the model
        :type model_path: str
        :param num_days: number of days to predict
        :type num_days: int
        """
        self.load_model(model_path)

        labels, pred, _eval_loss,stats = self.evaluate()

        _rmse, _mse = calculate_foodwise_errors(
            labels, pred, len(self.all_nodes))

        # get dish dictionary.pkl file
        with open(self.dish_dict_path, "rb") as f:
            dish_dict = pickle.load(f)

        # get dish name from dish id
        dish_name = []
        for dish_id in self.all_nodes:
            dish_name.append(
                list(dish_dict.keys())[list(dish_dict.values()).index(dish_id)]
            )

        # get dates from dates_dict.pkl file
        with open(self.dates_dict_path, "rb") as f:
            dates_dict = pickle.load(f)

        # get date from date id
        date = list(dates_dict.keys())
      

        df_pred = pd.DataFrame(columns=["Date"])
        #df_actual = pd.DataFrame(columns=["Date"] + dish_name)

        end = len(date)
        end = len(pred[0]) * (end // len(pred[0]))

        df_pred["Date"] = date[test_start + num_days:end + 1]
        #df_actual["Date"] = date[test_start + num_days:end + 1]

        df_pred.info()

        actuals = []
        preds = []
        for i in range(len(dish_name)):
            prediction_col_name = dish_name[i] + " (Prediction)"
            actual_col_name = dish_name[i] + " (Actual)"

            # Extract the data for the prediction and actual columns
            _food_pred = pred[i::len(dish_name)]
            _food_pred = [j for i in _food_pred for j in i]

            _food_actual = labels[i::len(dish_name)]
            _food_actual = [j for i in _food_actual for j in i]

            # Create Series for prediction and actual columns
            prediction_series = pd.Series(_food_pred, name=prediction_col_name)
            actual_series = pd.Series(_food_actual, name=actual_col_name)

            # Append the Series to the list
            preds.append(prediction_series)
            preds.append(actual_series)
            actuals.append(_food_actual)

        #df_pred['Q1'] = [stat[0] for stat in stats]
        #df_pred['Q3'] = [stat[1] for stat in stats]
        #df_pred['Median'] = [stat[2] for stat in stats]
        #df_pred['Outlier'] = [True if actuals[i] >= stats[i][3] and actuals <= stats[i][4] else False for i in range(0, len(actuals))]
        # save the dataframe
        df_pred.to_csv(self.log_dir + "/prediction.csv", index=False)
        #df_actual.to_csv(self.log_dir + "/actual.csv", index=False)

        # add dish name and save the rmse and mse
        df = pd.DataFrame()
        df["dish_name"] = dish_name
        df["rmse"] = _rmse
        df["mse"] = _mse
        df.to_csv(self.log_dir + "/rmse_mse.csv", index=False)

    def load_model(self, model_path: str) -> None:
        """
        Load the model.

        :param model_path: path to the model
        :type model_path: str

        :return: None

        :raises ValueError: if model_path is None and model is not saved
        """
        if model_path is not None:
            self.time_stamp_model = torch.load(
                os.path.join(model_path, "time_stamp_model.pth")
            )
            self.regression_model = torch.load(
                os.path.join(model_path, "regression_model.pth")
            )
            self.log_dir = model_path
            self.writer = SummaryWriter(self.log_dir)
            self.run_version = int(self.log_dir.split("_")[-1])
        elif self.writer is None:
            self.initiate_writer()
        else:
            raise ValueError("No model path provided")

    def save_model(self) -> None:
        """Save the model."""
        torch.save(
            self.time_stamp_model,
            os.path.join(self.log_dir, "time_stamp_model.pth")
        )
        torch.save(
            self.regression_model,
            os.path.join(self.log_dir, "regression_model.pth")
        )


    def __get_distribution_stats(self,embedding):
        data = embedding.cpu().detach().numpy()
        median = np.percentile(data, 50)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_threshold = q1 - 1.5 * iqr
        upper_threshold = q3 + 1.5 * iqr
        return q1, q3, median, lower_threshold, upper_threshold