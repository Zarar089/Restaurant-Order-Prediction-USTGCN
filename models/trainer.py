# -*- coding: utf-8 -*-

"""
This module a Trainer class for training and evaluating the GNN model.

Classes:
    - py:class:`GNNTrainer`: Trainer class for GNN model.
"""

__author__ = "Mir Sazzat Hossain"


import math
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.gnn import CombinedGNN
from models.regression import Regression
from utils.tools import mae, mape, rmse


class GNNTrainer(object):
    """GNN trainer."""

    def __init__(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        adj_matrix: torch.Tensor,
        num_gnn_layers: int,
        epochs: int,
        learning_rate: float,
        device: str,
        work_dir: str,
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
        :param device: device
        :type device: str
        :param work_dir: working directory
        :type work_dir: str
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

        self.all_nodes = [i for i in range(self.adj_matrix.shape[0])]
        self.node_batch_size = 200  # will be changed later

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
        """
        Train the model.

        :param evaluate: whether to evaluate the model
        :type evaluate: bool
        """
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
                    node_batch_loss += loss/(len(nodes_in_batch))

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

            loop.set_description(f"Epoch {epoch}/{self.epochs-1}")
            loop.set_postfix(loss=train_loss.item())

            self.writer.add_scalar("Loss/train", train_loss, epoch)

            _rmse, _mae, _mape, _eval_loss = self.evaluate()

            self.writer.add_scalar("Loss/validation", _eval_loss, epoch)
            self.writer.add_scalar("RMSE/validation", _rmse, epoch)
            self.writer.add_scalar("MAE/validation", _mae, epoch)
            self.writer.add_scalar("MAPE/validation", _mape, epoch)

            if _eval_loss < best_test:
                best_test = _eval_loss
                self.save_model()

            min_rmse = min(min_rmse, _rmse)
            min_mae = min(min_mae, _mae)
            min_mape = min(min_mape, _mape)

            self.writer.add_scalar("Evaluation/Min_RMSE", min_rmse, epoch)
            self.writer.add_scalar("Evaluation/Min_MAE", min_mae, epoch)
            self.writer.add_scalar("Evaluation/Min_MAPE", min_mape, epoch)

        self.writer.close()

    def evaluate(self) -> tuple:
        """
        Evaluate the model.

        :return: tuple of rmse, mae, mape and loss
        :rtype: tuple
        """
        pred = []
        labels = []
        total_timestamp = len(self.test_data)
        indices = torch.randperm(total_timestamp)

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
            logits = self.regression_model(embading)
            loss = torch.nn.MSELoss()(logits, label)
            loss = loss/len(self.all_nodes)
            total_loss += loss.item()

            labels = labels + label.detach().tolist()
            pred = pred + logits.detach().tolist()

            for param in parameters:
                param.requires_grad = True

        total_loss = total_loss / len(indices)

        _rmse = rmse(labels, pred)
        _mae = mae(labels, pred)
        _mape = mape(labels, pred)

        return _rmse, _mae, _mape, total_loss

    def test(self, model_path=None) -> None:
        """
        Test the model.

        :param model_path: path to the model
        :type model_path: str
        """
        self.load_model(model_path)
        _rmse, _mae, _mape, _eval_loss = self.evaluate()

        self.writer.add_scalar("RMSE/test", _rmse, self.run_version)
        self.writer.add_scalar("MAE/test", _mae, self.run_version)
        self.writer.add_scalar("MAPE/test", _mape, self.run_version)

        self.writer.close()

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
        """
        Save the model.

        :param epoch: epoch number
        """
        torch.save(
            self.time_stamp_model,
            os.path.join(self.log_dir, "time_stamp_model.pth")
        )
        torch.save(
            self.regression_model,
            os.path.join(self.log_dir, "regression_model.pth")
        )
