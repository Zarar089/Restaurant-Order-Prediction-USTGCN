# -*- coding: utf-8 -*-

"""
Training script for USTGCN order prediction model.

This script trains the USTGCN model on the order prediction task.
"""

__author__ = "Mir Sazzat Hossain"

import argparse
import random

import numpy as np
import torch

from models.trainer import GNNTrainer
from utils.config import load_config
from utils.data import DataCenter, DataLoader


def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility.

    :param seed: The seed value.
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed value for reproducibility.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ustgcn",
        help="The config to use.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="The mode to use.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    config = load_config(args.config)

    data_center = DataCenter()
    data_loader = DataLoader(
        config["data_params"]["adj_path"],
        config["data_params"]["content_path"],
        config["data_params"]["dates_dict_path"],
        config["model_params"]["num_days"],
        config["model_params"]["pred_len"],
        config["model_params"]["train_end"],
        config["model_params"]["test_end"],
        config["model_params"]["train_stride"],
        config["model_params"]["test_stride"],
    )

    adj_matrix = data_center.load_adj(
        config["data_params"]["adj_path"]
    )
    train_data, train_label = data_center.load_data(
        config["data_params"]["content_path"],
        config["data_params"]["dates_dict_path"],
        config["model_params"]["train_start"],
        config["model_params"]["train_end"],
        config["model_params"]["num_days"],
        config["model_params"]["pred_len"],
        config["model_params"]["train_stride"],
    )
    test_data, test_label = data_center.load_data(
        config["data_params"]["content_path"],
        config["data_params"]["dates_dict_path"],
        config["model_params"]["test_start"],
        config["model_params"]["test_end"],
        config["model_params"]["num_days"],
        config["model_params"]["pred_len"],
        config["model_params"]["test_stride"],
    )

    trainer = GNNTrainer(
        train_data,
        train_label,
        test_data,
        test_label,
        adj_matrix,
        config["exp_params"]["num_gnn_layers"],
        config["exp_params"]["epochs"],
        config["exp_params"]["learning_rate"],
        config["exp_params"]["batch_size"],
        torch.device(config["exp_params"]["device"]),
        config["logging_params"]["work_dir"],
        config["data_params"]["dish_dict_path"],
        config["data_params"]["dates_dict_path"],
        processed_data_path= config["data_params"]["processed_data_path"]
    )

    if args.mode == "train":
        trainer.train()

    elif args.mode == "test":
        trainer.test(
            config["model_params"]["test_start"],
            config["exp_params"]["test_model_path"],
            config["model_params"]["num_days"],
        )
    else:
        raise ValueError("Invalid mode.")
