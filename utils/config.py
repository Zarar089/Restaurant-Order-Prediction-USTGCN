# -*- coding: utf-8 -*-

"""
Configuration file for the USTGCN model.

This file contains hyperparameters and other configurations for the USTGCN.
"""

__author__ = "Mir Sazzat Hossain"


import argparse
import os

import torch
import yaml

ustgcn_config = {
    "exp_params": {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_gnn_layers": 3,
        "epochs": 500,
        "learning_rate": 0.001,
        "batch_size": 256,
        "test_model_path": os.path.join(
            os.getcwd(),
            'logs',
            'run_24',
        ),

    },
    "model_params": {
        "num_days": 30,
        "pred_len": 7,
        "train_start": 1,
        "train_end": 552,
        "test_start": 552,
        "test_end": 690,
        "train_stride": 1,
        "test_stride": 7,
    },
    "logging_params": {
        "work_dir": os.getcwd(),
        "last_saved_model": os.path.join(
            os.getcwd(),
            'logs',
            'run_' +
            str(sorted(
                # get all the numbers from the log directory names
                [
                    int(x.split('_')[1])
                    for x in os.listdir(os.path.join(
                        os.getcwd(),
                        'logs',
                    ))
                    if x.startswith('run_')
                ], reverse=True,
            )[0]) if len(os.listdir(os.path.join(
                os.getcwd(),
                'logs',
            ))) > 0 else 'run_0',
        ),
    },
    "data_params": {
        "adj_path": "data/processed/co_occurrence_matrix.csv",
        "content_path": "data/processed/order_matrix_new.csv",
        "dish_dict_path": "data/processed/dish_dict_new.pkl",
        "dates_dict_path": "data/processed/dates_dict.pkl",
    },
}


def write_config(config: dict, config_name: str) -> None:
    """
    Write the configuration dictionary to a yaml file.

    :param config: The configuration dictionary.
    :type config: dict
    :param config_name: The name of the configuration file.
    :type config_name: str
    """
    with open(
        f'configs/{config_name}_config.yaml',
        'w', encoding='utf8'
    ) as config_file:
        yaml.dump(config, config_file, default_flow_style=False)


def load_config(config_name: str) -> dict:
    """
    Load the configuration dictionary from a yaml file.

    :param config_name: The name of the configuration file.
    :type config_name: str
    :return: The configuration dictionary.
    :rtype: dict
    """
    with open(
        f'configs/{config_name}_config.yaml',
        'r', encoding='utf8'
    ) as config_file:
        config = yaml.safe_load(config_file)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_name',
        type=str,
        default='ustgcn',
        help='The name of the configuration file.'
    )
    args = parser.parse_args()

    if args.config_name == 'ustgcn':
        write_config(ustgcn_config, args.config_name)
    else:
        raise ValueError(
            'The configuration file name is not valid. '
            'Please choose from the following: '
            '- ustgcn'
        )
