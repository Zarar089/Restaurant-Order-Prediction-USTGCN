def Config():
    """Config class for setting hyperparameters."""
    config = {
        'num_days': 30,
        'pred_len': 7,
        'train_end': 80,
        'test_end': 126,
        # 'week_days': sinosidal encoding
        'adj_path': 'data/processed/food_adj.csv',
        'content_path': 'data/processed/order_matrix.csv',
    }

    return type('Config', (object,), config)()
