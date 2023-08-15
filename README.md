#### Run Tests

```bash
$ python -m unittest -v
```

#### Run a single test

```bash
$ python -m unittest -v test.test_dataloader
```

#### Setup config file

```bash
$ python -m utils.config --config_name ustgcn
```

#### Train

```bash
$ python train.py --seed 42 --config ustgcn --mode test
```

#### Run Tensorboard:

```bash
$ tensorboard --logdir=logs/ --port=2171
```
