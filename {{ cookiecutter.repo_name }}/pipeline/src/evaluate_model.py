from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import mlflow
from mlflow.tracking import MlflowClient
from utils.data_exchange import select_first_file


import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

# from azureml.tensorboard import Tensorboard
# tb = Tensorboard([], local_root="lightning_logs")


# 0. Parse all arguments

parser = argparse.ArgumentParser("evaluate_model")
parser.add_argument("--train_input_path", type=str, help="Input path of the train set")
parser.add_argument("--test_input_path", type=str, help="Input path of the train set")
parser.add_argument("--model_input_path", type=str, help="Input path of the model")
parser.add_argument("--params_input_path", type=str, help="Parameters of the model")

args = parser.parse_args()


# 1. Load data and create data loader

train_data = pd.read_parquet(select_first_file(args.train_input_path))
test_data = pd.read_parquet(select_first_file(args.test_input_path))

max_prediction_length = 24
max_encoder_length = 7*24
training_cutoff = train_data["hours_from_start"].max() - max_prediction_length

training = TimeSeriesDataSet(
    train_data[lambda x: x.hours_from_start <= training_cutoff],
    time_idx="hours_from_start",
    target="power_usage",
    group_ids=["consumer_id"],
    min_encoder_length=max_encoder_length // 2, 
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["consumer_id"],
    time_varying_known_reals=["hours_from_start","day","day_of_week", "month", 'hour'],
    time_varying_unknown_reals=['power_usage'],
    target_normalizer=GroupNormalizer(
        groups=["consumer_id"], transformation="softplus"
    ),  # we normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True, 
)

validation = TimeSeriesDataSet.from_dataset(training, train_data, predict=True, stop_randomization=True)
test = TimeSeriesDataSet.from_dataset(training, test_data, predict=True, stop_randomization=True)

# create dataloaders for our model
batch_size = 64 
# if you have a strong GPU, feel free to increase the number of workers  
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=0)


# 2. Load model and make predictions to get metircs

with Path(select_first_file(args.params_input_path)).open(mode="r") as f:
    params = json.load(f)
print(params)

best_tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=params["hidden_size"],
    attention_head_size=params["attention_head_size"],
    dropout=0.1,
    hidden_continuous_size=160,
    output_size=7,  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    loss=QuantileLoss(),
    log_interval=10, 
    reduce_on_plateau_patience=4,
)

best_tft.load_state_dict(torch.load(select_first_file(args.model_input_path)))

# Calculate metric for validation set

actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)

# mean absolute error (p50) over all time series
mean_absolute_error = (actuals - predictions).abs().mean().item()
print(f"Mean Absolute Error: {mean_absolute_error}")
mlflow.log_metric("mean_absolute_error", mean_absolute_error)

# Calculate metric for test set

actuals = torch.cat([y[0] for x, y in iter(test_dataloader)])
predictions = best_tft.predict(test_dataloader)

# mean absolute error (p50) over all time series
mean_absolute_error = (actuals - predictions).abs().mean().item()
print(f"Mean Absolute Error: {mean_absolute_error}")
# mlflow.log_metric("mean_absolute_error", mean_absolute_error)


# 3. Retrain with all data (to come...)


# 4. Save and register final model

mlflow.pytorch.log_model(best_tft, "model")
mlflow_run = mlflow.active_run()
run_id = mlflow_run.info.run_id
model_path = "model"
model_uri = "runs:/{}/{}".format(run_id, model_path)
mlflow.register_model(model_uri, "tft_model")