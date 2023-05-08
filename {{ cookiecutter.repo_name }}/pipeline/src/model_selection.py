from pathlib import Path
import argparse
import pandas as pd
import mlflow
from utils.data_exchange import select_first_file
import json
import os

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters


# 0. Parse all arguments
parser = argparse.ArgumentParser("model_selection")
parser.add_argument("--train_input_path", type=str, help="Input path of the train set")
parser.add_argument("--hidden_size", type=int, help="Hidden size of tft model")
parser.add_argument("--attention_head_size", type=int, help="Attention head size of tft model")
parser.add_argument("--model_output_path", type=str, help="Output path of the model")
parser.add_argument("--params_output_path", type=str, help="Output path of the parameters")

args = parser.parse_args()


# 1. We load the train data and we pass our time_df to the TimeSeriesDataSet format which is immensely useful 
# because:
# - It spares us from writing our own Dataloader.
# - We can specify how TFT will handle the dataset’s features.
# - We can normalize our dataset with ease. In our case, normalization is mandatory because all time 
#   sequences differ in magnitude. Thus, we use the GroupNormalizer to normalize each time-series individually.


train_data = pd.read_parquet(select_first_file(args.train_input_path))

#Hyperparameters
batch_size=64
#number heads=4, hidden sizes=160, lr=0.001, gr_clip=0.1

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

# create dataloaders for  our model
# if you have a strong GPU, feel free to increase the number of workers  
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)


# 2. We can train our TFT model using the familiar Trainer interface from PyTorch Lightning.
# Notice the following things:
# - We use the EarlyStopping callback to monitor the validation loss.
# - We use Tensorboard to log our training and validation metrics.
# - Our model uses Quantile Loss — a special type of loss that helps us output the prediction intervals. For more on the Quantile Loss function, check this article.
# - We use 4 attention heads, like the original paper.

# Anpassung wichtiger Variablen und Logging and lokale Umgebung (bei lokaler Ausführung)
if False: #load_dotenv():
    print("Lokale Ausführung")
    accelerator="cpu"
    max_epochs=1
else:
    print("Ausführung auf Azure")
    accelerator = "gpu"
    max_epochs=2
    mlflow.pytorch.autolog(log_models=False) # Funktioniert in Azure ML ohne mlflow.run() (für pytorch-lightning<1.94)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=True, mode="min")
lr_logger = LearningRateMonitor() 

log_dir = str(Path.cwd() / "logs")
logger = TensorBoardLogger(log_dir)  # Kein Zugriff mit Azure SDK v2, da Run Objekte verändert sind

trainer = pl.Trainer(
    max_epochs=max_epochs, # change to larger value!!! 
    accelerator=accelerator, 
    devices=1,
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger, # Muss gesetzt werden, True/False ergeben Fehler...
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=args.hidden_size,                   # parameter will be tuned via sweep
    attention_head_size=args.attention_head_size,   # parameter will be tuned via sweep
    dropout=0.1,
    hidden_continuous_size=160,
    output_size=7,  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    loss=QuantileLoss(),
    log_interval=10, 
    reduce_on_plateau_patience=4,
)

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)


# # 3. Evaluate tft model and track with mlflow

# actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
# predictions = best_tft.predict(val_dataloader)

# # mean absolute error (p50) over all time series
# mean_absolute_error = (actuals - predictions).abs().mean().item()
# print(f"Mean Absolute Error: {mean_absolute_error}")
# mlflow.log_metric("mean_absolute_error", mean_absolute_error)


# 4. Save model and save parameters (after hyperparameter tuning)
torch.save(best_tft.state_dict(), Path(args.model_output_path, "trained_tft_weights.pth"))

params = {"hidden_size": args.hidden_size, "attention_head_size": args.attention_head_size}
with Path(args.params_output_path, "model_params.json").open(mode="w") as f:
    json.dump(params, f)



