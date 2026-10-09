import torch
import numpy as np
import torch.nn as nn
from lstm import LSTMModel

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)

from scipy.stats import pearsonr

import config

from preprocessing import (
    load_dataset,
    separate_features_target,
    scale_data,
    create_sequences,
    convert_to_tensors
)


df = load_dataset()

X, y = separate_features_target(df)

X_scaled, y_scaled, feature_scaler, target_scaler = scale_data(X, y)

X_sequences, y_sequences = create_sequences(
    X_scaled,
    y_scaled,
    config.SEQUENCE_LENGTH
)

config.INPUT_SIZE = X_sequences.shape[2]

X_tensor, y_tensor = convert_to_tensors(
    X_sequences,
    y_sequences
)


model = LSTMModel().to(config.DEVICE)

model.load_state_dict(
    torch.load(
        "lstm_model.pth",
        map_location=config.DEVICE
    )
)

model.eval()


with torch.no_grad():

    predictions = model(
        X_tensor.to(config.DEVICE)
    )

predictions = predictions.cpu().numpy()

actual = y_tensor.numpy()


predictions = target_scaler.inverse_transform(predictions)

actual = target_scaler.inverse_transform(actual)


mse = mean_squared_error(
    actual,
    predictions
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    actual,
    predictions
)

plcc, _ = pearsonr(
    actual.flatten(),
    predictions.flatten()
)


print("\nModel Evaluation")

print(f"MSE  : {mse:.4f}")

print(f"RMSE : {rmse:.4f}")

print(f"MAE  : {mae:.4f}")

print(f"PLCC : {plcc:.4f}")