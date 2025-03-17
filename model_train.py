"""
model_train.py

Contains functions for training and forecasting stock prices using three model types:
1) LSTM (train_stock_lstm_model, forecast_stock_next_day_lstm)
2) GRU (renamed as StockGRU: train_stock_gru_model, forecast_stock_next_day_gru)
3) Transformer-based model (renamed as StockTransformer: train_stock_transformer_model, forecast_stock_next_day_transformer)
"""

import os
import numpy as np
import pandas as pd
from math import sqrt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Helper function for sequence creation
def create_sequences(dataset: np.ndarray, window_size: int = 60):
    X, y = [], []
    for i in range(window_size, len(dataset)):
        X.append(dataset[i-window_size:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

### LSTM Functions ###
def train_stock_lstm_model(df: pd.DataFrame, window_size: int = 60, model_path: str = 'lstm_model.h5'):
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")
    close_data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    X_train, y_train = create_sequences(train_data, window_size)
    X_test, y_test = create_sequences(test_data, window_size)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    if os.path.exists(model_path):
        print("[train_stock_lstm_model] Loading existing model...")
        model = tf.keras.models.load_model(model_path)
    else:
        model = Sequential()
        model.add(Input(shape=(window_size, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        print("[train_stock_lstm_model] Training LSTM model...")
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, callbacks=[early_stop])
        model.save(model_path)
        print(f"[train_stock_lstm_model] Model saved to {model_path}")
    print("[train_stock_lstm_model] Evaluating on test set...")
    predictions_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions_scaled)
    actual_y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    mae = mean_absolute_error(actual_y_test, predictions)
    mse = mean_squared_error(actual_y_test, predictions)
    rmse = sqrt(mse)
    r2 = r2_score(actual_y_test, predictions)
    print(f"LSTM Test MAE:  {mae:.4f}")
    print(f"LSTM Test RMSE: {rmse:.4f}")
    print(f"LSTM Test R^2:  {r2:.4f}")
    return model, scaler, X_test, y_test

def forecast_stock_next_day_lstm(model, scaler: MinMaxScaler, df: pd.DataFrame, window_size: int = 60):
    close_data = df[['Close']].values
    scaled_data = scaler.transform(close_data)
    last_window = scaled_data[-window_size:]
    last_window_reshaped = last_window.reshape(1, window_size, 1)
    next_day_scaled = model.predict(last_window_reshaped)
    next_day_price = scaler.inverse_transform(next_day_scaled)
    return float(next_day_price[0][0])

### GRU (StockGRU) Functions ###
def train_stock_gru_model(df: pd.DataFrame, window_size: int = 60, model_path: str = 'gru_model.h5'):
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")
    close_data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    X_train, y_train = create_sequences(train_data, window_size)
    X_test, y_test = create_sequences(test_data, window_size)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    if os.path.exists(model_path):
        print("[train_stock_gru_model] Loading existing model...")
        model = tf.keras.models.load_model(model_path)
    else:
        model = Sequential()
        model.add(Input(shape=(window_size, 1)))
        model.add(GRU(50, return_sequences=True))
        model.add(GRU(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        print("[train_stock_gru_model] Training GRU model...")
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
        model.save(model_path)
        print(f"[train_stock_gru_model] Model saved to {model_path}")
    print("[train_stock_gru_model] Evaluating on test set...")
    predictions_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions_scaled)
    actual_y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    mae = mean_absolute_error(actual_y_test, predictions)
    mse = mean_squared_error(actual_y_test, predictions)
    rmse = sqrt(mse)
    r2 = r2_score(actual_y_test, predictions)
    print(f"GRU Test MAE:  {mae:.4f}")
    print(f"GRU Test RMSE: {rmse:.4f}")
    print(f"GRU Test R^2:  {r2:.4f}")
    return model, scaler, X_test, y_test

def forecast_stock_next_day_gru(model, scaler: MinMaxScaler, df: pd.DataFrame, window_size: int = 60):
    close_data = df[['Close']].values
    scaled_data = scaler.transform(close_data)
    last_window = scaled_data[-window_size:]
    last_window_reshaped = last_window.reshape(1, window_size, 1)
    next_day_scaled = model.predict(last_window_reshaped)
    next_day_price = scaler.inverse_transform(next_day_scaled)
    return float(next_day_price[0][0])

### Transformer (StockTransformer) Functions ###
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    x_ff = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    x_ff = tf.keras.layers.Dense(inputs.shape[-1])(x_ff)
    x_ff = tf.keras.layers.Dropout(dropout)(x_ff)
    x_out = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + x_ff)
    return x_out

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, dropout=0, mlp_units=[128], mlp_dropout=0, num_outputs=1):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    for units in mlp_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.Dropout(mlp_dropout)(x)
    outputs = tf.keras.layers.Dense(num_outputs)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_stock_transformer_model(df: pd.DataFrame, window_size: int = 60, model_path: str = 'transformer_model.h5'):
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")
    close_data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    X_train, y_train = create_sequences(train_data, window_size)
    X_test, y_test = create_sequences(test_data, window_size)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    input_shape = (window_size, 1)
    if os.path.exists(model_path):
        print("[train_stock_transformer_model] Loading existing model...")
        model = tf.keras.models.load_model(model_path)
    else:
        model = build_transformer_model(
            input_shape=input_shape,
            head_size=64,
            num_heads=4,
            ff_dim=128,
            num_transformer_blocks=2,
            dropout=0.1,
            mlp_units=[128],
            mlp_dropout=0.1,
            num_outputs=1
        )
        early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        print("[train_stock_transformer_model] Training Transformer model...")
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, callbacks=[early_stop])
        model.save(model_path)
        print(f"[train_stock_transformer_model] Model saved to {model_path}")
    print("[train_stock_transformer_model] Evaluating on test set...")
    predictions_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions_scaled)
    actual_y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    mae = mean_absolute_error(actual_y_test, predictions)
    mse = mean_squared_error(actual_y_test, predictions)
    rmse = sqrt(mse)
    r2 = r2_score(actual_y_test, predictions)
    print(f"Transformer Test MAE:  {mae:.4f}")
    print(f"Transformer Test RMSE: {rmse:.4f}")
    print(f"Transformer Test R^2:  {r2:.4f}")
    return model, scaler, X_test, y_test

def forecast_stock_next_day_transformer(model, scaler, df: pd.DataFrame, window_size: int = 60):
    close_data = df[['Close']].values
    scaled_data = scaler.transform(close_data)
    last_window = scaled_data[-window_size:]
    last_window_reshaped = last_window.reshape(1, window_size, 1)
    next_day_scaled = model.predict(last_window_reshaped)
    next_day_price = scaler.inverse_transform(next_day_scaled)
    return float(next_day_price[0][0])
