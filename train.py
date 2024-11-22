# train.py
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import os


# Define project paths
project_root = os.getcwd()
train_data_path = os.path.join(project_root, "TrainData", "train_data.csv")
output_dir = os.path.join(project_root, "Module")
os.makedirs(output_dir, exist_ok=True)


def build_and_train_lstm(X_train, y_train, output_dir, epochs=100, batch_size=128):
    """
    Build and train an LSTM model, then save the model to the specified directory.
    """
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=64))
    model.add(Dropout(0.2))
    model.add(Dense(units=y_train.shape[1]))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    model_filename = "LSTM_Model.h5"
    model_path = os.path.join(output_dir, model_filename)
    model.save(model_path)
    print(f"LSTM model has been saved to {model_path}")


def build_and_train_regression(lstm_scaler, Regression_X, Regression_y, output_dir):
    """
    Build and train a regression model, then save the model to the specified directory.
    """
    model = LinearRegression()
    model.fit(lstm_scaler.transform(Regression_X), Regression_y)

    model_filename = "Regression_Model"
    model_path = os.path.join(output_dir, model_filename)
    joblib.dump(model, model_path)
    print(f"Regression model has been saved to {model_path}")

    print("Intercept: ", model.intercept_)
    print("Coefficients: ", model.coef_)
    print("R squared: ", model.score(lstm_scaler.transform(Regression_X), Regression_y))


def main():
    # Load training data
    df = pd.read_csv(train_data_path, encoding="utf-8")

    # Define regression and LSTM features and target
    regression_features = [
        "Pressure(hpa)",
        "Temperature(°C)",
        "Humidity(%)",
        "Sunlight(Lux)",
        "Month",
        "Hour",
        "DeviceID_01",
        "DeviceID_02",
        "DeviceID_03",
        "DeviceID_04",
        "DeviceID_05",
        "DeviceID_06",
        "DeviceID_07",
        "DeviceID_08",
        "DeviceID_09",
        "DeviceID_10",
        "DeviceID_11",
        "DeviceID_12",
        "DeviceID_13",
        "DeviceID_14",
        "DeviceID_15",
        "DeviceID_16",
        "DeviceID_17",
    ]
    regression_target = ["Power(mW)"]

    # Prepare regression data
    Regression_X = df[regression_features].values
    Regression_y = df[regression_target].values

    # LSTM features (same as regression features excluding the target)
    AllOutPut = df[regression_features].values

    # Normalize LSTM data
    print("Normalizing LSTM and Regression features data...")
    lstm_scaler = MinMaxScaler().fit(AllOutPut)
    AllOutPut_MinMax = lstm_scaler.transform(AllOutPut)

    # Save LSTM scaler with fixed filename
    lstm_scaler_filename = "LSTM_MinMaxScaler"
    lstm_scaler_path = os.path.join(output_dir, lstm_scaler_filename)
    joblib.dump(lstm_scaler, lstm_scaler_path)
    print(f"LSTM scaler has been saved to {lstm_scaler_path}")

    # Set LookBackNum
    LookBackNum = 12

    # Build LSTM training data
    print("Building LSTM training data...")
    X_train_lstm = []
    y_train_lstm = []

    for i in range(LookBackNum, len(AllOutPut_MinMax)):
        X_train_lstm.append(AllOutPut_MinMax[i - LookBackNum : i, :])
        y_train_lstm.append(AllOutPut_MinMax[i, :])

    X_train_lstm = np.array(X_train_lstm)
    y_train_lstm = np.array(y_train_lstm)

    # Reshape to (samples, timesteps, features)
    num_features = X_train_lstm.shape[2]  # Dynamic feature count
    X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], X_train_lstm.shape[1], num_features))
    print(f"LSTM training data shape: {X_train_lstm.shape}")

    # Train LSTM model
    print("Training LSTM model...")
    build_and_train_lstm(X_train_lstm, y_train_lstm, output_dir, epochs=100, batch_size=128)

    # Train regression model
    print("Training regression model...")
    build_and_train_regression(lstm_scaler, Regression_X, Regression_y, output_dir)


if __name__ == "__main__":
    main()
