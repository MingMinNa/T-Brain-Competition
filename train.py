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


def build_and_train_regression(X_train, y_train, output_dir):
    """
    Build and train a regression model, then save the model to the specified directory.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    model_filename = "Regression_Model.joblib"
    model_path = os.path.join(output_dir, model_filename)
    joblib.dump(model, model_path)
    print(f"Regression model has been saved to {model_path}")

    print("Intercept: ", model.intercept_)
    print("Coefficients: ", model.coef_)
    print("R squared: ", model.score(X_train, y_train))


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

    # Check for missing columns
    required_columns = regression_features + regression_target
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"The following required columns are missing from the data: {missing_cols}")

    # Prepare regression data
    Regression_X = df[regression_features].values
    Regression_y = df[regression_target].values

    # LSTM features (same as regression features excluding the target)
    AllOutPut = df[regression_features].values

    # Normalize LSTM data
    print("Normalizing LSTM data...")
    lstm_scaler = MinMaxScaler()
    AllOutPut_MinMax = lstm_scaler.fit_transform(AllOutPut)

    # Save LSTM scaler
    lstm_scaler_filename = f"LSTM_MinMaxScaler_{datetime.now().strftime('%Y-%m-%dT%H_%M_%SZ')}.joblib"
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
    X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], X_train_lstm.shape[2]))
    print(f"LSTM training data shape: {X_train_lstm.shape}")

    # Normalize regression data
    print("Normalizing regression data...")
    regression_scaler = MinMaxScaler()
    Regression_X_scaled = regression_scaler.fit_transform(Regression_X)

    # Save regression scaler
    regression_scaler_filename = f"Regression_MinMaxScaler_{datetime.now().strftime('%Y-%m-%dT%H_%M_%SZ')}.joblib"
    regression_scaler_path = os.path.join(output_dir, regression_scaler_filename)
    joblib.dump(regression_scaler, regression_scaler_path)
    print(f"Regression scaler has been saved to {regression_scaler_path}")

    # Train LSTM model
    print("Training LSTM model...")
    build_and_train_lstm(X_train_lstm, y_train_lstm, output_dir, epochs=100, batch_size=128)

    # Train regression model
    print("Training regression model...")
    build_and_train_regression(Regression_X_scaled, Regression_y, output_dir)


if __name__ == "__main__":
    main()
