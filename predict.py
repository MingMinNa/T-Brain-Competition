from tensorflow.keras.models import load_model  # type:ignore

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
submission_path = os.path.join(project_root, "Submission", "upload.csv")
output_dir = os.path.join(project_root, "Output")
os.makedirs(output_dir, exist_ok=True)

# Constants
LookBackNum = 12  # Number of past time steps to consider for LSTM
ForecastNum = 48  # Number of time steps to forecast


def main():
    # Load the trained LSTM model
    lstm_model_path = os.path.join(project_root, "Module", "LSTM_Model.h5")
    lstm_model = load_model(lstm_model_path)
    print(f"LSTM model loaded from {lstm_model_path}")

    # Load the trained Regression model
    regression_model_path = os.path.join(project_root, "Module", "Regression_Model.joblib")
    regression_model = joblib.load(regression_model_path)
    print(f"Regression model loaded from {regression_model_path}")

    # Load the LSTM scaler
    lstm_scaler_path = os.path.join(project_root, "Module", "LSTM_MinMaxScaler.joblib")
    lstm_scaler = joblib.load(lstm_scaler_path)
    print(f"LSTM scaler loaded from {lstm_scaler_path}")

    # Read the submission data
    submission_data = pd.read_csv(submission_path, encoding="utf-8")
    target_column = "序號"  # Assuming "序號" is the identifier column
    EXquestion = submission_data[target_column].values

    # Prepare list to store predictions
    PredictPower = []

    # Read the training data once outside the loop for efficiency
    train_data = pd.read_csv(train_data_path, encoding="utf-8")

    # Extract relevant features from the training data
    feature_columns = [
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

    refer_titles = train_data["Serial"].values
    refer_data = train_data[feature_columns].values

    # Iterate over each entry in the submission data
    for count in range(0, len(EXquestion), ForecastNum):
        print(f"Processing batch starting at index: {count}")

        # Extract the current batch of identifiers
        batch_ids = EXquestion[count : count + ForecastNum]

        # Initialize lists to store inputs and predictions for the current batch
        inputs = []
        PredictOutput = []

        # For each identifier in the current batch
        for serial in batch_ids:
            print(f"Processing Serial: {serial}")

            # Assuming 'Serial' contains date information in the first 8 characters (e.g., YYYYMMDD)
            serial_str = str(int(serial))  # Ensure serial is a string of digits
            serial_date_prefix = serial_str[:8]  # Adjust based on actual date format in 'Serial'

            # Find all rows in training data that match the current serial's date
            matching_indices = [idx for idx, s in enumerate(refer_titles) if str(int(s)).startswith(serial_date_prefix)]

            if not matching_indices:
                print(f"No matching training data found for Serial: {serial}")
                continue  # Skip to the next serial if no matching data

            # Collect and scale the input features for the LSTM model
            for idx in matching_indices:
                temp_data = refer_data[idx].reshape(1, -1)  # Reshape to 2D array for scaler
                temp_data_scaled = lstm_scaler.transform(temp_data)
                inputs.append(temp_data_scaled.flatten())  # Flatten back to 1D

            # Ensure we have enough data points
            if len(inputs) < LookBackNum:
                print(f"Not enough data points for Serial: {serial} to create LookBack window.")
                continue  # Skip if not enough data

            # Convert inputs to numpy array and prepare LSTM input
            inputs_array = np.array(inputs)
            X_input = []
            # Use the last LookBackNum entries as the initial input
            X_input.append(inputs_array[-LookBackNum:, :])

            X_input = np.array(X_input)  # Shape: (1, LookBackNum, num_features)

            # Perform forecasting
            for i in range(ForecastNum):
                # Predict the next time step's features using LSTM
                lstm_pred = lstm_model.predict(X_input)[0]  # Shape: (num_features,)

                # Use the Regression model to predict power based on the LSTM prediction
                regression_input = lstm_pred.reshape(1, -1)  # Shape: (1, num_features)
                power_pred_scaled = regression_model.predict(regression_input)  # Shape: (1, 1)

                power_pred = power_pred_scaled[0][0]
                PredictPower.append(round(power_pred, 2))

                # Append the LSTM prediction to the input sequence for the next prediction
                # 移除最早的輸入，並加入新的預測值
                X_input = np.append(X_input[:, 1:, :], [lstm_pred.reshape(1, -1)], axis=1)

        # Convert the predictions to a DataFrame
        output_df = pd.DataFrame(PredictPower, columns=["答案"])

        # Save the predictions to a CSV file
        timestamp = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
        output_csv_path = os.path.join(output_dir, f"output_{timestamp}.csv")
        output_df.to_csv(output_csv_path, index=False)
        print(f"Output CSV File Saved to {output_csv_path}")


if __name__ == "__main__":
    main()
