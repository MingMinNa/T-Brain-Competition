# predict.py
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
incomplete_avg_train_data_path = os.path.join(project_root, "TrainData", "incomplete_avg_train_data.csv")
submission_path = os.path.join(project_root, "Submission", "upload.csv")
output_dir = os.path.join(project_root, "Output")
os.makedirs(output_dir, exist_ok=True)

# Constants
LookBackNum = 12  # Number of past time steps to consider for LSTM
ForecastNum = 48  # Number of time steps to forecast


def main():
    lstm_model_path = os.path.join(project_root, "Module", "LSTM_Model.h5")
    lstm_model = load_model(lstm_model_path)

    regression_model_path = os.path.join(project_root, "Module", "Regression_Model")
    regression_model = joblib.load(regression_model_path)

    lstm_scaler_path = os.path.join(project_root, "Module", "LSTM_MinMaxScaler")
    lstm_scaler = joblib.load(lstm_scaler_path)

    submission_data = pd.read_csv(submission_path, encoding="utf-8")
    EXquestion = submission_data["序號"].values

    train_data = pd.read_csv(incomplete_avg_train_data_path, encoding="utf-8")

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

    inputs = []  # 存放參考資料
    PredictOutput = []  # 存放預測值(天氣參數)
    PredictPower = []  # 存放預測值(發電量)

    count = 0
    while count < len(EXquestion):
        print("count: ", count)
        inputs = []

        for DaysCount in range(len(refer_titles)):
            if str(int(refer_titles[DaysCount]))[:8] == str(int(EXquestion[count]))[:8]:
                TempData = refer_data[DaysCount].reshape(1, -1)
                TempData = lstm_scaler.transform(TempData)
                inputs.append(TempData)

        for i in range(ForecastNum):
            if i > 0:
                inputs.append(PredictOutput[i - 1].reshape(1, 23))

            X_test = []
            X_test.append(inputs[0 + i : LookBackNum + i])

            NewTest = np.array(X_test)
            NewTest = np.reshape(NewTest, (NewTest.shape[0], NewTest.shape[1], 23))

            predicted = lstm_model.predict(NewTest)
            PredictOutput.append(predicted)
            PredictPower.append(np.round(regression_model.predict(predicted), 2).flatten())
            print(np.round(regression_model.predict(predicted), 2).flatten())

        count += 48

    serial_numbers = EXquestion[: len(PredictPower)]

    PredictPower_flat = [item for sublist in PredictPower for item in sublist]

    df = pd.DataFrame({"序號": serial_numbers, "答案": PredictPower_flat})

    output_file_path = os.path.join(output_dir, "output.csv")
    df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    print(f"Output CSV File Saved to {output_file_path}")


if __name__ == "__main__":
    main()
