# %%
from tensorflow.keras.models import Sequential, load_model  # type:ignore
from tensorflow.keras.layers import Dense, Dropout, LSTM  # type:ignore
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import joblib
import os

LookBackNum = 12
ForecastNum = 48

project_root = os.getcwd()
submission_path = os.path.join(project_root, "Submission", "upload.csv")
train_data_path = os.path.join(project_root, "TrainData", "total_train_data.csv")
incomplete_train_data_path = os.path.join(project_root, "TrainData", "incomplete_avg_train_data.csv")
output_lstm_module_path = os.path.join(project_root, "Module", "lstm_model.h5")
output_regression_module_path = os.path.join(project_root, "Module", "regression_model")
output_csv_path = os.path.join(project_root, "Output", "lstm.csv")

train_df = pd.read_csv(train_data_path)

Regression_X_train = train_df[
    [
        # "WindSpeed(m/s)",
        "Pressure(hpa)",
        "Temperature(°C)",
        "Humidity(%)",
        "Sunlight(Lux)",
    ]
].values

Regression_y_train = train_df[["Power(mW)"]].values

AllOutPut = train_df[
    [
        # "WindSpeed(m/s)",
        "Pressure(hpa)",
        "Temperature(°C)",
        "Humidity(%)",
        "Sunlight(Lux)",
    ]
].values

LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)
AllOutPut_MinMax = LSTM_MinMaxModel.transform(AllOutPut)

X_train = []
y_train = []

for i in range(LookBackNum, len(AllOutPut_MinMax)):
    X_train.append(AllOutPut_MinMax[i - LookBackNum : i, :])
    y_train.append(AllOutPut_MinMax[i, :])


X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))

# %%
# ============================建置&訓練「LSTM模型」============================
regressor = Sequential()
regressor.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 4)))
regressor.add(LSTM(units=64))
regressor.add(Dropout(0.2))

# output layer
regressor.add(Dense(units=4, activation="relu"))
regressor.compile(optimizer="adam", loss="mean_squared_error")

# 開始訓練
regressor.fit(X_train, y_train, epochs=100, batch_size=128)
regressor.save(output_lstm_module_path)
print("LSTM Model Saved")

# %%
# ============================建置&訓練「回歸模型」========================
RegressionModel = LinearRegression()
RegressionModel.fit(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train)

joblib.dump(RegressionModel, output_regression_module_path)

print("截距: ", RegressionModel.intercept_)
print("係數: ", RegressionModel.coef_)
print("R squared: ", RegressionModel.score(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train))

# %%
# ============================預測數據============================
regressor = load_model(output_lstm_module_path)
Regression = joblib.load(output_regression_module_path)

submission_df = pd.read_csv(submission_path, encoding="utf-8")
target = ["Serial"]
EXquestion = submission_df[target].values

inputs = []  # 存放參考資料
PredictOutput = []  # 存放預測值(天氣參數)
PredictPower = []  # 存放預測值(發電量)

count = 0
while count < len(EXquestion):
    print("count: ", count)
    LocationCode = int(EXquestion[count].item())
    strLocationCode = str(LocationCode)[-2:]
    if LocationCode < 10:
        strLocationCode = "0" + LocationCode

    train_df = pd.read_csv(incomplete_train_data_path, encoding="utf-8")
    ReferTitle = train_df[["Serial"]].values
    ReferData = train_df[
        [
            # "WindSpeed(m/s)",
            "Pressure(hpa)",
            "Temperature(°C)",
            "Humidity(%)",
            "Sunlight(Lux)",
        ]
    ].values

    inputs = []  # 重置存放參考資料

    # 找到相同的一天，把12個資料都加進inputs
    for DaysCount in range(len(ReferTitle)):
        if str(int(ReferTitle[DaysCount].item()))[:8] == str(int(EXquestion[count].item()))[:8]:
            TempData = ReferData[DaysCount].reshape(1, -1)
            TempData = LSTM_MinMaxModel.transform(TempData)
            inputs.append(TempData)

    # 用迴圈不斷使新的預測值塞入參考資料，並預測下一筆資料
    for i in range(ForecastNum):
        # 將新的預測值加入參考資料(用自己的預測值往前看)
        if i > 0:
            inputs.append(PredictOutput[i - 1].reshape(1, 4))

        # 切出新的參考資料12筆(往前看12筆)
        X_test = []
        X_test.append(inputs[0 + i : LookBackNum + i])

        # Reshaping
        NewTest = np.array(X_test)
        NewTest = np.reshape(NewTest, (NewTest.shape[0], NewTest.shape[1], 4))

        predicted = regressor.predict(NewTest)
        PredictOutput.append(predicted.flatten())
        PredictPower.append(np.round(Regression.predict(predicted), 2).flatten())

    count += 48

PredictOutput = np.array(PredictOutput)
PredictPower = np.array(PredictPower)
PredictSerial = np.array(submission_df[target].values).flatten()

PredictOutput_inverse = LSTM_MinMaxModel.inverse_transform(PredictOutput)
PredictOutput_2d = PredictOutput_inverse.reshape(-1, PredictOutput.shape[-1])

df = pd.DataFrame(
    {
        "Serial": PredictSerial,
        # "WindSpeed(m/s)": PredictOutput_2d[:, 0],
        "Pressure(hpa)": PredictOutput_2d[:, 1],
        "Temperature(°C)": PredictOutput_2d[:, 2],
        "Humidity(%)": PredictOutput_2d[:, 3],
        "Sunlight(Lux)": PredictOutput_2d[:, 4],
    }
)

df[["Pressure(hpa)", "Temperature(°C)", "Humidity(%)", "Sunlight(Lux)"]] = df[
    ["Pressure(hpa)", "Temperature(°C)", "Humidity(%)", "Sunlight(Lux)"]
].round(2)

df.to_csv(output_csv_path, index=False)
print(f"Data saved to {output_csv_path}")

# %%
