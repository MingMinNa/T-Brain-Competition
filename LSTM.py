# %%
from tensorflow.keras.models import load_model, Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import joblib
import os

np.set_printoptions(suppress=True, precision=2)

# 設定LSTM往前看的筆數和預測筆數
LookBackNum = 12  # LSTM往前看的筆數
ForecastNum = 48  # 預測筆數

# Define project paths
project_root = os.getcwd()
train_data_path = os.path.join(project_root, "TrainData", "avg_train_data.csv")
incomplete_avg_train_data_path = os.path.join(project_root, "TrainData", "incomplete_avg_train_data.csv")
submission_path = os.path.join(project_root, "Submission", "upload.csv")
output_dir = os.path.join(project_root, "Output")
os.makedirs(output_dir, exist_ok=True)

# 載入訓練資料
SourceData = pd.read_csv(train_data_path, encoding="utf-8")

# 迴歸分析 選擇要留下來的資料欄位
Regression_X_train = SourceData[
    [
        "Pressure(hpa)",
        "Temperature(°C)",
        "Humidity(%)",
        "Sunlight(Lux)",
        "Month",
        "DeviceID",
    ]
].values
Regression_y_train = SourceData[["Power(mW)"]].values

# LSTM 選擇要留下來的資料欄位
AllOutPut = SourceData[
    [
        "Pressure(hpa)",
        "Temperature(°C)",
        "Humidity(%)",
        "Sunlight(Lux)",
        "Month",
        "DeviceID",
    ]
].values

# 正規化
LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)
AllOutPut_MinMax = LSTM_MinMaxModel.transform(AllOutPut)

X_train = []
y_train = []

# 設定每i-12筆資料(X_train)就對應到第i筆資料(y_train)
for i in range(LookBackNum, len(AllOutPut_MinMax)):
    X_train.append(AllOutPut_MinMax[i - LookBackNum : i, :])
    y_train.append(AllOutPut_MinMax[i, :])


X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping
# (samples 是訓練樣本數量,timesteps 是每個樣本的時間步長,features 是每個時間步的特徵數量)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))


# %%
# ============================建置&訓練「LSTM模型」============================
# 建置LSTM模型
regressor = Sequential()
regressor.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
regressor.add(LSTM(units=64))
regressor.add(Dropout(0.2))

# output layer
regressor.add(Dense(units=6, activation="relu"))
regressor.compile(optimizer="adam", loss="mean_squared_error")

# 開始訓練
regressor.fit(X_train, y_train, epochs=100, batch_size=128)

# 保存模型
regressor.save("WheatherLSTM.h5")
print("Model Saved")


# %%
# ============================建置&訓練「回歸模型」========================
# 開始迴歸分析(對發電量做迴歸)
RegressionModel = Ridge(positive=True, fit_intercept=False)
RegressionModel.fit(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train)

joblib.dump(RegressionModel, "WheatherRegression")

# 取得係數
print("係數 : ", RegressionModel.coef_)

# 取得R平方
print("R squared: ", RegressionModel.score(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train))

# %%
# ============================預測數據============================
# 載入模型
regressor = load_model("WheatherLSTM.h5")
Regression = joblib.load("WheatherRegression")


# 載入測試資料
SourceData = pd.read_csv(submission_path, encoding="utf-8")
target = ["序號"]
EXquestion = SourceData[target].values

inputs = []  # 存放參考資料
PredictOutput = []  # 存放預測值(天氣參數)
PredictPower = []  # 存放預測值(發電量)

count = 0
while count < len(EXquestion):
    print("count : ", count)
    LocationCode = int(EXquestion[count].item())
    strLocationCode = str(LocationCode)[-2:]
    if LocationCode < 10:
        strLocationCode = "0" + LocationCode

    SourceData = pd.read_csv(incomplete_avg_train_data_path, encoding="utf-8")
    ReferTitle = SourceData[["Serial"]].values
    ReferData = SourceData[
        [
            "Pressure(hpa)",
            "Temperature(°C)",
            "Humidity(%)",
            "Sunlight(Lux)",
            "Month",
            "DeviceID",
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
            inputs.append(PredictOutput[i - 1].reshape(1, 6))

        # 切出新的參考資料12筆(往前看12筆)
        X_test = []
        X_test.append(inputs[0 + i : LookBackNum + i])

        # Reshaping
        NewTest = np.array(X_test)
        NewTest = np.reshape(NewTest, (NewTest.shape[0], NewTest.shape[1], 6))

        # 使用訓練好的模型進行預測
        predicted = regressor.predict(NewTest)

        # 將預測結果加入 PredictOutput
        PredictOutput.append(predicted)

        # 使用回歸模型進行二次預測
        regression_pred = Regression.predict(predicted)

        # 將負數預測值校正為0
        regression_pred = np.maximum(regression_pred, 0)

        # 將預測值四捨五入到小數點後兩位並展平
        regression_pred = np.round(regression_pred, 2).flatten()

        # 將校正後的預測值加入 PredictPower
        PredictPower.append(regression_pred)

        # 打印校正後的預測值
        print(regression_pred)

    # 每次預測都要預測48個，因此加48個會切到下一天
    # 0~47,48~95,96~143...
    count += 48

# 寫預測結果寫成新的CSV檔案
# 將陣列轉換為 DataFrame
df = pd.DataFrame(PredictPower, columns=["答案"])

# 將 DataFrame 寫入 CSV 檔案
df.to_csv("output.csv", index=False)
print("Output CSV File Saved")

# %%
