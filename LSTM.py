# %%
from tensorflow.keras.models import load_model, Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore

from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import os

# 設定LSTM往前看的筆數和預測筆數
LookBackNum = 12  # LSTM往前看的筆數
ForecastNum = 48  # 預測筆數

# Define project paths
project_root = os.getcwd()
train_data_path = os.path.join(project_root, "TrainData", "train_data.csv")
incomplete_avg_train_data_path = os.path.join(project_root, "TrainData", "incomplete_avg_train_data.csv")
submission_path = os.path.join(project_root, "Submission", "upload.csv")
output_dir = os.path.join(project_root, "Output")
os.makedirs(output_dir, exist_ok=True)

# 載入訓練資料
SourceData = pd.read_csv(train_data_path, encoding="utf-8")

# 選擇要留下來的資料欄位 (僅使用氣象參數作為特徵)
Features = SourceData[
    [
        "Pressure(hpa)",
        "Temperature(°C)",
        "Humidity(%)",
        "Sunlight(Lux)",
        "Month",
        "Hour",
        "DeviceID",
    ]
].values

# 目標變數 (Power)
Target = SourceData[["Power(mW)"]].values

# 正規化特徵和目標
Feature_Scaler = MinMaxScaler().fit(Features)
Features_MinMax = Feature_Scaler.transform(Features)

# 準備LSTM的輸入和輸出
X_train = []
y_train = []

# 設定每i-12筆資料(X_train)就對應到第i筆資料(y_train)
for i in range(LookBackNum, len(Features_MinMax)):
    X_train.append(Features_MinMax[i - LookBackNum : i, :])
    y_train.append(Target[i, 0])  # 使用原始Power值作為目標

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping
# (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))


# %%
# ============================建置&訓練「LSTM模型」============================
# 建置LSTM模型
model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=64))
model.add(Dropout(0.2))

# output layer (單一輸出節點對應Power)
model.add(Dense(units=1, activation="linear"))  # 使用線性激活適合回歸
model.compile(optimizer="adam", loss="mean_squared_error")

# 開始訓練
model.fit(X_train, y_train, epochs=100, batch_size=128)

# 保存模型
model.save("WeatherLSTM.h5")
print("LSTM Model Saved")


# %%
# ============================預測數據============================
# 載入模型
model = load_model("WeatherLSTM.h5")

# 載入測試資料
SourceData = pd.read_csv(submission_path, encoding="utf-8")
target = ["序號"]
EXquestion = SourceData[target].values

inputs = []  # 存放參考資料
PredictPower = []  # 存放預測值(發電量)

count = 0
while count < len(EXquestion):
    print("Processing count:", count)
    LocationCode = int(EXquestion[count])
    strLocationCode = str(LocationCode).zfill(2)  # 確保至少兩位數

    # 載入參考資料
    ReferenceData = pd.read_csv(incomplete_avg_train_data_path, encoding="utf-8")
    ReferTitle = ReferenceData[["Serial"]].values.flatten()
    ReferData = ReferenceData[
        [
            "Pressure(hpa)",
            "Temperature(°C)",
            "Humidity(%)",
            "Sunlight(Lux)",
            "Month",
            "Hour",
            "DeviceID",
        ]
    ].values

    # 重置存放參考資料
    inputs = []

    # 找到相同的一天，把12個資料都加進inputs
    target_date_prefix = str(int(EXquestion[count]))[:8]
    for DaysCount in range(len(ReferTitle)):
        if str(int(ReferTitle[DaysCount]))[:8] == target_date_prefix:
            TempData = ReferData[DaysCount].reshape(1, -1)
            TempData = Feature_Scaler.transform(TempData)
            inputs.append(TempData.flatten())

    # 轉換為numpy array
    inputs = np.array(inputs)

    # 檢查是否有足夠的參考資料
    if len(inputs) < LookBackNum:
        print(f"Not enough reference data for count {count}. Skipping.")
        count += ForecastNum
        continue

    # 用迴圈不斷預測並更新inputs
    for i in range(ForecastNum):
        # 準備當前的輸入序列 (LookBackNum timesteps, 7 features)
        current_input = inputs[-LookBackNum:].reshape(1, LookBackNum, 7)

        # 預測Power
        predicted_power = model.predict(current_input)
        predicted_power_value = round(predicted_power[0, 0], 2)
        PredictPower.append(predicted_power_value)
        print(f"Predicted Power: {predicted_power_value:.2f} mW")

        # 應對未來特徵的缺失，假設未來的特徵與最後一個已知特徵相同
        # 這是簡單的假設，實際應用中應根據具體情況處理
        last_features = inputs[-1].copy()

        inputs = np.vstack([inputs, last_features])

    # 每次預測都要預測48個，因此加48個會切到下一天
    # 0~47,48~95,96~143...
    count += ForecastNum
# 將預測結果寫成新的CSV檔案
df = pd.DataFrame(PredictPower, columns=["Predicted_Power(mW)"])

# 將 DataFrame 寫入 CSV 檔案
output_file_path = os.path.join(output_dir, "output.csv")
df.to_csv(output_file_path, index=False)
print("Output CSV File Saved at", output_file_path)

# %%
