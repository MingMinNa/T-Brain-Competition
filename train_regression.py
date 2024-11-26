# %%
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import joblib
import os

# 設定路徑
project_root = os.getcwd()
submission_path = os.path.join(project_root, "Submission", "upload.csv")
train_data_path = os.path.join(project_root, "TrainData", "total_train_data.csv")
incomplete_train_data_path = os.path.join(project_root, "TrainData", "incomplete_avg_train_data.csv")
output_model_path_rf = os.path.join(project_root, "Module", "random_forest_model.pkl")
output_model_path_gb = os.path.join(project_root, "Module", "gradient_boosting_model.pkl")
output_scaler_path = os.path.join(project_root, "Module", "scaler.pkl")
output_csv_path = os.path.join(project_root, "Output", "regression_predictions.csv")

# 讀取訓練數據
train_df = pd.read_csv(train_data_path)

# 特徵和標籤
features = [
    "Pressure(hpa)",
    "Temperature(°C)",
    "Humidity(%)",
    "Sunlight(Lux)",
    "SunlightSaturated",
]
Regression_X_train = train_df[features].copy()
Regression_y_train = train_df[["Power(mW)"]].values

# 處理光照度感測器飽和問題
max_lux = 117758.2
Regression_X_train["Sunlight(Lux)"] = Regression_X_train["Sunlight(Lux)"].clip(upper=max_lux)

# 數據標準化
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(Regression_X_train[features])

# 建立並訓練隨機森林回歸模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, Regression_y_train.ravel())
joblib.dump(rf_model, output_model_path_rf)
print("隨機森林回歸模型已儲存")

# 建立並訓練梯度提升回歸模型
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_scaled, Regression_y_train.ravel())
joblib.dump(gb_model, output_model_path_gb)
print("梯度提升回歸模型已儲存")

# 儲存標準化器
joblib.dump(scaler, output_scaler_path)
print("標準化器已儲存")

# 評估模型性能
rf_predictions = rf_model.predict(X_train_scaled)
gb_predictions = gb_model.predict(X_train_scaled)
print("隨機森林 R²:", r2_score(Regression_y_train, rf_predictions))
print("梯度提升 R²:", r2_score(Regression_y_train, gb_predictions))

# %%
# ============================預測數據============================
# 載入模型和標準化器
rf_model_loaded = joblib.load(output_model_path_rf)
gb_model_loaded = joblib.load(output_model_path_gb)
scaler_loaded = joblib.load(output_scaler_path)

# 讀取提交的數據
submission_df = pd.read_csv(submission_path, encoding="utf-8")
target = ["Serial"]
PredictSerial = submission_df[target].values.flatten()

# 讀取參考數據
train_df_incomplete = pd.read_csv(incomplete_train_data_path, encoding="utf-8")
ReferTitle = train_df_incomplete["Serial"].values
ReferData = train_df_incomplete[features].copy()

# 處理光照度感測器飽和問題
ReferData["Sunlight(Lux)"] = ReferData["Sunlight(Lux)"].clip(upper=max_lux)
ReferData["Sunlight_Saturated"] = (ReferData["Sunlight(Lux)"] >= max_lux).astype(int)

PredictOutput = []  # 存放預測值(天氣參數)
PredictPower_rf = []  # 存放預測值(發電量) - 隨機森林
PredictPower_gb = []  # 存放預測值(發電量) - 梯度提升

for serial in PredictSerial:
    # 找到對應的參考資料
    matching_indices = np.where(ReferTitle == serial)[0]
    if len(matching_indices) == 0:
        print(f"找不到序號 {serial} 的參考資料")
        # 使用平均值或其他填補方法
        latest_data = train_df[features].mean().to_frame().T
        latest_data["Sunlight(Lux)"] = latest_data["Sunlight(Lux)"].clip(upper=max_lux)
        latest_data["SunlightSaturated"] = (latest_data["Sunlight(Lux)"] >= max_lux).astype(int)
    else:
        # 假設使用最新的一筆數據進行預測
        latest_data = ReferData.iloc[matching_indices[-1]].to_frame().T

    # 標準化
    input_scaled = scaler_loaded.transform(latest_data[features])

    # 預測發電量
    predicted_power_rf = rf_model_loaded.predict(input_scaled)[0]
    predicted_power_gb = gb_model_loaded.predict(input_scaled)[0]

    # 儲存預測結果
    PredictOutput.append(latest_data[features].values.flatten())
    PredictPower_rf.append(round(predicted_power_rf, 2))
    PredictPower_gb.append(round(predicted_power_gb, 2))

# 建立結果 DataFrame
df = pd.DataFrame(
    {
        "Serial": PredictSerial,
        # "WindSpeed(m/s)": [x[0] for x in PredictOutput],  # 如果有使用風速
        "Pressure(hpa)": [x[0] for x in PredictOutput],
        "Temperature(°C)": [x[1] for x in PredictOutput],
        "Humidity(%)": [x[2] for x in PredictOutput],
        "Sunlight(Lux)": [x[3] for x in PredictOutput],
        "Predicted_Power_RF(mW)": PredictPower_rf,
        "Predicted_Power_GB(mW)": PredictPower_gb,
    }
)

# 儲存預測結果
df.to_csv(output_csv_path, index=False)
print(f"預測結果已儲存到 {output_csv_path}")

# %%
