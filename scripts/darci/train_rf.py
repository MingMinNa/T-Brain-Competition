# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import numpy as np
import joblib
import os

import const

const.build_model_folder()
train_data_path = os.path.join(const.TRAINING_FOLDER, "TrainData", "avg_train_data_closest_5d.csv")
submission_path = os.path.join(const.SUBMISSION_FOLDER, "Submission", "submission.csv")
output_module_dir = os.path.join(const.MODELS_FOLDER)
output_ans_dir = os.path.join(const.SUBMISSION_FOLDER, "Output")
os.makedirs(output_module_dir, exist_ok=True)
os.makedirs(output_ans_dir, exist_ok=True)


train_df = pd.read_csv(train_data_path)
train_df["Date"] = pd.to_datetime(train_df["Date"])

feature_columns = [
    # "WindSpeed(m/s)",
    "Pressure(hpa)",
    "Temperature(°C)",
    "Humidity(%)",
    "Sunlight(Lux)",
    "ElevationAngle(degree)",
    "Azimuth(degree)",
]

# 特徵
model_features = [f"Avg_5D_{feature}" for feature in feature_columns] + ["Month", "Hour"]
device_id_columns = [col for col in train_df.columns if "DeviceID_" in col]
model_features += device_id_columns

# 目標
target = "Power(mW)"

X = train_df[model_features]
y = train_df[target]


# %%
# ============================ 訓練隨機森林模型 ============================
# 分割訓練和驗證數據
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化並訓練模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 評估模型
y_pred = rf_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# 保存模型
output_module = os.path.join(output_module_dir, "rf_closest_5d_model.pkl")
joblib.dump(rf_model, output_module)
print(f"Model saved to {output_module}")


# %%
# ============================ 預測發電量 ============================
# 載入模型
rf_model = joblib.load(output_module)

submission_df = pd.read_csv(submission_path)
submission_df["Date"] = pd.to_datetime(submission_df["Date"])

# 初始化預測結果列表
predictions = []

# 遍歷每一行 submission_df 進行預測
for idx, row in submission_df.iterrows():
    serial = row["Serial"]
    prediction_date = row["Date"]
    device_id = row["DeviceID"]

    relevant_data = train_df[
        (train_df["DeviceID"] == device_id) & (train_df["Hour"] == row["Hour"]) & (train_df["Minute"] == row["Minute"])
    ].copy()

    relevant_data["DaysDiff"] = (prediction_date - relevant_data["Date"]).dt.days.abs()

    nearest_5_days = relevant_data.nsmallest(5, "DaysDiff")

    if nearest_5_days.empty:
        print(f"警告：對於 Serial {serial}，找不到最近 5 天的訓練數據。使用當月同時間同裝置的平均值進行預測。")
        avg_features = train_df[
            (train_df["DeviceID"] == device_id)
            & (train_df["Month"] == row["Month"])
            & (train_df["Hour"] == row["Hour"])
            & (train_df["Minute"] == row["Minute"])
        ][feature_columns].mean()
    else:
        avg_features = nearest_5_days[feature_columns].mean()

    feature_dict = {
        # "Avg_5D_WindSpeed(m/s)": avg_features["WindSpeed(m/s)"],
        "Avg_5D_Pressure(hpa)": avg_features["Pressure(hpa)"],
        "Avg_5D_Temperature(°C)": avg_features["Temperature(°C)"],
        "Avg_5D_Humidity(%)": avg_features["Humidity(%)"],
        "Avg_5D_Sunlight(Lux)": avg_features["Sunlight(Lux)"],
        "Avg_5D_ElevationAngle(degree)": avg_features["ElevationAngle(degree)"],
        "Avg_5D_Azimuth(degree)": avg_features["Azimuth(degree)"],
        "Month": row["Month"],
        "Hour": row["Hour"],
    }

    # 添加 DeviceID 的 One-Hot 編碼
    for col in device_id_columns:
        feature_dict[col] = 1 if col == f"DeviceID_{str(device_id).zfill(2)}" else 0

    # 創建 DataFrame
    feature_df = pd.DataFrame([feature_dict])

    # 預測
    power_pred = rf_model.predict(feature_df)[0]

    # 保存預測結果
    predictions.append(
        {
            "序號": serial,
            "答案": round(power_pred, 2),
        }
    )

# 保存預測結果
output_file = os.path.join(output_ans_dir, "rf_cloest_5d.csv")
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv(output_file, index=False, encoding="utf-8")
print(f"Data saved to {output_file}")


# %%
