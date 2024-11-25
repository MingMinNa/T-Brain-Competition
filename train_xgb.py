# %%
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from datetime import timedelta

import pandas as pd
import numpy as np
import joblib
import os


project_root = os.getcwd()
submission_path = os.path.join(project_root, "Submission", "submission.csv")
train_data_path = os.path.join(project_root, "TrainData", "avg_train_data_closest_lag.csv")
closest_train_data_path = os.path.join(project_root, "TrainData", "avg_train_data_closest.csv")
incomplete_train_data_path = os.path.join(project_root, "TrainData", "incomplete_avg_train_data.csv")
output_module_path = os.path.join(project_root, "Module", "xgb_closest_lag_model.pkl")
output_csv_path = os.path.join(project_root, "Output", "xgb_closest_lag.csv")

train_df = pd.read_csv(train_data_path)
train_df["Date"] = pd.to_datetime(train_df["Date"])
train_df = train_df.sort_values(by=["DeviceID", "Date", "Month", "Hour", "Minute"]).reset_index(drop=True)

feature_columns = [
    # "WindSpeed(m/s)",
    "Pressure(hpa)",
    "Temperature(°C)",
    "Humidity(%)",
    "Sunlight(Lux)",
    "ElevationAngle(degree)",
    "Azimuth(degree)",
]

lags_columns = [
    "lag_1_Power(mW)",
    "lag_3_Power(mW)",
    "lag_6_Power(mW)",
    "lag_12_Power(mW)",
]

device_id_columns = [col for col in train_df.columns if "DeviceID_" in col]
model_features = (
    [f"Avg_5D_{feature}" for feature in feature_columns]
    + [f"{lag}" for lag in lags_columns]
    + ["Month"]
    + device_id_columns
)

print(model_features)

# %%
# ============================ 訓練 XGBoost 模型 ============================
target = "Power(mW)"
X = train_df[model_features]
y = train_df[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [4, 6, 8, 10],
    "min_child_weight": [1, 3, 5, 7],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "n_estimators": [100, 200, 300, 500],
    "gamma": [0, 0.1, 0.2, 0.3],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [1, 1.5, 2, 2.5],
}

xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    tree_method="auto",
)

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    scoring="neg_mean_squared_error",
    cv=5,
    verbose=2,
    n_iter=100,
    n_jobs=-1,
    random_state=42,
)

random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")

best_model = random_search.best_estimator_
val_predictions = best_model.predict(X_val)
mse = mean_squared_error(y_val, val_predictions)
r2 = r2_score(y_val, val_predictions)
print(f"Validation MSE: {mse}")
print(f"Validation R2: {r2}")

joblib.dump(best_model, output_module_path)
print(f"Model saved to {output_module_path}")

# %%
# ============================ 預測發電量 ============================
# 載入模型
xgb_model = joblib.load(output_module_path)

submission_df = pd.read_csv(submission_path)
submission_df["Date"] = pd.to_datetime(submission_df["Date"])
submission_df["Datetime"] = pd.to_datetime(submission_df["Datetime"])

closest_train_df = pd.read_csv(closest_train_data_path)
closest_train_df["Date"] = pd.to_datetime(train_df["Date"])

incomplete_df = pd.read_csv(incomplete_train_data_path)
incomplete_df["Date"] = pd.to_datetime(incomplete_df["Date"])
incomplete_df["Datetime"] = pd.to_datetime(incomplete_df["Datetime"])

# 初始化預測結果列表
predictions = []
predicted_power = {}

# 遍歷每一行 submission_df 進行預測
for idx, row in submission_df.iterrows():
    serial = row["Serial"]
    prediction_datetime = row["Datetime"]
    prediction_date = row["Date"]
    device_id = row["DeviceID"]
    hour = row["Hour"]
    minute = row["Minute"]
    month = row["Month"]

    current_time = prediction_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

    relevant_data = closest_train_df[
        (closest_train_df["DeviceID"] == device_id)
        & (closest_train_df["Hour"] == row["Hour"])
        & (closest_train_df["Minute"] == row["Minute"])
    ].copy()

    relevant_data["DaysDiff"] = (prediction_date - relevant_data["Date"]).dt.days.abs()

    nearest_5_days = relevant_data.nsmallest(5, "DaysDiff")

    if nearest_5_days.empty:
        print(f"Error: 對於 Serial {serial}，找不到最近 5 天的訓練數據。使用當月同時間同裝置的平均值進行預測。")
        avg_features = closest_train_df[
            (closest_train_df["DeviceID"] == device_id)
            & (closest_train_df["Month"] == month)
            & (closest_train_df["Hour"] == hour)
            & (closest_train_df["Minute"] == minute)
        ][feature_columns].mean()
    else:
        avg_features = nearest_5_days[feature_columns].mean()

    # 計算滯後時間點
    lag_10_min = prediction_datetime - timedelta(minutes=10)
    lag_30_min = prediction_datetime - timedelta(minutes=30)
    lag_60_min = prediction_datetime - timedelta(hours=1)
    lag_120_min = prediction_datetime - timedelta(hours=2)

    # 初始化滯後特徵
    lag_10_power = 0
    lag_30_power = 0
    lag_60_power = 0
    lag_120_power = 0

    lag_10_row = incomplete_df[(incomplete_df["DeviceID"] == device_id) & (incomplete_df["Datetime"] == lag_10_min)]
    lag_30_row = incomplete_df[(incomplete_df["DeviceID"] == device_id) & (incomplete_df["Datetime"] == lag_30_min)]
    lag_60_row = incomplete_df[(incomplete_df["DeviceID"] == device_id) & (incomplete_df["Datetime"] == lag_60_min)]
    lag_120_row = incomplete_df[(incomplete_df["DeviceID"] == device_id) & (incomplete_df["Datetime"] == lag_120_min)]

    if lag_10_row.empty:
        lag_10_power = predicted_power.get(device_id, {}).get(lag_10_min, 0)
    else:
        lag_10_power = lag_10_row["Power(mW)"].values[0]

    if lag_30_row.empty:
        lag_30_power = predicted_power.get(device_id, {}).get(lag_30_min, 0)
    else:
        lag_30_power = lag_30_row["Power(mW)"].values[0]

    if lag_60_row.empty:
        lag_60_power = predicted_power.get(device_id, {}).get(lag_60_min, 0)
    else:
        lag_60_power = lag_60_row["Power(mW)"].values[0]

    if lag_120_row.empty:
        lag_120_power = predicted_power.get(device_id, {}).get(lag_120_min, 0)
    else:
        lag_120_power = lag_120_row["Power(mW)"].values[0]

    print(lag_10_power, lag_30_power, lag_60_power, lag_120_power)

    feature_dict = {
        # "Avg_5D_WindSpeed(m/s)": avg_features["WindSpeed(m/s)"],
        "Avg_5D_Pressure(hpa)": avg_features["Pressure(hpa)"],
        "Avg_5D_Temperature(°C)": avg_features["Temperature(°C)"],
        "Avg_5D_Humidity(%)": avg_features["Humidity(%)"],
        "Avg_5D_Sunlight(Lux)": avg_features["Sunlight(Lux)"],
        "Avg_5D_ElevationAngle(degree)": avg_features["ElevationAngle(degree)"],
        "Avg_5D_Azimuth(degree)": avg_features["Azimuth(degree)"],
        "lag_1_Power(mW)": lag_10_power,
        "lag_3_Power(mW)": lag_30_power,
        "lag_6_Power(mW)": lag_60_power,
        "lag_12_Power(mW)": lag_120_power,
        "Month": row["Month"],
    }

    # 添加 DeviceID 的 One-Hot 編碼
    for col in device_id_columns:
        feature_dict[col] = 1 if col == f"DeviceID_{str(device_id).zfill(2)}" else 0

    # 創建 DataFrame
    feature_df = pd.DataFrame([feature_dict])

    # 預測
    power_pred = xgb_model.predict(feature_df)[0]
    rounded_power_pred = round(power_pred, 2)
    formatted_pred = f"{rounded_power_pred:.2f}"
    print(f"{idx + 1}: {formatted_pred}")

    predictions.append(
        {
            "序號": serial,
            "答案": max(0, float(formatted_pred)),
        }
    )

    if device_id not in predicted_power:
        predicted_power[device_id] = {}
    predicted_power[device_id][current_time] = max(0, float(formatted_pred))

# 保存預測結果
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv(output_csv_path, index=False, encoding="utf-8")
print(f"Data saved to {output_csv_path}")

# %%
