# %%
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from datetime import timedelta

import pandas as pd
import numpy as np
import joblib
import os

import const

const.build_model_folder()
project_root = os.getcwd()
submission_path = os.path.join(const.SUBMISSION_FOLDER, "Submission", "submission.csv")
lstm_submission_path = os.path.join(const.SUBMISSION_FOLDER, "Submission", "LSTM+Regression.csv")
train_data_path = os.path.join(const.TRAINING_FOLDER, "TrainData", "total_train_data_lag.csv")
complete_train_data_path = os.path.join(const.TRAINING_FOLDER, "TrainData", "avg_train_data.csv")
incomplete_train_data_path = os.path.join(const.TRAINING_FOLDER, "TrainData", "incomplete_avg_train_data.csv")
output_module_path = os.path.join(const.MODELS_FOLDER, "xgb_lag_model.pkl")

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

lags_columns = [f"lag_{i}_Power(mW)" for i in range(1, 13)]

device_id_columns = [col for col in train_df.columns if "DeviceID_" in col]
model_features = [f"{feature}" for feature in feature_columns] + lags_columns + ["Month", "Hour"] + device_id_columns

print(model_features)

# %%
# ============================ 訓練 XGBoost 模型 ============================
target = "Power(mW)"
X = train_df[model_features]
y = train_df[target]

tscv = TimeSeriesSplit(n_splits=5)

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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
    verbosity=0,
)

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    scoring="neg_mean_squared_error",
    cv=tscv,
    verbose=2,
    n_iter=100,
    n_jobs=4,
    random_state=42,
)

random_search.fit(
    X,
    y,
    verbose=False,
)

print(f"Best parameters: {random_search.best_params_}")

best_model = random_search.best_estimator_
val_predictions = best_model.predict(X)
mse = mean_squared_error(y, val_predictions)
r2 = r2_score(y, val_predictions)
print(f"Validation MSE: {mse}")
print(f"Validation R2: {r2}")

joblib.dump(best_model, output_module_path)
print(f"Model saved to {output_module_path}")

# %%
# ============================ 預測發電量(回歸) ============================
xgb_model = joblib.load(output_module_path)

submission_df = pd.read_csv(submission_path)
submission_df["Date"] = pd.to_datetime(submission_df["Date"])
submission_df["Datetime"] = pd.to_datetime(submission_df["Datetime"])

incomplete_df = pd.read_csv(incomplete_train_data_path)
incomplete_df["Date"] = pd.to_datetime(incomplete_df["Date"])
incomplete_df["Datetime"] = pd.to_datetime(incomplete_df["Datetime"])

predictions = []
predicted_power = {}

for idx, row in submission_df.iterrows():
    serial = row["Serial"]
    prediction_datetime = row["Datetime"]
    prediction_date = row["Date"]
    device_id = row["DeviceID"]
    month = row["Month"]

    current_time = prediction_date.replace(hour=row["Hour"], minute=row["Minute"], second=0, microsecond=0)

    lags = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    lag_powers = {}

    for lag in lags:
        lag_time = prediction_datetime - timedelta(minutes=lag)
        lag_row = incomplete_df[(incomplete_df["DeviceID"] == device_id) & (incomplete_df["Datetime"] == lag_time)]

        if lag_row.empty:
            lag_powers[lag] = predicted_power.get(device_id, {}).get(lag_time, 0)
        else:
            lag_powers[lag] = lag_row["Power(mW)"].values[0]

    feature_dict = {
        "Pressure(hpa)": row["Pressure(hpa)"],
        "Temperature(°C)": row["Temperature(°C)"],
        "Humidity(%)": row["Humidity(%)"],
        "Sunlight(Lux)": row["Sunlight(Lux)"],
        "ElevationAngle(degree)": row["ElevationAngle(degree)"],
        "Azimuth(degree)": row["Azimuth(degree)"],
        **{f"lag_{i+1}_Power(mW)": lag_powers[lag] for i, lag in enumerate(lags)},
        "Month": row["Month"],
        "Hour": row["Hour"],
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
output_csv_path = os.path.join(const.SUBMISSION_FOLDER, "Output", "xgb_lag_regression_v2.csv")
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv(output_csv_path, index=False, encoding="utf-8")
print(f"Data saved to {output_csv_path}")


# %%
# ============================ 預測發電量(平均) ============================
xgb_model = joblib.load(output_module_path)

submission_df = pd.read_csv(submission_path)
submission_df["Date"] = pd.to_datetime(submission_df["Date"])
submission_df["Datetime"] = pd.to_datetime(submission_df["Datetime"])

complete_df = pd.read_csv(complete_train_data_path)
complete_df["Date"] = pd.to_datetime(train_df["Date"])

incomplete_df = pd.read_csv(incomplete_train_data_path)
incomplete_df["Date"] = pd.to_datetime(incomplete_df["Date"])
incomplete_df["Datetime"] = pd.to_datetime(incomplete_df["Datetime"])

predictions = []
predicted_power = {}

for idx, row in submission_df.iterrows():
    serial = row["Serial"]
    prediction_datetime = row["Datetime"]
    prediction_date = row["Date"]
    device_id = row["DeviceID"]

    current_time = prediction_date.replace(hour=row["Hour"], minute=row["Minute"], second=0, microsecond=0)

    relevant_data = complete_df[
        (complete_df["DeviceID"] == device_id)
        & (complete_df["Hour"] == row["Hour"])
        & (complete_df["Minute"] == row["Minute"])
    ].copy()

    relevant_data["DaysDiff"] = (prediction_date - relevant_data["Date"]).dt.days.abs()
    nearest_2_days = relevant_data.nsmallest(5, "DaysDiff")
    avg_features = nearest_2_days[feature_columns].mean()
    print(avg_features)

    lags = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    lag_powers = {}

    for lag in lags:
        lag_time = prediction_datetime - timedelta(minutes=lag)
        lag_row = incomplete_df[(incomplete_df["DeviceID"] == device_id) & (incomplete_df["Datetime"] == lag_time)]

        if lag_row.empty:
            lag_powers[lag] = predicted_power.get(device_id, {}).get(lag_time, 0)
        else:
            lag_powers[lag] = lag_row["Power(mW)"].values[0]

    feature_dict = {
        "Pressure(hpa)": avg_features["Pressure(hpa)"],
        "Temperature(°C)": avg_features["Temperature(°C)"],
        "Humidity(%)": avg_features["Humidity(%)"],
        "Sunlight(Lux)": avg_features["Sunlight(Lux)"],
        "ElevationAngle(degree)": avg_features["ElevationAngle(degree)"],
        "Azimuth(degree)": avg_features["Azimuth(degree)"],
        **{f"lag_{i+1}_Power(mW)": lag_powers[lag] for i, lag in enumerate(lags)},
        "Month": row["Month"],
        "Hour": row["Hour"],
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
output_csv_path = os.path.join(const.SUBMISSION_FOLDER, "Output", "xgb_lag_closest2_v4.csv")
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv(output_csv_path, index=False, encoding="utf-8")
print(f"Data saved to {output_csv_path}")

# %%
# ============================ 預測發電量(LSTM) ============================
xgb_model = joblib.load(output_module_path)

lstm_df = pd.read_csv(lstm_submission_path)

submission_df = pd.read_csv(submission_path)
submission_df["Date"] = pd.to_datetime(submission_df["Date"])
submission_df["Datetime"] = pd.to_datetime(submission_df["Datetime"])

incomplete_df = pd.read_csv(incomplete_train_data_path)
incomplete_df["Date"] = pd.to_datetime(incomplete_df["Date"])
incomplete_df["Datetime"] = pd.to_datetime(incomplete_df["Datetime"])

predictions = []
predicted_power = {}

for idx, row in submission_df.iterrows():
    serial = row["Serial"]
    prediction_datetime = row["Datetime"]
    prediction_date = row["Date"]
    device_id = row["DeviceID"]
    month = row["Month"]

    current_time = prediction_date.replace(hour=row["Hour"], minute=row["Minute"], second=0, microsecond=0)

    lags = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    lag_powers = {}

    for lag in lags:
        lag_time = prediction_datetime - timedelta(minutes=lag)
        lag_row = incomplete_df[(incomplete_df["DeviceID"] == device_id) & (incomplete_df["Datetime"] == lag_time)]

        if lag_row.empty:
            lag_powers[lag] = predicted_power.get(device_id, {}).get(lag_time, 0)
        else:
            lag_powers[lag] = lag_row["Power(mW)"].values[0]

    pressure = lstm_df[(lstm_df["Serial"] == serial)]["Pressure(hpa)"]
    temperature = lstm_df[(lstm_df["Serial"] == serial)]["Temperature(°C)"]
    humidity = lstm_df[(lstm_df["Serial"] == serial)]["Humidity(%)"]
    sunlight = lstm_df[(lstm_df["Serial"] == serial)]["Sunlight(Lux)"]

    feature_dict = {
        "Pressure(hpa)": pressure.iloc[0],
        "Temperature(°C)": temperature.iloc[0],
        "Humidity(%)": humidity.iloc[0],
        "Sunlight(Lux)": sunlight.iloc[0],
        "ElevationAngle(degree)": row["ElevationAngle(degree)"],
        "Azimuth(degree)": row["Azimuth(degree)"],
        **{f"lag_{i+1}_Power(mW)": lag_powers[lag] for i, lag in enumerate(lags)},
        "Month": row["Month"],
        "Hour": row["Hour"],
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
output_csv_path = os.path.join(const.SUBMISSION_FOLDER, "Output", "xgb_lag_lstm_v1.csv")
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv(output_csv_path, index=False, encoding="utf-8")
print(f"Data saved to {output_csv_path}")

# %%
