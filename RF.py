import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import os

# 1. 加載和預處理數據

project_root = os.getcwd()
avg_data_dir = os.path.join(project_root, "TrainData(AVG)")
incomplete_avg_data_dir = os.path.join(project_root, "TrainData(IncompleteAVG)")
additional_dir = os.path.join(project_root, "AdditionalTrainData")
output_csv_dir = os.path.join(project_root, "TrainData")

# 加載 traindata.csv
train_data = os.path.join(project_root, "TrainData", "total_train_data.csv")
train_df = pd.read_csv(train_data)

# 假設 'Serial' 格式為 "YYYYMMDDXXXX"（前8位為日期）
train_df["Date"] = train_df["Serial"].astype(str).str[:8]
train_df["Date"] = pd.to_datetime(train_df["Date"], format="%Y%m%d")

# 加載 update.csv
upload_data = os.path.join(project_root, "Submission", "upload.csv")
update_df = pd.read_csv(upload_data)


# 解析 Serial 列中的日期和時間
def parse_serial(serial):
    serial_str = str(serial)
    # 假設序號 3 位，年份 4 位，月份 2 位，日期 2 位，時間 4 位，DeviceID 2 位
    serial_length = len(serial_str)
    index = 0
    serial_no = serial_str[index : index + 3]
    index += 3
    year = serial_str[index : index + 4]
    index += 4
    month = serial_str[index : index + 2]
    index += 2
    day = serial_str[index : index + 2]
    index += 2
    time = serial_str[index : index + 4]
    index += 4
    device_id = serial_str[index : index + 2]
    date = f"{year}-{month}-{day} {time[:2]}:{time[2:]}"
    return pd.to_datetime(date, format="%Y-%m-%d %H:%M")


update_df["Datetime"] = update_df["Serial"].apply(parse_serial)
update_df["DeviceID"] = update_df["Serial"].astype(str).str[-2:]
update_df["DeviceID"] = update_df["DeviceID"].astype(int)

# 2. 查找最接近的 5 天數據

update_df["Date"] = update_df["Datetime"].dt.date
train_df["DateOnly"] = train_df["Date"].dt.date

unique_train_dates = train_df["DateOnly"].unique()
unique_train_dates = np.array(sorted(unique_train_dates))


def get_closest_dates(target_date, unique_dates, num=5):
    diffs = np.abs(unique_dates - target_date)
    closest_indices = np.argsort(diffs)[:num]
    return unique_dates[closest_indices]


update_df["ClosestDates"] = update_df["Date"].apply(lambda x: get_closest_dates(x, unique_train_dates, num=5))

# 3. 準備特徵和目標變量

features = [
    "WindSpeed(m/s)",
    "Pressure(hpa)",
    "Temperature(°C)",
    "Humidity(%)",
    "Sunlight(Lux)",
    "ElevationAngle",
    "Azimuth",
    "Month",
    "Hour",
    "Minute",
]

# 添加 DeviceID 編碼
train_df["DeviceID"] = train_df["DeviceID"].astype(int)
train_df = pd.get_dummies(train_df, columns=["DeviceID"])
device_id_columns = [col for col in train_df.columns if "DeviceID_" in col]
features.extend(device_id_columns)

# 處理時間特徵
train_df["Month"] = train_df["Date"].dt.month
train_df["Hour"] = train_df["Date"].dt.hour
train_df["Minute"] = train_df["Date"].dt.minute

# 目標變量
target = "Power(mW)"

# 4. 構建訓練數據集

X_train_list = []
y_train_list = []

for idx, row in update_df.iterrows():
    device_id = row["DeviceID"]
    closest_dates = row["ClosestDates"]

    # 選擇相同 DeviceID 並且日期在最近的 5 天內的數據
    date_mask = train_df["DateOnly"].isin(closest_dates)
    device_mask = train_df[f"DeviceID_{device_id}"] == 1
    subset = train_df[date_mask & device_mask]

    if subset.empty:
        continue

    X_train_list.append(subset[features])
    y_train_list.append(subset[target])

# 檢查是否有數據
if not X_train_list:
    print("沒有找到匹配的訓練數據。")
    exit()

X_train = pd.concat(X_train_list)
y_train = pd.concat(y_train_list)

# 5. 訓練隨機森林模型

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_tr, y_tr)

# 評估模型
y_pred = rf_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Validation MSE: {mse}")

# 6. 生成預測

start_time = datetime.strptime("09:00", "%H:%M")
end_time = datetime.strptime("17:00", "%H:%M")
delta = timedelta(minutes=10)

predictions = []

for idx, row in update_df.iterrows():
    device_id = row["DeviceID"]
    prediction_date = row["Date"]

    current_time = start_time
    while current_time <= end_time:
        prediction_datetime = datetime.combine(prediction_date, current_time.time())

        # 構建特徵向量
        feature_dict = {
            "WindSpeed(m/s)": train_df["WindSpeed(m/s)"].mean(),
            "Pressure(hpa)": train_df["Pressure(hpa)"].mean(),
            "Temperature(°C)": train_df["Temperature(°C)"].mean(),
            "Humidity(%)": train_df["Humidity(%)"].mean(),
            "Sunlight(Lux)": train_df["Sunlight(Lux)"].mean(),
            "ElevationAngle": train_df["ElevationAngle"].mean(),
            "Azimuth": train_df["Azimuth"].mean(),
            "Month": prediction_datetime.month,
            "Hour": prediction_datetime.hour,
            "Minute": prediction_datetime.minute,
        }

        # 添加 DeviceID 編碼
        for col in device_id_columns:
            feature_dict[col] = 1 if col == f"DeviceID_{device_id}" else 0

        # 創建特徵 DataFrame
        feature_df = pd.DataFrame([feature_dict])

        # 預測
        power_pred = rf_model.predict(feature_df)[0]

        # 保存預測結果
        predictions.append(
            {
                "Serial": row["Serial"],
                "Datetime": prediction_datetime.strftime("%Y-%m-%d %H:%M"),
                "Predicted_Power(mW)": power_pred,
            }
        )

        current_time += delta

# 7. 保存預測結果

predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv("power_predictions.csv", index=False)
print("預測完成，結果已保存到 'power_predictions.csv'")
