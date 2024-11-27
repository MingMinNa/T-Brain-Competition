import pandas as pd
import os


project_root = os.getcwd()
train_data_path = os.path.join(project_root, "TrainData", "avg_train_data_lag.csv")
output_dir = os.path.join(project_root, "TrainData")
os.makedirs(output_dir, exist_ok=True)


def main():
    train_df = pd.read_csv(train_data_path)
    train_df["Date"] = pd.to_datetime(train_df["Date"])
    train_df = train_df.sort_values(by=["DeviceID", "Hour", "Minute", "Date"]).reset_index(drop=True)

    feature_columns = [
        "WindSpeed(m/s)",
        "Pressure(hpa)",
        "Temperature(°C)",
        "Humidity(%)",
        "Sunlight(Lux)",
        "ElevationAngle(degree)",
        "Azimuth(degree)",
    ]

    for feature in feature_columns:
        avg_values = []
        for idx, row in train_df.iterrows():
            device_id = row["DeviceID"]
            hour = row["Hour"]
            minute = row["Minute"]
            current_date = row["Date"]

            relevant_data = train_df[
                (train_df["DeviceID"] == device_id)
                & (train_df["Hour"] == hour)
                & (train_df["Minute"] == minute)
                & (train_df["Date"] != current_date)
            ].copy()

            relevant_data["DaysDiff"] = (current_date - relevant_data["Date"]).dt.days.abs()

            nearest_5_days = relevant_data.nsmallest(5, "DaysDiff")

            avg_value = nearest_5_days[feature].mean()
            avg_values.append(round(nearest_5_days[feature].mean(), 2))

            if pd.isna(avg_value):
                print(f"{idx+1}, {feature}: {avg_values[idx]}")

        train_df[f"Avg_5D_{feature}"] = avg_values

    output_file = os.path.join(output_dir, "avg_train_data_lag_closest.csv")
    train_df.dropna(inplace=True)
    train_df.to_csv(output_file, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
