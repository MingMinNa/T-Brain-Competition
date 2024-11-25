import pandas as pd
import os


project_root = os.getcwd()
train_data_path = os.path.join(project_root, "TrainData", "avg_train_data_closest.csv")
output_dir = os.path.join(project_root, "TrainData")
os.makedirs(output_dir, exist_ok=True)


def main():
    train_df = pd.read_csv(train_data_path)
    train_df["Datetime"] = pd.to_datetime(train_df["Datetime"])
    train_df = train_df.sort_values(by=["DeviceID", "Datetime"])

    lags = [1, 3, 6, 12]

    for lag in lags:
        lag_col_name = f"lag_{lag}_Power(mW)"
        train_df[lag_col_name] = train_df.groupby(["DeviceID", train_df["Datetime"].dt.date])["Power(mW)"].shift(lag)

    train_df.dropna(inplace=True)

    output_file = os.path.join(output_dir, "avg_train_data_closest_lag.csv")
    train_df.to_csv(output_file, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
