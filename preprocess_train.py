from datetime import datetime

import pandas as pd
import numpy as np
import os

project_root = os.getcwd()
avg_data_dir = os.path.join(project_root, "TrainData(AVG)")
incomplete_avg_data_dir = os.path.join(project_root, "TrainData(IncompleteAVG)")
additional_dir = os.path.join(project_root, "AdditionalTrainData")
submission_dir = os.path.join(project_root, "Submission")
output_csv_dir = os.path.join(project_root, "TrainData")


def parse_serial(df, additional_data_df, serial_column="Serial"):
    df[serial_column] = df[serial_column].astype(str)
    additional_data_df[serial_column] = additional_data_df[serial_column].astype(str)

    additional_cols = ["Serial", "ElevationAngle(degree)", "Azimuth(degree)"]
    df = df.merge(additional_data_df[additional_cols], left_on="Serial", right_on="Serial", how="left")

    MAX_LUX = 117758.2
    df["SunlightSaturated"] = (df["Sunlight(Lux)"] >= MAX_LUX).astype(int)

    df["Datetime"] = pd.to_datetime(df[serial_column].str[:12], format="%Y%m%d%H%M", errors="coerce")
    df["Date"] = df["Datetime"].dt.date
    df["Month"] = df["Datetime"].dt.month
    df["Hour"] = df["Datetime"].dt.hour
    df["Minute"] = df["Datetime"].dt.minute
    df["DeviceID"] = df[serial_column].str[-2:]
    return df


def parse_submission_serial(df, serial_column="Serial"):
    df[serial_column] = df[serial_column].astype(str)
    df["Datetime"] = pd.to_datetime(df[serial_column].str[:12], format="%Y%m%d%H%M", errors="coerce")
    df["Date"] = df["Datetime"].dt.date
    df["Month"] = df["Datetime"].dt.month
    df["Hour"] = df["Datetime"].dt.hour
    df["Minute"] = df["Datetime"].dt.minute
    df["DeviceID"] = df[serial_column].str[-2:]
    return df


def one_hot_encode_device(df, all_device_ids):
    df["DeviceID"] = df["DeviceID"].astype(str).str.zfill(2)
    for device in all_device_ids:
        col_name = f"DeviceID_{device}"
        df[col_name] = (df["DeviceID"] == device).astype(int)
    return df


def load_train_data(train_avg_dir, train_incomplete_avg_dir, train_additional_dir):
    all_data_avg = []
    for i in range(1, 18):
        file_name = f"AvgDATA_{i:02d}.csv"
        file_path = os.path.join(train_avg_dir, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, encoding="utf-8")
            all_data_avg.append(df)

    all_data_incomplete = []
    for i in range(1, 18):
        file_name = f"IncompleteAvgDATA_{i:02d}.csv"
        file_path = os.path.join(train_incomplete_avg_dir, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, encoding="utf-8")
            all_data_incomplete.append(df)

    all_data_additional = []
    for i in range(1, 18):
        file_name = f"AdditionalTrainDATA_{i:02d}.csv"
        file_path = os.path.join(train_additional_dir, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, encoding="utf-8")
            all_data_additional.append(df)

    combined_avg = pd.concat(all_data_avg, ignore_index=True) if all_data_avg else pd.DataFrame()

    combined_incomplete = pd.concat(all_data_incomplete, ignore_index=True) if all_data_incomplete else pd.DataFrame()

    combined_additional = pd.concat(all_data_additional, ignore_index=True) if all_data_additional else pd.DataFrame()

    combined_data = (
        pd.concat([combined_avg, combined_incomplete], ignore_index=True)
        if not combined_incomplete.empty
        else combined_avg
    )

    return combined_data, combined_avg, combined_incomplete, combined_additional


def process_and_save(df, additional_data_df, output_file):
    df = parse_serial(df, additional_data_df, serial_column="Serial")

    unique_device_ids = df["DeviceID"].unique()
    df = one_hot_encode_device(df, unique_device_ids)

    df = df.sort_values(by=["DeviceID", "Datetime"]).reset_index(drop=True)

    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Data saved to {output_file}")


def process_and_save_submission(df, output_file):
    df = parse_submission_serial(df, serial_column="Serial")
    df = df.drop(columns=["Answer"], errors="ignore")
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Data saved to {output_file}")


def main():
    # Load data
    combined_df, combined_avg_df, combined_incomplete_df, combined_additional_df = load_train_data(
        avg_data_dir, incomplete_avg_data_dir, additional_dir
    )

    submission_path = os.path.join(submission_dir, "upload.csv")
    submission_df = pd.read_csv(submission_path, encoding="utf-8")

    print(f"Total records loaded (combined): {len(combined_df)}")
    print(f"Total records loaded (AVG): {len(combined_avg_df)}")
    print(f"Total records loaded (Incomplete AVG): {len(combined_incomplete_df)}")
    print(f"Total records loaded (Additional): {len(combined_additional_df)}")
    print(f"Total records loaded (Submission): {len(submission_df)}")

    # Define output files
    output_file_combined = os.path.join(output_csv_dir, "total_train_data.csv")
    output_file_complete = os.path.join(output_csv_dir, "avg_train_data.csv")
    output_file_incomplete = os.path.join(output_csv_dir, "incomplete_avg_train_data.csv")
    output_file_submission = os.path.join(submission_dir, "submission.csv")

    # Process and save combined data
    process_and_save(combined_df, combined_additional_df, output_file_combined)
    process_and_save(combined_avg_df, combined_additional_df, output_file_complete)
    process_and_save(combined_incomplete_df, combined_additional_df, output_file_incomplete)
    process_and_save_submission(submission_df, output_file_submission)
    print("Data processing complete.")


if __name__ == "__main__":
    main()
