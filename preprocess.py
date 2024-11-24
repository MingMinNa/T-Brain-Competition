from datetime import datetime

import pandas as pd
import numpy as np
import os

# Define project directories
project_root = os.getcwd()
additional_data_path = os.path.join(project_root, "AdditionalData", "additional_data.csv")
avg_data_dir = os.path.join(project_root, "TrainData(AVG)")
incomplete_avg_data_dir = os.path.join(project_root, "TrainData(IncompleteAVG)")
output_csv_dir = os.path.join(project_root, "TrainData")


def parse_serial(df, additional_data_df, serial_column="Serial"):
    """
    Extracts time-related information from the 'Serial' column, including Datetime, DeviceID, Month, Hour, Season, and PartOfDay.
    """
    df[serial_column] = df[serial_column].astype(str)
    df["Date"] = df[serial_column].str[:8]

    additional_data_df["日期"] = additional_data_df["日期"].astype(str)
    additional_cols = ["日期", "日出方位角", "日中天仰角", "日落方位角"]
    df = df.merge(additional_data_df[additional_cols], left_on="Date", right_on="日期", how="left")
    df.drop(columns=["Date", "日期"], inplace=True)

    df["Datetime"] = pd.to_datetime(df[serial_column].str[:12], format="%Y%m%d%H%M", errors="coerce")
    df["Month"] = df["Datetime"].dt.month
    df["Hour"] = df["Datetime"].dt.hour
    df["Minute"] = df["Datetime"].dt.minute
    df["DeviceID"] = df[serial_column].str[-2:]
    return df


def one_hot_encode_device(df, all_device_ids):
    """
    Performs One-Hot Encoding on the 'DeviceID' column.
    """
    df["DeviceID"] = df["DeviceID"].astype(str).str.zfill(2)  # Ensure DeviceID is two digits
    for device in all_device_ids:
        col_name = f"DeviceID_{device}"
        df[col_name] = (df["DeviceID"] == device).astype(int)
    return df


def load_train_data(train_avg_dir, train_incomplete_avg_dir, additional_data_path):
    """
    Loads all CSV files from 'TrainData(AVG)' and 'TrainData(IncompleteAVG)' directories and merges them into separate DataFrames.
    """
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

    if os.path.exists(additional_data_path):
        additional_data = pd.read_csv(additional_data_path, encoding="utf-8")

    combined_avg = pd.concat(all_data_avg, ignore_index=True) if all_data_avg else pd.DataFrame()

    combined_incomplete = pd.concat(all_data_incomplete, ignore_index=True) if all_data_incomplete else pd.DataFrame()

    combined_data = (
        pd.concat([combined_avg, combined_incomplete], ignore_index=True)
        if not combined_incomplete.empty
        else combined_avg
    )

    return combined_data, combined_avg, combined_incomplete, additional_data


def process_and_save(df, additional_data_df, output_file):
    """
    Processes the DataFrame by parsing the 'Serial' column, one-hot encoding 'DeviceID',
    sorting, dropping unnecessary columns, and saving to a CSV file.

    Args:
        df (DataFrame): The DataFrame to process.
        output_file (str): The path to save the processed CSV.
    """
    df = parse_serial(df, additional_data_df, serial_column="Serial")

    unique_device_ids = df["DeviceID"].unique()
    df = one_hot_encode_device(df, unique_device_ids)

    df = df.sort_values(by=["DeviceID", "Datetime"]).reset_index(drop=True)

    columns_to_drop = ["Datetime"]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Data saved to {output_file}")


def main():
    # Load the data
    combined_df, combined_avg_df, combined_incomplete_df, additional_data_df = load_train_data(
        avg_data_dir, incomplete_avg_data_dir, additional_data_path
    )
    print(f"Total records loaded (combined): {len(combined_df)}")
    print(f"Total records loaded (AVG): {len(combined_avg_df)}")
    print(f"Total records loaded (Incomplete AVG): {len(combined_incomplete_df)}")
    print(f"Total records loaded (Additional): {len(additional_data_df)}")

    # Define output files
    output_file_combined = os.path.join(output_csv_dir, "total_train_data.csv")
    output_file_complete = os.path.join(output_csv_dir, "avg_train_data.csv")
    output_file_incomplete = os.path.join(output_csv_dir, "incomplete_avg_train_data.csv")

    # Process and save combined data
    process_and_save(combined_df, additional_data_df, output_file_combined)
    process_and_save(combined_avg_df, additional_data_df, output_file_complete)
    process_and_save(combined_incomplete_df, additional_data_df, output_file_incomplete)
    print("Data processing complete.")


if __name__ == "__main__":
    main()
