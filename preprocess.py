from datetime import datetime

import pandas as pd
import numpy as np
import os

# Define project directories
project_root = os.getcwd()
train_avg_path = os.path.join(project_root, "TrainData(AVG)")
train_incomplete_avg_path = os.path.join(project_root, "TrainData(IncompleteAVG)")
output_csv_path = os.path.join(project_root, "Output-CSV")


def parse_serial(df, serial_column="Serial"):
    """
    Extracts time-related information from the 'Serial' column, including Datetime, DeviceID, Month, Hour, Season, and PartOfDay.
    """
    df[serial_column] = df[serial_column].astype(str)

    df["Datetime"] = pd.to_datetime(df[serial_column].str[:12], format="%Y%m%d%H%M", errors="coerce")

    df["DeviceID"] = df[serial_column].str[-2:]

    df["Month"] = df["Datetime"].dt.month
    df["Hour"] = df["Datetime"].dt.hour
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


def load_train_data(train_avg_dir, train_incomplete_avg_dir):
    """
    Loads all CSV files from 'TrainData(AVG)' and 'TrainData(IncompleteAVG)' directories and merges them into a single DataFrame.
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

    # Combine all average data
    combined_avg = pd.concat(all_data_avg, ignore_index=True) if all_data_avg else pd.DataFrame()
    # Combine all incomplete average data
    combined_incomplete = pd.concat(all_data_incomplete, ignore_index=True) if all_data_incomplete else pd.DataFrame()
    # Merge the two datasets
    combined_data = pd.concat([combined_avg, combined_incomplete], ignore_index=True)

    return combined_data


def main():
    # Load the data
    combined_df = load_train_data(train_avg_path, train_incomplete_avg_path)
    print(f"Total records loaded: {len(combined_df)}")

    # Parse the 'Serial' column
    combined_df = parse_serial(combined_df, serial_column="Serial")

    # One-Hot Encode 'DeviceID'
    unique_device_ids = combined_df["DeviceID"].unique()
    combined_df = one_hot_encode_device(combined_df, unique_device_ids)

    # Sort the data by 'DeviceID' and 'Datetime'
    combined_df = combined_df.sort_values(by=["DeviceID", "Datetime"]).reset_index(drop=True)

    # Remove original 'DeviceID' and 'Datetime' columns if not needed
    columns_to_drop = ["DeviceID", "Datetime"]
    combined_df = combined_df.drop(columns=columns_to_drop, errors="ignore")

    # Create output directory if it doesn't exist
    os.makedirs(output_csv_path, exist_ok=True)

    # Save the processed DataFrame to a CSV file
    output_file = os.path.join(output_csv_path, "train_data.csv")
    combined_df.to_csv(output_file, index=False, encoding="utf-8")
    print("Data processing complete.")


if __name__ == "__main__":
    main()
