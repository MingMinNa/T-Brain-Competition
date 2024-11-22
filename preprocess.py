from datetime import datetime

import pandas as pd
import numpy as np
import os

# Define project directories
project_root = os.getcwd()
train_avg_path = os.path.join(project_root, "TrainData(AVG)")
train_incomplete_avg_path = os.path.join(project_root, "TrainData(IncompleteAVG)")
output_csv_path = os.path.join(project_root, "TrainData")


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
    Loads all CSV files from 'TrainData(AVG)' and 'TrainData(IncompleteAVG)' directories and merges them into separate DataFrames.

    Returns:
        combined_data (DataFrame): Combined data from both average and incomplete average datasets.
        combined_incomplete (DataFrame): Combined data from incomplete average datasets only.
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
    combined_data = (
        pd.concat([combined_avg, combined_incomplete], ignore_index=True)
        if not combined_incomplete.empty
        else combined_avg
    )

    return combined_data, combined_incomplete


def process_and_save(df, output_file):
    """
    Processes the DataFrame by parsing the 'Serial' column, one-hot encoding 'DeviceID',
    sorting, dropping unnecessary columns, and saving to a CSV file.

    Args:
        df (DataFrame): The DataFrame to process.
        output_file (str): The path to save the processed CSV.
    """
    # Parse the 'Serial' column
    df = parse_serial(df, serial_column="Serial")

    # One-Hot Encode 'DeviceID'
    unique_device_ids = df["DeviceID"].unique()
    df = one_hot_encode_device(df, unique_device_ids)

    # Sort the data by 'DeviceID' and 'Datetime'
    df = df.sort_values(by=["DeviceID", "Datetime"]).reset_index(drop=True)

    # Remove original 'DeviceID' and 'Datetime' columns if not needed
    columns_to_drop = ["DeviceID", "Datetime"]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the processed DataFrame to a CSV file
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Data saved to {output_file}")


def main():
    # Load the data
    combined_df, combined_incomplete_df = load_train_data(train_avg_path, train_incomplete_avg_path)
    print(f"Total records loaded (combined): {len(combined_df)}")
    print(f"Total records loaded (Incomplete AVG): {len(combined_incomplete_df)}")

    # Define output files
    output_file_combined = os.path.join(output_csv_path, "train_data.csv")
    output_file_incomplete = os.path.join(output_csv_path, "incomplete_avg_train_data.csv")

    # Process and save combined data
    if not combined_df.empty:
        process_and_save(combined_df, output_file_combined)
    else:
        print("No data found in combined datasets to save.")

    # Process and save incomplete average data
    if not combined_incomplete_df.empty:
        process_and_save(combined_incomplete_df, output_file_incomplete)
    else:
        print("No data found in Incomplete AVG datasets to save.")

    print("Data processing complete.")


if __name__ == "__main__":
    main()
