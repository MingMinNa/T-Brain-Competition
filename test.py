import os
import pandas as pd

# 設定專案根目錄和 CSV 檔案路徑
project_root = os.getcwd()
additional_data_path = os.path.join(project_root, "AdditionalData", "additional_data_08.csv")

# 讀取 CSV
try:
    data = pd.read_csv(additional_data_path)
except FileNotFoundError:
    print(f"Error: File not found at {additional_data_path}")
    exit()

# 確認 "裝置id" 欄位是否存在
if "裝置ID" not in data.columns:
    print("Error: The column '裝置id' does not exist in the CSV file.")
    exit()

# 修改 "裝置id" 欄位，將其全部設為 '08'
data["裝置id"] = "08"

# 保存修改後的 CSV
data.to_csv(additional_data_path, index=False)
print(f"Updated CSV saved to {additional_data_path}")
