import pandas as pd

# 讀取 upload.csv，假設檔案編碼為 UTF-8
try:
    upload_df = pd.read_csv("upload.csv", encoding="utf-8")
except UnicodeDecodeError:
    # 如果 UTF-8 編碼失敗，可以嘗試其他編碼，如 'gbk'
    upload_df = pd.read_csv("upload.csv", encoding="gbk")
except FileNotFoundError:
    print("錯誤：找不到 'upload.csv' 檔案。")
    exit(1)

# 讀取 output.csv，假設檔案編碼為 UTF-8
try:
    output_df = pd.read_csv("output.csv", encoding="utf-8")
except UnicodeDecodeError:
    # 如果 UTF-8 編碼失敗，可以嘗試其他編碼，如 'gbk'
    output_df = pd.read_csv("output.csv", encoding="gbk")
except FileNotFoundError:
    print("錯誤：找不到 'output.csv' 檔案。")
    exit(1)

# 確認 '序號' 和 '答案' 欄位存在
if "序號" not in upload_df.columns:
    print("錯誤：'upload.csv' 中找不到 '序號' 欄位。")
    exit(1)

if "答案" not in output_df.columns:
    print("錯誤：'output.csv' 中找不到 '答案' 欄位。")
    exit(1)

# 選取需要的欄位
serial_numbers = upload_df["序號"]
answers = output_df["答案"]

# 確保兩個欄位的長度相同
if len(serial_numbers) != len(answers):
    print("警告：'序號' 和 '答案' 欄位的行數不一致。進行合併時將以最小的行數為準。")

min_length = min(len(serial_numbers), len(answers))
serial_numbers = serial_numbers.iloc[:min_length]
answers = answers.iloc[:min_length]

# 建立新的 DataFrame
merged_df = pd.DataFrame({"序號": serial_numbers, "答案": answers})

# 儲存合併後的結果到新的 CSV 檔案
merged_df.to_csv("merged_output.csv", index=False, encoding="utf-8-sig")  # 使用 'utf-8-sig' 以確保 Excel 正確顯示

print("合併完成！結果已儲存為 'merged_output.csv'。")
