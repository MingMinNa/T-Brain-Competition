import pandas as pd
import os

# 設定檔案路徑
submission_path = os.path.join(os.getcwd(), "SubmissionTemplate", "upload.csv")
output_path = os.path.join(os.getcwd(), "output.csv")
answer_path = os.path.join(os.getcwd(), "answer.csv")

# 讀取 SubmissionTemplate\upload.csv 並提取 "序號" 列
try:
    submission_df = pd.read_csv(submission_path, encoding="utf-8-sig")
    serial_numbers = submission_df["序號"]
except FileNotFoundError:
    print(f"錯誤：找不到檔案 {submission_path}")
    exit(1)
except KeyError:
    print(f"錯誤：檔案 {submission_path} 中沒有 '序號' 列")
    exit(1)

# 讀取 output.csv 並提取 "答案" 列
try:
    output_df = pd.read_csv(output_path, encoding="utf-8-sig")
    answers = output_df["答案"]
except FileNotFoundError:
    print(f"錯誤：找不到檔案 {output_path}")
    exit(1)
except KeyError:
    print(f"錯誤：檔案 {output_path} 中沒有 '答案' 列")
    exit(1)

# 檢查兩個列的長度是否相同
if len(serial_numbers) != len(answers):
    print("錯誤：'序號' 列和 '答案' 列的長度不一致，無法合併。")
    exit(1)

# 創建新的 DataFrame，包含 "序號" 和 "答案" 列
answer_df = pd.DataFrame({"序號": serial_numbers, "答案": answers})

# 將新的 DataFrame 保存為 answer.csv
try:
    answer_df.to_csv(answer_path, index=False, encoding="utf-8-sig")
    print(f"成功：已將 '序號' 和 '答案' 合併並保存為 {answer_path}")
except Exception as e:
    print(f"錯誤：無法保存 {answer_path}。原因：{e}")
