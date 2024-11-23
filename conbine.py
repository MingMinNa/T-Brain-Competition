import pandas as pd
import os

project_root = os.getcwd()
submission_path = os.path.join(project_root, "Submission", "upload.csv")
output_dir = os.path.join(project_root, "Output")
os.makedirs(output_dir, exist_ok=True)

upload_df = pd.read_csv(submission_path, encoding="utf-8")
output_df = pd.read_csv("output.csv", encoding="utf-8")

serial_numbers = upload_df["序號"]
answers = output_df["答案"]

min_length = min(len(serial_numbers), len(answers))
serial_numbers = serial_numbers.iloc[:min_length]
answers = answers.iloc[:min_length]

merged_df = pd.DataFrame({"序號": serial_numbers, "答案": answers})
merged_df.to_csv("merged_output.csv", index=False, encoding="utf-8-sig")

print("合併完成！")
