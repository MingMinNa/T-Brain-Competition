import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# 定義函數
# =============================================================================
def parse_serial(df, serial_column="Serial"):
    """
    解析指定的序號欄位並新增多個日期和裝置相關的特徵。
    序號格式為 YYYYMMDDHHMMXX，其中 XX 為裝置代碼。
    """
    if serial_column not in df.columns:
        raise KeyError(f"欄位 '{serial_column}' 不存在於資料中。請確認資料包含此欄位。")

    df[serial_column] = df[serial_column].astype(str)
    # 解析 Datetime，前12個字符為日期時間，遇到錯誤則設為 NaT
    df["Datetime"] = pd.to_datetime(df[serial_column].str[:12], format="%Y%m%d%H%M", errors="coerce")
    # 提取 DeviceID，最後2個字符
    df["DeviceID"] = df[serial_column].str[-2:]
    # 提取月份、小時和分鐘
    df["Month"] = df["Datetime"].dt.month
    df["Hour"] = df["Datetime"].dt.hour
    df["Minute"] = df["Datetime"].dt.minute
    return df


def one_hot_encode_device(df, all_device_ids):
    """
    對 DeviceID 進行 One-Hot 編碼。
    """
    df["DeviceID"] = df["DeviceID"].astype(str).str.zfill(2)  # 確保 DeviceID 為兩位數
    for device in all_device_ids:
        col_name = f"DeviceID_{device}"
        df[col_name] = (df["DeviceID"] == device).astype(int)
    return df


# =============================================================================
# 預測流程
# =============================================================================
def predict_power():
    # =============================================================================
    # 1. 定義路徑
    # =============================================================================
    project_root = os.getcwd()

    # 模型與Scaler的路徑
    lstm_model_filename = "WheatherLSTM_2024-11-21T00_29_47Z.keras"  # 請確保文件名稱正確
    regression_model_filename = "WheatherRegression_2024-11-21T00_29_47Z.joblib"  # 請確保文件名稱正確
    scaler_filename = "scaler.joblib"  # 請確保文件名稱正確

    # 測試資料路徑
    submission_path = os.path.join(project_root, "Submission", "upload.csv")

    # 輸出資料夾路徑
    output_dir = os.path.join(project_root, "Output-CSV")
    os.makedirs(output_dir, exist_ok=True)

    # =============================================================================
    # 2. 載入模型和 Scaler
    # =============================================================================
    lstm_model_path = os.path.join(output_dir, lstm_model_filename)
    regression_model_path = os.path.join(output_dir, regression_model_filename)
    scaler_path = os.path.join(output_dir, scaler_filename)

    if not os.path.exists(lstm_model_path):
        raise FileNotFoundError(f"LSTM 模型文件 {lstm_model_path} 不存在。")
    if not os.path.exists(regression_model_path):
        raise FileNotFoundError(f"回歸模型文件 {regression_model_path} 不存在。")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler 文件 {scaler_path} 不存在。")

    print("載入 LSTM 模型...")
    lstm_model = load_model(lstm_model_path)
    print("LSTM 模型載入完成。")

    print("載入回歸模型...")
    regression_model = joblib.load(regression_model_path)
    print("回歸模型載入完成。")

    print("載入 Scaler...")
    scaler = joblib.load(scaler_path)
    print("Scaler 載入完成。")

    # =============================================================================
    # 3. 載入測試資料
    # =============================================================================
    print("載入測試資料...")
    submission_df = pd.read_csv(submission_path, encoding="utf-8")
    target = ["序號"]
    EXquestion = submission_df[target].values
    print(f"測試資料載入完成，共 {len(EXquestion)} 筆資料。")

    # =============================================================================
    # 4. 獲取所有 DeviceID
    # =============================================================================
    # 假設所有 DeviceID 在訓練階段已知，並保存於 feature_columns.txt
    feature_columns_path = os.path.join(output_dir, "feature_columns.txt")
    if not os.path.exists(feature_columns_path):
        raise FileNotFoundError(f"特徵欄位文件 {feature_columns_path} 不存在。")

    with open(feature_columns_path, "r", encoding="utf-8") as f:
        feature_columns = [line.strip() for line in f.readlines()]

    # 假設數值特徵與 One-Hot 特徵分開定義
    numerical_features = [
        "WindSpeed(m/s)",
        "Pressure(hpa)",
        "Temperature(°C)",
        "Humidity(%)",
        "Sunlight(Lux)",
        "Hour",
        "Minute",
        "Month",
    ]
    one_hot_features = [col for col in feature_columns if col.startswith("DeviceID_")]
    all_features = numerical_features + one_hot_features

    # 獲取所有 DeviceID
    device_id_columns = [col for col in one_hot_features]
    all_device_ids = [col.replace("DeviceID_", "") for col in device_id_columns]

    # =============================================================================
    # 5. 初始化參數
    # =============================================================================
    LookBackNum = 12  # 前兩小時的資料，假設每筆資料為10分鐘
    ForecastNum = 48  # 預測筆數（競賽要求）

    PredictPower = []  # 存放預測值(發電量)

    # =============================================================================
    # 6. 預測迴圈
    # =============================================================================
    DataName = os.getcwd() + r"\ExampleTestData\upload.csv"
    SourceData = pd.read_csv(DataName, encoding="utf-8")
    target = ["序號"]
    EXquestion = SourceData[target].values

    inputs = []  # 存放參考資料
    PredictOutput = []  # 存放預測值(天氣參數)
    PredictPower = []  # 存放預測值(發電量)

    count = 0
    while count < len(EXquestion):
        print("count : ", count)
        LocationCode = int(EXquestion[count])
        strLocationCode = str(LocationCode)[-2:]
        if LocationCode < 10:
            strLocationCode = "0" + LocationCode

        DataName = os.getcwd() + "\TrainData(IncompleteAVG)\IncompleteAvgDATA_" + strLocationCode + ".csv"
        SourceData = pd.read_csv(DataName, encoding="utf-8")
        ReferTitle = SourceData[["Serial"]].values
        ReferData = SourceData[
            ["WindSpeed(m/s)", "Pressure(hpa)", "Temperature(°C)", "Humidity(%)", "Sunlight(Lux)"]
        ].values

        inputs = []  # 重置存放參考資料

        # 找到相同的一天，把12個資料都加進inputs
        for DaysCount in range(len(ReferTitle)):
            if str(int(ReferTitle[DaysCount]))[:8] == str(int(EXquestion[count]))[:8]:
                TempData = ReferData[DaysCount].reshape(1, -1)
                TempData = LSTM_MinMaxModel.transform(TempData)
                inputs.append(TempData)

        # 用迴圈不斷使新的預測值塞入參考資料，並預測下一筆資料
        for i in range(ForecastNum):

            # print(i)

            # 將新的預測值加入參考資料(用自己的預測值往前看)
            if i > 0:
                inputs.append(PredictOutput[i - 1].reshape(1, 5))

            # 切出新的參考資料12筆(往前看12筆)
            X_test = []
            X_test.append(inputs[0 + i : LookBackNum + i])

            # Reshaping
            NewTest = np.array(X_test)
            NewTest = np.reshape(NewTest, (NewTest.shape[0], NewTest.shape[1], 5))

            predicted = regressor.predict(NewTest)
            PredictOutput.append(predicted)
            PredictPower.append(np.round(Regression.predict(predicted), 2).flatten())

        # 每次預測都要預測48個，因此加48個會切到下一天
        # 0~47,48~95,96~143...
        count += 48
        
    # =============================================================================
    # 7. 生成提交文件
    # =============================================================================
    # 根據預測數量調整 submission_df
    submission_df = submission_df.iloc[: len(PredictPower)]
    submission_df["答案"] = PredictPower
    output_csv_path = os.path.join(output_dir, "ans.csv")
    submission_df.to_csv(output_csv_path, index=False, float_format="%.2f")
    print(f"提交文件已保存至 {output_csv_path}")


if __name__ == "__main__":
    predict_power()
