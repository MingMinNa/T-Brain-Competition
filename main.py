import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime

# =============================================================================
# 1. 定義路徑
# =============================================================================
# 定義專案根目錄
project_root = os.getcwd()

# 訓練資料路徑
train_avg_path = os.path.join(project_root, "TrainData(AVG)")
train_incomplete_avg_path = os.path.join(project_root, "TrainData(IncompleteAVG)")

# 測試資料路徑
submission_path = os.path.join(project_root, "Submission", "upload.csv")

# 輸出資料夾路徑
output_dir = os.path.join(project_root, "Output-CSV")
os.makedirs(output_dir, exist_ok=True)


# =============================================================================
# 2. 定義解析 Serial 函數並新增更多特徵
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
    df["DeviceID"] = df[serial_column].str[-2:].zfill(2)
    # 提取月份、小時和分鐘
    df["Month"] = df["Datetime"].dt.month
    df["Hour"] = df["Datetime"].dt.hour
    df["Minute"] = df["Datetime"].dt.minute
    return df


# =============================================================================
# 3. 定義 One-Hot 編碼函數
# =============================================================================
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
# 4. 載入並合併訓練資料
# =============================================================================
def load_train_data(train_avg_dir, train_incomplete_avg_dir):
    """
    載入 TrainData(AVG) 和 TrainData(IncompleteAVG) 資料夾中的所有 CSV 檔案，並合併為一個 DataFrame。
    """
    all_data_avg = []
    for i in range(1, 18):  # 1 到 17
        file_name = f"AvgDATA_{i:02d}.csv"
        file_path = os.path.join(train_avg_dir, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, encoding="utf-8")
            all_data_avg.append(df)
        else:
            print(f"警告: 文件 {file_path} 不存在，跳過。")

    all_data_incomplete = []
    for i in range(1, 18):
        file_name = f"IncompleteAvgDATA_{i:02d}.csv"
        file_path = os.path.join(train_incomplete_avg_dir, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, encoding="utf-8")
            all_data_incomplete.append(df)
        else:
            print(f"警告: 文件 {file_path} 不存在，跳過。")

    # 合併所有訓練資料
    if all_data_avg:
        combined_avg = pd.concat(all_data_avg, ignore_index=True)
    else:
        combined_avg = pd.DataFrame()
        print("警告: 沒有可用的 AvgDATA 檔案。")

    if all_data_incomplete:
        combined_incomplete = pd.concat(all_data_incomplete, ignore_index=True)
    else:
        combined_incomplete = pd.DataFrame()
        print("警告: 沒有可用的 IncompleteAvgDATA 檔案。")

    # 檢查是否有資料被讀取
    if combined_avg.empty and combined_incomplete.empty:
        raise ValueError("沒有任何訓練資料被載入。請檢查資料夾路徑和檔案存在性。")

    # 解析 Serial 並新增特徵
    combined_avg = parse_serial(combined_avg)  # 使用預設的 'Serial'
    combined_incomplete = parse_serial(combined_incomplete)  # 使用預設的 'Serial'

    # 獲取所有 DeviceID
    all_device_ids = pd.concat([combined_avg["DeviceID"], combined_incomplete["DeviceID"]]).unique()
    print(f"所有 DeviceID 數量: {len(all_device_ids)}")
    print(f"DeviceIDs: {all_device_ids}")

    # One-Hot 編碼 DeviceID
    combined_avg = one_hot_encode_device(combined_avg, all_device_ids)
    combined_incomplete = one_hot_encode_device(combined_incomplete, all_device_ids)
    print("One-Hot 編碼 DeviceID 完成。")

    # 合併所有訓練資料
    combined_train_df = pd.concat([combined_avg, combined_incomplete], ignore_index=True)
    print(f"合併後的訓練資料數量: {combined_train_df.shape}")

    return combined_train_df, all_device_ids


# =============================================================================
# 5. 資料預處理
# =============================================================================
def preprocess_data(combined_train_df, numerical_features, one_hot_features, target_column, scaler_path):
    """
    提取特徵和目標，並對數值特徵進行正規化。
    """
    # 定義特徵和目標
    FEATURES = numerical_features + one_hot_features
    TARGET = target_column

    # 確保所有 FEATURES 都存在於 DataFrame 中
    missing_features = [feat for feat in FEATURES if feat not in combined_train_df.columns]
    if missing_features:
        raise ValueError(f"缺少以下特徵欄位：{missing_features}")

    # 提取特徵和目標
    X = combined_train_df[FEATURES].values
    y = combined_train_df[TARGET].values.reshape(-1, 1)

    # 確保所有特徵都是數值類型
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # 正規化數值特徵
    scaler = MinMaxScaler()
    X[:, : len(numerical_features)] = scaler.fit_transform(X[:, : len(numerical_features)])

    # 保存 scaler 以便測試階段使用
    joblib.dump(scaler, scaler_path)
    print("數值特徵正規化完成並保存。")

    return X, y, scaler


# =============================================================================
# 6. 構建 LSTM 的輸入
# =============================================================================
def create_lstm_input(X, y, look_back):
    """
    根據 look_back 生成 LSTM 的輸入序列和目標。
    """
    X_train, y_train = [], []
    for i in range(look_back, len(X)):
        X_train.append(X[i - look_back : i, :])
        y_train.append(y[i])
    return np.array(X_train), np.array(y_train)


# =============================================================================
# 7. 建立並訓練 LSTM 模型
# =============================================================================
def build_and_train_lstm(X_train, y_train, output_dir, epochs=100, batch_size=128):
    """
    建立並訓練 LSTM 模型，並保存模型。
    """
    model = Sequential()
    model.add(
        LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), activation="relu")
    )
    model.add(LSTM(units=64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # 預測發電量

    model.compile(optimizer="adam", loss="mean_squared_error")

    # 設置早停
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    # 訓練模型
    history = model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[early_stop], verbose=1
    )

    # 保存模型，使用 .keras 格式
    now_datetime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
    model_filename = f"WheatherLSTM_{now_datetime}.keras"  # 修改副檔名
    model_path = os.path.join(output_dir, model_filename)
    model.save(model_path)
    print(f"LSTM 模型已保存至 {model_path}")

    return model, model_path


# =============================================================================
# 8. 建立並訓練回歸模型
# =============================================================================
def build_and_train_regression(X_train, y_train, output_dir):
    """
    建立並訓練回歸模型，並保存模型。
    """
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    print("回歸模型訓練完成。")

    # 保存回歸模型
    now_datetime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
    regression_model_filename = f"WheatherRegression_{now_datetime}.joblib"
    regression_model_path = os.path.join(output_dir, regression_model_filename)
    joblib.dump(regression_model, regression_model_path)
    print(f"回歸模型已保存至 {regression_model_path}")

    # 打印模型參數
    print("回歸模型截距 (Intercept):", regression_model.intercept_)
    print("回歸模型係數 (Coefficients):", regression_model.coef_)
    print("回歸模型 R squared:", regression_model.score(X_train, y_train))

    return regression_model, regression_model_path


# =============================================================================
# 9. 載入並預處理測試資料
# =============================================================================
def load_and_preprocess_test_data(submission_path, all_device_ids, numerical_features, one_hot_features, scaler):
    """
    載入測試資料並進行預處理。
    """
    submission_df = pd.read_csv(submission_path, encoding="utf-8")
    submission_df = parse_serial(submission_df, serial_column="序號")
    submission_df = one_hot_encode_device(submission_df, all_device_ids)

    # 提取特徵
    FEATURES = numerical_features + one_hot_features
    missing_features = [feat for feat in FEATURES if feat not in submission_df.columns]
    if missing_features:
        raise ValueError(f"測試資料缺少以下特徵欄位：{missing_features}")

    X_test = submission_df[FEATURES].values.astype(np.float32)
    X_test[:, : len(numerical_features)] = scaler.transform(X_test[:, : len(numerical_features)])

    return submission_df, X_test, FEATURES


# =============================================================================
# 10. 創建 LSTM 測試輸入序列
# =============================================================================
def create_lstm_test_input(X_test, look_back):
    """
    創建 LSTM 測試輸入序列。
    """
    X_test_sequences = []
    for i in range(len(X_test) - look_back + 1):
        X_test_sequences.append(X_test[i : i + look_back, :])
    return np.array(X_test_sequences)


# =============================================================================
# 11. 生成提交文件
# =============================================================================
def generate_submission_file(submission_df, predicted_power, look_back, output_dir):
    """
    根據預測結果生成提交文件。
    """
    # 提取預測對應的序號
    predicted_serials = submission_df["序號"].values[look_back:]

    # 構建 DataFrame
    output_df = pd.DataFrame({"序號": predicted_serials, "答案": predicted_power})

    # 保存為 CSV，並確保答案四捨五入至兩位小數
    output_csv_path = os.path.join(output_dir, "ans.csv")  # 將檔名改為 ans.csv
    output_df.to_csv(output_csv_path, index=False, float_format="%.2f")
    print(f"提交文件已保存至 {output_csv_path}")


# =============================================================================
# 12. 主流程
# =============================================================================
def main():
    # =============================================================================
    # 1. 載入並合併訓練資料
    # =============================================================================
    combined_train_df, all_device_ids = load_train_data(train_avg_path, train_incomplete_avg_path)
    print("訓練資料載入並合併完成。")

    # =============================================================================
    # 2. 定義特徵列表
    # =============================================================================
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

    # 獲取所有 One-Hot 編碼後的特徵名稱（DeviceID_*)
    one_hot_features = [col for col in combined_train_df.columns if col.startswith("DeviceID_")]

    target_column = "Power(mW)"

    # =============================================================================
    # 3. 資料預處理
    # =============================================================================
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    X, y, scaler = preprocess_data(combined_train_df, numerical_features, one_hot_features, target_column, scaler_path)
    print("資料預處理完成。")

    # =============================================================================
    # 4. 保存特徵欄位名稱
    # =============================================================================
    feature_columns = numerical_features + one_hot_features
    feature_columns_path = os.path.join(output_dir, "feature_columns.txt")
    with open(feature_columns_path, "w", encoding="utf-8") as f:
        for col in feature_columns:
            f.write(f"{col}\n")
    print(f"特徵欄位名稱已保存至 {feature_columns_path}")

    # =============================================================================
    # 5. 將資料切分為訓練集和驗證集
    # =============================================================================
    # 這裡，我們先切分資料，再創建 LSTM 序列，並保留原始的訓練資料以用於回歸模型
    X_train_original, X_val_original, y_train_original, y_val_original = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"資料切分完成。訓練集樣本數量: {X_train_original.shape[0]}, 驗證集樣本數量: {X_val_original.shape[0]}")

    # =============================================================================
    # 6. 創建 LSTM 的輸入序列
    # =============================================================================
    LookBackNum = 12  # 前兩小時的資料，假設每筆資料為10分鐘
    ForecastNum = 48  # 預測筆數（競賽要求）

    X_train_lstm, y_train_lstm = create_lstm_input(X_train_original, y_train_original, LookBackNum)
    X_val_lstm, y_val_lstm = create_lstm_input(X_val_original, y_val_original, LookBackNum)

    print(
        f"LSTM 訓練資料序列化完成。X_train_lstm shape: {X_train_lstm.shape}, y_train_lstm shape: {y_train_lstm.shape}"
    )
    print(f"LSTM 驗證資料序列化完成。X_val_lstm shape: {X_val_lstm.shape}, y_val_lstm shape: {y_val_lstm.shape}")

    # =============================================================================
    # 7. 建立並訓練 LSTM 模型
    # =============================================================================
    lstm_model, lstm_model_path = build_and_train_lstm(X_train_lstm, y_train_lstm, output_dir)
    print("LSTM 模型訓練完成。")

    # =============================================================================
    # 8. 建立並訓練回歸模型
    # =============================================================================
    regression_model, regression_model_path = build_and_train_regression(X_train_original, y_train_original, output_dir)
    print("回歸模型訓練完成。")

    # =============================================================================
    # 9. 載入並預處理測試資料
    # =============================================================================
    submission_df, X_test, FEATURES = load_and_preprocess_test_data(
        submission_path, all_device_ids, numerical_features, one_hot_features, scaler
    )
    print("測試資料載入並預處理完成。")

    # =============================================================================
    # 10. 創建 LSTM 測試輸入序列
    # =============================================================================
    X_test_sequences = create_lstm_test_input(X_test, LookBackNum)
    print(f"LSTM 測試輸入序列完成。X_test_sequences shape: {X_test_sequences.shape}")

    # =============================================================================
    # 11. 載入已保存的模型
    # =============================================================================
    # 如果需要重新載入模型，可以取消以下註解
    # lstm_model = load_model(lstm_model_path)
    # regression_model = joblib.load(regression_model_path)

    # =============================================================================
    # 12. 進行預測
    # =============================================================================
    # 使用訓練好的 LSTM 模型進行預測
    y_pred_lstm = lstm_model.predict(X_test_sequences)
    print(f"LSTM 預測完成，共預測 {len(y_pred_lstm)} 筆資料。")

    # 使用訓練好的回歸模型進行預測
    y_pred_regression = regression_model.predict(X_test)
    y_pred_regression = np.round(y_pred_regression, 2).flatten()
    print(f"回歸模型預測完成，共預測 {len(y_pred_regression)} 筆資料。")

    # =============================================================================
    # 13. 結合預測結果
    # =============================================================================
    # 這裡我們採用簡單的平均作為結合策略，您也可以根據需要採用其他方法
    # 確保兩個預測結果的長度一致
    min_length = min(len(y_pred_lstm), len(y_pred_regression))
    y_pred_combined = (y_pred_lstm[:min_length].flatten() + y_pred_regression[:min_length]) / 2
    y_pred_combined = np.round(y_pred_combined, 2)
    print("預測結果已結合完成。")

    # =============================================================================
    # 14. 生成提交文件
    # =============================================================================
    generate_submission_file(submission_df, y_pred_combined, LookBackNum, output_dir)
    print("提交文件已生成。")


if __name__ == "__main__":
    main()
