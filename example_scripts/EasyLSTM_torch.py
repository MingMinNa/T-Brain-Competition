#%%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

# 設定 LSTM 往前看的筆數和預測筆數
LookBackNum = 12  # LSTM 往前看的筆數
ForecastNum = 48  # 預測筆數


FOLDER_PATH = os.path.dirname(__file__)
# 載入訓練資料
DataName = os.path.join(FOLDER_PATH, 'ExampleTrainData(AVG)', 'AvgDATA_17.csv')
SourceData = pd.read_csv(DataName, encoding='utf-8')

# 選擇要留下來的資料欄位 (發電量)
target = ['Power(mW)']
AllOutPut = SourceData[target].values

X_train = []
y_train = []

# 設定每 i-12 筆資料 (X_train) 就對應到第 i 筆資料 (y_train)
for i in range(LookBackNum, len(AllOutPut)):
    X_train.append(AllOutPut[i-LookBackNum:i, 0])
    y_train.append(AllOutPut[i, 0])

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# 將資料轉為 PyTorch 的張量
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)

#%%
# ============================ 建置 & 訓練模型 ============================
# 定義 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x[:, -1, :])  # 取最後一個時間步的輸出
        x = self.fc(x)
        return x



# 初始化模型、損失函數與優化器
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 開始訓練
epochs = 100
batch_size = 128

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(X_train):.6f}")

# 保存模型
torch.save(model.state_dict(), os.path.join(FOLDER_PATH, 'WeatherLSTM.pth'))
print('Model Saved')


#%%
# ============================ 預測數據 ============================

# 載入模型
model = LSTMModel()
model.load_state_dict(torch.load(os.path.join(FOLDER_PATH, 'WeatherLSTM.pth'), weights_only = True))
model.eval()

# 載入測試資料
DataName = os.path.join(FOLDER_PATH, 'ExampleTestData', 'upload.csv')
SourceData = pd.read_csv(DataName, encoding='utf-8')
target = ['序號']
EXquestion = SourceData[target].values

inputs = []  # 存放參考資料
PredictOutput = []  # 存放預測值

count = 0
while count < len(EXquestion):
    print('count : ', count)
    LocationCode = int(EXquestion[count].squeeze())
    strLocationCode = str(LocationCode)[-2:]
    if LocationCode < 10:
        strLocationCode = '0' + str(LocationCode)

    DataName = os.path.join(FOLDER_PATH, 'ExampleTrainData(IncompleteAVG)', f'IncompleteAvgDATA_{strLocationCode}.csv')
    SourceData = pd.read_csv(DataName, encoding='utf-8')
    ReferTitle = SourceData[['Serial']].values
    ReferData = SourceData[['Power(mW)']].values

    inputs = []  # 重置存放參考資料

    # 找到相同的一天，把 12 個資料都加進 inputs
    for DaysCount in range(len(ReferTitle)):
        if str(int(ReferTitle[DaysCount].squeeze()))[:8] == str(int(EXquestion[count].squeeze()))[:8]:
            inputs = np.append(inputs, ReferData[DaysCount])

    # 用迴圈不斷使新的預測值塞入參考資料，並預測下一筆資料
    for i in range(ForecastNum):

        # 將新的預測值加入參考資料 (用自己的預測值往前看)
        if i > 0:
            inputs = np.append(inputs, PredictOutput[i-1])

        # 切出新的參考資料 12 筆 (往前看 12 筆)
        X_test = []
        X_test.append(inputs[0+i:LookBackNum+i])

        # Reshaping
        NewTest = np.array(X_test, dtype=np.float32)
        NewTest = np.reshape(NewTest, (NewTest.shape[0], NewTest.shape[1], 1))
        NewTest = torch.tensor(NewTest)

        with torch.no_grad():
            predicted = model(NewTest)
        PredictOutput.append(round(predicted[0, 0].item(), 2))

    count += 48

# 寫預測結果寫成新的 CSV 檔案
# 將陣列轉換為 DataFrame
df = pd.DataFrame(PredictOutput, columns=['答案'])

# 將 DataFrame 寫入 CSV 檔案
df.to_csv(os.path.join(FOLDER_PATH, 'output.csv'), index=False)
print('Output CSV File Saved')

# %%
