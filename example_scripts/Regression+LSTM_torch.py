#%% Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
 
# 設定LSTM往前看的筆數和預測筆數
LookBackNum = 12  # LSTM往前看的筆數
ForecastNum = 48  # 預測筆數

# 載入訓練資料
FOLDER_PATH = os.path.dirname(__file__)
DataName = os.path.join(FOLDER_PATH, 'ExampleTrainData(AVG)', 'AvgDATA_17.csv')
SourceData = pd.read_csv(DataName, encoding='utf-8')

# 迴歸分析選擇要留下來的資料欄位
# (風速,大氣壓力,溫度,濕度,光照度)
Regression_X_train = SourceData[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']].values
Regression_y_train = SourceData[['Power(mW)']].values

# LSTM 選擇要留下來的資料欄位
AllOutPut = SourceData[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']].values

# 正規化
LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)
AllOutPut_MinMax = LSTM_MinMaxModel.transform(AllOutPut)

X_train = []
y_train = []

# 設定每i-12筆資料(X_train)就對應到第i筆資料(y_train)
for i in range(LookBackNum, len(AllOutPut_MinMax)):
    X_train.append(AllOutPut_MinMax[i - LookBackNum:i, :])
    y_train.append(AllOutPut_MinMax[i, :])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping (samples 是訓練樣本數量, timesteps 是每個樣本的時間步長, features 是每個時間步的特徵數量)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))

#%% LSTM Model (PyTorch)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Instantiate and train the LSTM model
input_size = 5
hidden_size = 128
output_size = 5

# Model, Loss, Optimizer
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Train the model
epochs = 100
batch_size = 128
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), os.path.join(FOLDER_PATH, 'WeatherLSTM.pth'))
print("LSTM Model Saved")

#%% Train Linear Regression Model (sklearn remains the same)

# Start regression analysis (對發電量做迴歸)
RegressionModel = LinearRegression()
RegressionModel.fit(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train)

# Save regression model
NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
joblib.dump(RegressionModel, os.path.join(FOLDER_PATH, f'WeatherRegression_{NowDateTime}'))

# Get intercept and coefficients
print('Intercept: ', RegressionModel.intercept_)
print('Coefficients: ', RegressionModel.coef_)

# Get R-squared
print('R-squared: ', RegressionModel.score(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train))

#%% Prediction with LSTM and Regression models

# Load the models
model = LSTMModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load(os.path.join(FOLDER_PATH, 'WeatherLSTM.pth'), weights_only = True))
model.eval()

Regression = joblib.load(os.path.join(FOLDER_PATH, f'WeatherRegression_{NowDateTime}'))

# Load test data
DataName = os.path.join(FOLDER_PATH, 'ExampleTestData', 'upload.csv')
SourceData = pd.read_csv(DataName, encoding='utf-8')
target = ['序號']
EXquestion = SourceData[target].values

inputs = []  # 存放參考資料
PredictOutput = []  # 存放預測值(天氣參數)
PredictPower = []  # 存放預測值(發電量)

count = 0
while(count < len(EXquestion)):
    print('count : ', count)
    LocationCode = int(EXquestion[count].squeeze())
    strLocationCode = str(LocationCode)[-2:]
    if LocationCode < 10:
        strLocationCode = '0' + str(LocationCode)

    DataName = os.path.join(FOLDER_PATH, 'ExampleTrainData(IncompleteAVG)', f'IncompleteAvgDATA_{strLocationCode}.csv')
    SourceData = pd.read_csv(DataName, encoding='utf-8')
    ReferTitle = SourceData[['Serial']].values
    ReferData = SourceData[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']].values

    inputs = []  # 重置存放參考資料

    # 找到相同的一天，把12個資料都加進inputs
    for DaysCount in range(len(ReferTitle)):
        if str(int(ReferTitle[DaysCount].squeeze()))[:8] == str(int(EXquestion[count].squeeze()))[:8]:
            TempData = ReferData[DaysCount].reshape(1, -1)
            TempData = LSTM_MinMaxModel.transform(TempData)
            inputs.append(TempData)

    # 用迴圈不斷使新的預測值塞入參考資料，並預測下一筆資料
    for i in range(ForecastNum):
        #print(i)

        #將新的預測值加入參考資料(用自己的預測值往前看)
        if i > 0:
            inputs.append(PredictOutput[i - 1].reshape(1, 5))

        # 切出新的參考資料12筆(往前看12筆)
        X_test = []
        X_test.append(inputs[0 + i:LookBackNum + i])

        # Reshaping
        NewTest = np.array(X_test)
        NewTest = np.reshape(NewTest, (NewTest.shape[0], NewTest.shape[1], 5))

        # Convert to torch tensor
        NewTest_tensor = torch.tensor(NewTest, dtype=torch.float32)

        # Predict with LSTM model
        predicted = model(NewTest_tensor).detach().numpy()
        PredictOutput.append(predicted)
        PredictPower.append(np.round(Regression.predict(predicted), 2).flatten())

    # 每次預測都要預測48個，因此加48個會切到下一天
    # 0~47, 48~95, 96~143...
    count += 48

# Write the prediction results into a new CSV file
df = pd.DataFrame(PredictPower, columns=['答案'])
df.to_csv(os.path.join(FOLDER_PATH, 'output.csv'), index=False)
print('Output CSV File Saved')

# %%
