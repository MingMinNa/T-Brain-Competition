import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import const

device = const.device

class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.fc(x)


def build_model(X_train, y_train, random_seed = 42, learning_rate = 0.005, epochs = 150,batch_size = 32):
    
    X_train = X_train['Pressure(hpa),Temperature(°C),Humidity(%),Sunlight(Lux)'.split(',')]
    input_dim = X_train.shape[1]
    
    # 初始化模型
    model = RegressionModel(input_dim).to(device)
    
    # 定義損失函數和優化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=1e-5)
    

    tmp_X_train, tmp_X_val, tmp_y_train, tmp_y_val = train_test_split(X_train, y_train, test_size = 0.2, shuffle = True, random_state = random_seed)
    # 構建數據加載器
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(tmp_X_train.values).to(device), 
        torch.FloatTensor(tmp_y_train.values).to(device)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(tmp_X_val.values).to(device), 
        torch.FloatTensor(tmp_y_val.values).to(device)
    )
    all_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train.values).to(device), 
        torch.FloatTensor(y_train.values).to(device)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    all_loader = torch.utils.data.DataLoader(all_dataset, batch_size = batch_size, shuffle = False)
    

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)  # Add learning rate scheduler

    # 開始訓練
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X.unsqueeze(1)).squeeze()
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if epoch % 10 != 0: continue
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                predictions = model(batch_X.unsqueeze(1)).squeeze()
                loss = criterion(predictions, batch_y)
                epoch_val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Validation Loss: {epoch_val_loss:.4f}")
            
    return model

def predict(model, X_test, date, location):
    
    copy_X_test = X_test['Pressure(hpa),Temperature(°C),Humidity(%),Sunlight(Lux)'.split(',')].astype('float32')
    result_df = pd.DataFrame(columns = const.ans_df_columns)

    X_test_tensor = torch.tensor(copy_X_test.to_numpy(), dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu()

    for i in X_test.index:
        result_df.loc[i, :] = [f"{date}{int(X_test.loc[i, 'Hour']):02d}{int(X_test.loc[i, 'Minute']):02d}{location:02d}", f"{predictions[i].item():.2f}"]

    return result_df

if __name__ == '__main__':
    pass