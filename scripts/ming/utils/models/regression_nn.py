import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import copy

from .. import const

device = const.device

class RegressionModel(nn.Module):
    def __init__(self, input_dim, input_features):
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
        self.input_features = input_features
    
    def forward(self, x):
        return self.fc(x)


def build_model(X_train, y_train, input_features,
                random_seed = 42, learning_rate = 0.005, epochs = 150,batch_size = 32):
    
    X_train = X_train[input_features]
    input_dim = X_train.shape[1]

    torch.manual_seed(random_seed)
    
    model = RegressionModel(input_dim, input_features).to(device)
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=1e-5)
    

    tmp_X_train, tmp_X_val, tmp_y_train, tmp_y_val = train_test_split(X_train, y_train, test_size = 0.2, shuffle = True, random_state = random_seed)

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
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    all_loader = torch.utils.data.DataLoader(all_dataset, batch_size = batch_size, shuffle = True)
    

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)  # Add learning rate scheduler

    min_val_loss = np.inf
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in all_loader:
            optimizer.zero_grad()
            predictions = model(batch_X.unsqueeze(1)).squeeze()
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if epoch % 5 != 0: continue
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
        
        if epoch_val_loss < min_val_loss:
            min_val_loss = epoch_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

    best_model = RegressionModel(X_train.shape[1], input_features).to(device)
    best_model.load_state_dict(best_model_state)
    for param in best_model.parameters():
        param.requires_grad = False
    
    best_model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = best_model(batch_X.unsqueeze(1)).squeeze()
            loss = criterion(predictions, batch_y)
            epoch_val_loss += loss.item()

    print(f"Best Validation Loss: {epoch_val_loss:.4f}")

    return best_model

def predict(model, X_test):
    input_features = model.input_features
    
    copy_X_test = X_test[input_features].astype('float32')

    model = model.to(device)
    X_test_tensor = torch.tensor(copy_X_test.to_numpy(), dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu()

    return predictions


def load_model(model_path):
    regression_model = torch.load(model_path, weights_only = False)
    return regression_model

def save_model(model_path, regression_model):
    torch.save(regression_model, model_path)
    return

if __name__ == '__main__':
    pass