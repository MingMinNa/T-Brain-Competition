import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from scipy.stats import uniform, randint

from .. import const

def build_model(X_train, y_train, random_seed = 42):

    feature_columns = ['Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 
                       'Sunlight(Lux)', 'ElevationAngle', 'Azimuth']
    X_train = X_train[feature_columns]
    
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_seed)
    
    # 設定參數搜索範圍
    param_dist = {
        'learning_rate': uniform(0.01, 0.3),  # 取值範圍 0.01 到 0.31
        'max_depth': randint(3, 10),          # 隨機整數，範圍 3 到 9
        'n_estimators': randint(100, 501),    # 隨機整數，範圍 100 到 500
        'subsample': uniform(0.6, 0.4),       # 取值範圍 0.6 到 1.0
        'colsample_bytree': uniform(0.6, 0.4),# 取值範圍 0.6 到 1.0
        'reg_alpha': uniform(0, 1),           # 取值範圍 0 到 1
        'reg_lambda': uniform(1, 2),          # 取值範圍 1 到 3
    }
    
    # 初始化模型
    xgb_model = xgb.XGBRegressor(
        objective='reg:absoluteerror',  
        random_state=random_seed
    )
    
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter = 50,  
        scoring = 'neg_mean_absolute_error', 
        cv = 3,  
        verbose = 1,
        random_state = random_seed,
        n_jobs = -1 
    )
    
    random_search.fit(X_train_split, y_train_split)
    print("Best Parameters:", random_search.best_params_)
    print("Best CV Score (negative MAE):", random_search.best_score_)
    
    best_model = random_search.best_estimator_
    best_model.fit(X_train, y_train)
    
    val_predictions = best_model.predict(X_val_split)
    val_mae = mean_absolute_error(y_val_split, val_predictions)
    print(f"[best rf model] Validation MAE: {val_mae:.4f}")
    
    return best_model


def predict(model, X_test):

    features = 'Pressure(hpa),Temperature(°C),Humidity(%),Sunlight(Lux),ElevationAngle,Azimuth'.split(',')

    copy_X_test = X_test[features].astype('float32')
    
    # 預測
    predictions = model.predict(copy_X_test)
    return predictions

if __name__ == '__main__':
    pass