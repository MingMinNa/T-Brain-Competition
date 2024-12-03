from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import randint
import pandas as pd

from .. import const

def build_model(X_train, y_train, random_seed = 42):
    
    param_dist = {
        'n_estimators': randint(100, 501),          # 樹的數量，範圍 100 到 500
        'max_depth': randint(3, 20),                # 樹的最大深度，範圍 3 到 20
        'min_samples_split': randint(2, 10),        # 節點劃分的最小樣本數
        'min_samples_leaf': randint(1, 5),          # 葉子節點的最小樣本數
        'max_features': [0.6, 0.7, 0.8, 0.9, 1.0],  # 隨機選取的特徵比例（0.6~1.0）
        'bootstrap': [True, False],                 # 是否有放回地抽樣
    }
    tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = train_test_split(
        X_train, y_train, test_size = 0.2, random_state = random_seed, shuffle = True
    )

    rf_model = RandomForestRegressor(random_state = random_seed)
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_dist,
        n_iter = 50,  
        cv = 3,        
        scoring='neg_mean_absolute_error',  
        random_state = 42,
        verbose = 0,
        n_jobs = -1
    )

    random_search.fit(X_train, y_train)
    
    print("Best parameters found: ", random_search.best_params_)
    print("Best CV Score (negative MAE): ", random_search.best_score_)

    best_rf_model = RandomForestRegressor(
        random_state = random_seed,
        **random_search.best_params_
    )

    best_rf_model.fit(X_train, y_train)
    y_val_pred = best_rf_model.predict(tmp_X_test)
    mae = mean_absolute_error(tmp_y_test, y_val_pred)
    print(f"Validation MAE: {mae:.4f}")

    return best_rf_model

def predict(model, X_test, date, location):
    features = 'Pressure(hpa),Temperature(°C),Humidity(%),Sunlight(Lux),ElevationAngle,Azimuth'.split(',')

    result_df = pd.DataFrame(columns = const.ans_df_columns)
    copy_X_test = X_test[features].astype('float32')
    
    predictions = model.predict(copy_X_test)
    for i in X_test.index:
        result_df.loc[i, :] = [f"{date}{int(X_test.loc[i, 'Hour']):02d}{int(X_test.loc[i, 'Minute']):02d}{location:02d}", f"{predictions[i].item():.2f}"]

    return result_df


if __name__ == '__main__':
    pass