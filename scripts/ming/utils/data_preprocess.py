import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess(train_data, input_features, is_train = True):

    if is_train:
        X_train = train_data[input_features]
        # fill NaN values in each row of X_train
        for i in X_train[X_train.isna().any(axis=1)].index:
            for col in X_train.columns:
                if pd.isna(X_train.loc[i, col]):
                    try:    X_train.loc[i, col] = X_train.loc[i - 1, col]
                    except: X_train.loc[i, col] = X_train.loc[i + 1, col]

        y_train = train_data['Power(mW)'].astype('float32')
    else:
        X_train = train_data[input_features + ['Minute', 'Hour']]
        y_train = None
    
    return X_train, y_train

if __name__ == '__main__':
    pass