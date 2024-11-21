import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def data_transform(train_data):
    '''
    train_data = train_data.assign(
        # Year = train_data['Serial'].astype(str).str[:4].astype(int),
        # Month = train_data['Serial'].astype(str).str[4:6].astype(int),
        Day = train_data['Serial'].astype(str).str[6:8].astype(int),
        Hour = train_data['Serial'].astype(str).str[8:10].astype(int),
        Minute = train_data['Serial'].astype(str).str[10:12].astype(int),
    )
    '''
    # train_data.loc[:, 'Time'] = train_data.apply(lambda row: int(str(row['Serial'])[8:10])*60 + int(str(row['Serial'])[10:12]), axis=1)
    
    # remove 'WindSpeed(m/s)' 
    train_data = train_data.drop(columns = ['Serial', 'WindSpeed(m/s)']) 
    return train_data

def preprocess(train_data):

    X_train, y_train = None, None
    if 'Power(mW)' in train_data.columns:
        y_train = train_data['Power(mW)']
        y_train = y_train.astype('float32')
        train_data.drop(columns = ['Power(mW)'], inplace = True)
        X_train = train_data['Pressure(hpa),Temperature(°C),Humidity(%),Sunlight(Lux)'.split(',')]
    else:        
        X_train = train_data['Pressure(hpa),Temperature(°C),Humidity(%),Sunlight(Lux),Day,Hour,Minute'.split(',')]


    '''
    numeric_features = 'Pressure(hpa),Temperature(°C),Humidity(%),Sunlight(Lux)'.split(',')
    numeric_data = X_train[numeric_features]
    imputer = SimpleImputer(strategy = "mean")
    imputed_numeric_data = imputer.fit_transform(numeric_data)
    
    scaler = StandardScaler()
    scaled_numeric_data = scaler.fit_transform(imputed_numeric_data)
    X_train.loc[:, numeric_features] = scaled_numeric_data
    X_train = X_train.astype('float32')
    '''
    return X_train, y_train

if __name__ == '__main__':
    pass