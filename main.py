import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from utils import related_data, const, data_preprocess
from utils.models import regression_nn, xgboost, random_forest, ensemble_nn


def main():
    q_date_df = pd.read_csv(os.path.join(const.SUBMISSION_FOLDER, 'q_date.csv'), encoding = 'utf-8')
    current_location = None

    regression_model = xgboost_model = rf_model = ensemble_model = None
    input_features = 'Pressure(hpa),Temperature(°C),Humidity(%),Sunlight(Lux),ElevationAngle,Azimuth'.split(',')

    output_df = pd.DataFrame(columns = const.ans_df_columns)
    for idx in tqdm(q_date_df.index):
        
        if current_location != q_date_df.loc[idx, 'location']:
            train_data = related_data.get_all_data(location = q_date_df.loc[idx, 'location'], src = 'AVG')
            train_data = data_preprocess.data_transform(train_data)
            X_train, y_train = data_preprocess.preprocess(train_data, input_features = input_features)

            current_location = q_date_df.loc[idx, 'location']
            features_df = pd.read_csv(os.path.join(const.GENERATE_FOLDER, 'Additional_Features', f'AdditionalTrainData_{int(current_location):02d}.csv'))
            features_df['Serial'] = features_df['Serial'].astype(str)
            features_df.set_index('Serial', inplace = True)

            if regression_model is not None:    del regression_model
            if xgboost_model is not None:       del xgboost_model
            if rf_model is not None:            del rf_model
            if ensemble_model is not None:      del ensemble_model

            model_path = os.path.join(const.PROJECT_FOLDER, 'models', 'regression', f'{current_location}.pth')
            if os.path.exists(model_path):
                regression_model = regression_nn.load_model(model_path)
            else:
                regression_model = regression_nn.build_model(X_train, y_train, input_features = input_features, epochs = 75)
                regression_nn.save_model(model_path, regression_model)

            model_path = os.path.join(const.PROJECT_FOLDER, 'models', 'xgboost', f'{current_location}.json')
            if os.path.exists(model_path):
                xgboost_model = xgboost.load_model(model_path)
            else:
                xgboost_model  = xgboost.build_model(X_train, y_train, input_features = input_features)
                xgboost.save_model(model_path, xgboost_model)

            regression_prediction = regression_nn.predict(regression_model, X_train)
            xgb_prediction = xgboost.predict(xgboost_model, input_features, X_train)

            for i in X_train.index:
                X_train.loc[i, 'regression_prediction'] = regression_prediction[i].item()
                X_train.loc[i, 'xgb_prediction'] = xgb_prediction[i].item()
            X_train.loc[:, 'Power'] = y_train
            
            model_path = os.path.join(const.PROJECT_FOLDER, 'models', 'ensemble', f'{current_location}.pth')

            if os.path.exists(model_path):
                ensemble_model = ensemble_nn.load_model(model_path)
            else:
                ensemble_model = ensemble_nn.build_model(X_train, y_train, epochs = 100, input_features = input_features + ['regression_prediction', 'xgb_prediction'])
                ensemble_nn.save_model(model_path, ensemble_model)

            ensemble_prediction = ensemble_nn.predict(ensemble_model, X_train)
            for i in X_train.index:
                X_train.loc[i, 'ensemble_prediction'] = ensemble_prediction[i].item()

        date = str(q_date_df.loc[idx, 'date'])
        input_df = pd.read_csv(os.path.join(const.SUBMISSION_FOLDER, 'input.csv'))
        input_df = input_df[(input_df['Date'] == f"{date[:4]}-{date[4:6]}-{date[6:]}") & (input_df['location'].astype(int) == int(q_date_df.loc[idx, 'location']))]
        X_test, _ = data_preprocess.preprocess(input_df, input_features = input_features)
        X_test = X_test.reset_index(drop = True).astype('float32')
                                                        
        regression_prediction = regression_nn.predict(regression_model, X_test)
        xgb_prediction = xgboost.predict(xgboost_model, input_features, X_test)

        for i in X_test.index:
            X_test.loc[i, 'regression_prediction'] = regression_prediction[i].item()
            X_test.loc[i, 'xgb_prediction'] = xgb_prediction[i].item()
        ensemble_prediction = ensemble_nn.predict(ensemble_model, X_test)
        for i in X_test.index:
            X_test.loc[i, 'ensemble_prediction'] = ensemble_prediction[i].item()

        date = str(q_date_df.loc[idx, 'date'])
        location =  int(q_date_df.loc[idx, 'location'])
        result_df = pd.DataFrame(columns = const.ans_df_columns)
        for i in X_test.index:
            result_df.loc[i, :] = [f"{date}{int(X_test.loc[i, 'Hour']):02d}{int(X_test.loc[i, 'Minute']):02d}{location:02d}", f"{ensemble_prediction[i].item():.2f}"]
        
        output_df = pd.concat([output_df, result_df], axis = 0, ignore_index = True) if output_df.shape[0] > 0 else result_df
        output_df.to_csv(os.path.join(const.SUBMISSION_FOLDER, 'output.csv'), index=False, encoding="utf-8-sig")

if __name__ == '__main__':
    main()