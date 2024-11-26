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

    output_df = pd.read_csv(os.path.join(const.SUBMISSION_FOLDER, 'output.csv'))
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
            
            regression_model = regression_nn.build_model(X_train, y_train, 
                                                         input_features = input_features,
                                                         epochs = 75)
            xgboost_model  = xgboost.build_model(X_train, y_train)
            regression_prediction = regression_nn.predict(regression_model, X_train)
            xgb_prediction = xgboost.predict(xgboost_model, X_train)

            for i in X_train.index:
                X_train.loc[i, 'regression_prediction'] = regression_prediction[i].item()
                X_train.loc[i, 'xgb_prediction'] = xgb_prediction[i].item()
            X_train.loc[:, 'Power'] = y_train
            
            ensemble_model = ensemble_nn.build_model(X_train, y_train, epochs = 100,
                                                     input_features = input_features + ['regression_prediction', 'xgb_prediction'])
            
            ensemble_prediction = ensemble_nn.predict(ensemble_model, X_train)
            for i in X_train.index:
                X_train.loc[i, 'ensemble_prediction'] = ensemble_prediction[i].item()
            X_train.to_csv(os.path.join(const.PROJECT_FOLDER, 'TEST', f'X_TRAIN_{current_location}.csv'), index = False)
            # rf_model  = random_forest.build_model(X_train, y_train)

        recent_df = related_data.get_date_data(target_date = str(q_date_df.loc[idx, 'date']), location = q_date_df.loc[idx, 'location'], day_diff = 5, src = 'AVG')
        input_df = related_data.merge_by_time(recent_df, date = str(q_date_df.loc[idx, 'date']))

        # complement the empty day with average value
        for time in range(9 * 60, 17 * 60, 10):
            if time not in input_df:
                input_df.loc[time, :] = input_df.mean()

        input_df = input_df.reset_index(drop = False)
        input_df = input_df.assign(
            Hour = input_df.loc[:, 'Time'].astype(int) // 60,
            Minute = np.mod(input_df.loc[:, 'Time'].astype(int), 60),
        )
        input_df.drop(columns = ['Time'], inplace = True)

        X_test, _ = data_preprocess.preprocess(input_df, input_features = input_features)

        for i in X_test.index:
            serial = f"{q_date_df.loc[idx, 'date']}{int(X_test.loc[i, 'Hour']):02d}{int(X_test.loc[i, 'Minute']):02d}{int(q_date_df.loc[idx, 'location']):02d}"
            if serial in features_df.index:
                for col in features_df.columns:
                    if pd.isna(features_df.loc[serial, col]) == True:  continue
                    X_test.loc[i, col] = features_df.loc[serial, col]
        X_test = X_test.astype('float32')
        
        regression_prediction = regression_nn.predict(regression_model, X_test)
        xgb_prediction = xgboost.predict(xgboost_model, X_test)

        for i in X_test.index:
            X_test.loc[i, 'regression_prediction'] = regression_prediction[i].item()
            X_test.loc[i, 'xgb_prediction'] = xgb_prediction[i].item()
        ensemble_prediction = ensemble_nn.predict(ensemble_model, X_test)
        for i in X_test.index:
            X_test.loc[i, 'ensemble_prediction'] = ensemble_prediction[i].item()
        X_test.to_csv(os.path.join(const.PROJECT_FOLDER, 'TEST', f'X_TEST_{idx}.csv'), index = False)

        date = str(q_date_df.loc[idx, 'date'])
        location =  int(q_date_df.loc[idx, 'location'])
        result_df = pd.DataFrame(columns = const.ans_df_columns)
        for i in X_test.index:
            result_df.loc[i, :] = [f"{date}{int(X_test.loc[i, 'Hour']):02d}{int(X_test.loc[i, 'Minute']):02d}{location:02d}", f"{ensemble_prediction[i].item():.2f}"]
        
        output_df = pd.concat([output_df, result_df], axis = 0, ignore_index = True) if output_df.shape[0] > 0 else result_df
        output_df.to_csv(os.path.join(const.SUBMISSION_FOLDER, 'output.csv'), index=False, encoding="utf-8-sig")

if __name__ == '__main__':
    main()