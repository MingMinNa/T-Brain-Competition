import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from utils import const, data_preprocess
from utils.models import regression_nn, xgboost, ensemble_nn


def main():

    # Load the test data file for predictions
    test_date_df = pd.read_csv(os.path.join(const.SUBMISSION_FOLDER, 'test_date.csv'))

    current_location = None

    regression_model = xgboost_model = ensemble_model = None
    input_features = 'Pressure(hpa),Temperature(°C),Humidity(%),Sunlight(Lux),ElevationAngle,Azimuth'.split(',')

    output_df = pd.DataFrame(columns = const.ans_df_columns)
    
    for idx in tqdm(test_date_df.index):
        
        if current_location != test_date_df.loc[idx, 'location']:
            train_data = pd.read_csv(os.path.join(const.GENERATE_AVG_FOLDER, f'AvgDATA_{int(test_date_df.loc[idx, 'location']):02d}.csv'), encoding = 'utf-8')
            X_train, y_train = data_preprocess.preprocess(train_data, input_features = input_features, is_train = True)

            current_location = test_date_df.loc[idx, 'location']

            # Clean up previously used models to save memory
            if regression_model is not None:    del regression_model
            if xgboost_model is not None:       del xgboost_model
            if ensemble_model is not None:      del ensemble_model


            # Load or build the regression model
            model_path = os.path.join(const.MODELS_FOLDER, 'regression', f'{current_location}.pth')
            if os.path.exists(model_path):
                regression_model = regression_nn.load_model(model_path)
            else:
                regression_model = regression_nn.build_model(X_train, y_train, input_features = input_features, epochs = 75)
                regression_nn.save_model(model_path, regression_model)

            # Load or build the XGBoost model
            model_path = os.path.join(const.MODELS_FOLDER, 'xgboost', f'{current_location}.json')
            if os.path.exists(model_path):
                xgboost_model = xgboost.load_model(model_path)
            else:
                xgboost_model  = xgboost.build_model(X_train, y_train, input_features = input_features)
                xgboost.save_model(model_path, xgboost_model)

            regression_prediction = regression_nn.predict(regression_model, X_train)
            xgb_prediction = xgboost.predict(xgboost_model, input_features, X_train)

            # Combine predictions into X_train to prepare for ensemble model training
            for i in X_train.index:
                X_train.loc[i, 'regression_prediction'] = regression_prediction[i].item()
                X_train.loc[i, 'xgb_prediction'] = xgb_prediction[i].item()
            
            # Load or build the ensemble model
            model_path = os.path.join(const.MODELS_FOLDER, 'ensemble', f'{current_location}.pth')
            if os.path.exists(model_path):
                ensemble_model = ensemble_nn.load_model(model_path)
            else:
                ensemble_model = ensemble_nn.build_model(X_train, y_train, epochs = 100, input_features = input_features + ['regression_prediction', 'xgb_prediction'])
                ensemble_nn.save_model(model_path, ensemble_model)

            ensemble_prediction = ensemble_nn.predict(ensemble_model, X_train)

            for i in X_train.index:
                X_train.loc[i, 'ensemble_prediction'] = ensemble_prediction[i].item()
            
            # Optionally save the training results for observation
            # X_train.to_csv('train_result.csv',index = False)

        # Prepare test data for the current date and location
        date = str(test_date_df.loc[idx, 'date'])
        input_df = pd.read_csv(os.path.join(const.SUBMISSION_FOLDER, 'input_features', 'input_2.csv'))
        input_df = input_df[(input_df['Date'] == f"{date[:4]}-{date[4:6]}-{date[6:]}") & (input_df['Location'].astype(int) == int(test_date_df.loc[idx, 'location']))]
        X_test, _ = data_preprocess.preprocess(input_df, input_features = input_features, is_train = False)
        X_test = X_test.reset_index(drop = True).astype('float32')

        # Predict using regression and XGBoost models
        regression_prediction = regression_nn.predict(regression_model, X_test)
        xgb_prediction = xgboost.predict(xgboost_model, input_features, X_test)

        # Add model predictions to test data for ensemble prediction
        for i in X_test.index:
            X_test.loc[i, 'regression_prediction'] = regression_prediction[i].item()
            X_test.loc[i, 'xgb_prediction'] = xgb_prediction[i].item()
        
        ensemble_prediction = ensemble_nn.predict(ensemble_model, X_test)
        for i in X_test.index:
            X_test.loc[i, 'ensemble_prediction'] = ensemble_prediction[i].item()

        date = str(test_date_df.loc[idx, 'date'])
        location =  int(test_date_df.loc[idx, 'location'])

        # Format the results into the output DataFrame
        result_df = pd.DataFrame(columns = const.ans_df_columns)
        for i in X_test.index:
            result_df.loc[i, :] = [f"{date}{int(X_test.loc[i, 'Hour']):02d}{int(X_test.loc[i, 'Minute']):02d}{location:02d}", f"{ensemble_prediction[i].item():.2f}"]
        
        # Append results to the final output DataFrame
        output_df = pd.concat([output_df, result_df], axis = 0, ignore_index = True) if output_df.shape[0] > 0 else result_df
        output_df.to_csv(os.path.join(const.SUBMISSION_FOLDER, 'output.csv'), index = False, encoding = "utf-8-sig")

if __name__ == '__main__':
    # create save_models folder to save the built model
    if not os.path.exists(os.path.join(const.PROJECT_FOLDER, 'save_models')):
        os.mkdir(os.path.join(const.PROJECT_FOLDER, 'save_models'))
    if not os.path.exists(os.path.join(const.MODELS_FOLDER)):
        os.mkdir(os.path.join(const.MODELS_FOLDER))

    for model_name in ['regression', 'ensemble', 'xgboost']:
        if not os.path.exists(os.path.join(const.MODELS_FOLDER, model_name)):
            os.mkdir(os.path.join(const.MODELS_FOLDER, model_name))
    
    main()