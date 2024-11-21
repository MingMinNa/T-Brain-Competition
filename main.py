import pandas as pd
import numpy as np
import os

from utils import related_data, const, data_preprocess
from utils.models import regression_nn



if __name__ == '__main__':
    q_date_df = pd.read_csv(os.path.join(const.SUBMISSION_FOLDER, 'q_date.csv'), encoding = 'utf-8')
    current_location, regression_model = None, None

    collection = []
    output_df = pd.DataFrame(columns = const.ans_df_columns)
    for idx in q_date_df.index:
        
        if current_location != q_date_df.loc[idx, 'location']:

            train_data = related_data.get_all_data(location = q_date_df.loc[idx, 'location'])
            train_data = data_preprocess.data_transform(train_data)
            X_train, y_train = data_preprocess.preprocess(train_data)
            current_location = q_date_df.loc[idx, 'location']

            if regression_model is not None:   del regression_model
            regression_model = regression_nn.build_model(X_train, y_train, epochs = 100)

        recent_df = related_data.get_date_data(target_date = str(q_date_df.loc[idx, 'date']), location = q_date_df.loc[idx, 'location'], day_diff = 5)
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
        X_test, _ = data_preprocess.preprocess(input_df)

        result_df = regression_nn.predict(regression_model, X_test, date = str(q_date_df.loc[idx, 'date']), location =  int(q_date_df.loc[idx, 'location']))
        output_df = pd.concat([output_df, result_df], axis = 0, ignore_index = True) if output_df.shape[0] > 0 else result_df

        output_df.to_csv(os.path.join(const.SUBMISSION_FOLDER, 'output.csv'), index=False, encoding="utf-8-sig")