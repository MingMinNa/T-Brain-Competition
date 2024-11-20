import pandas as pd
import numpy as np
import os

from utils import related_data, const, data_preprocess
from utils.models import regression_nn



if __name__ == '__main__':
    q_date_df = pd.read_csv(os.path.join(const.SUBMISSION_FOLDER, 'q_date.csv'), encoding = 'utf-8')
    q_test_df = pd.read_csv(os.path.join(const.SUBMISSION_FOLDER, 'q_test.csv'), encoding = 'utf-8')

    current_date, current_location, model = None, None, None

    collection = []
    ans_df = pd.DataFrame(columns = const.ans_df_columns)
    for idx in q_date_df.index:
        
        if current_location != q_date_df.loc[idx, 'location'] or \
           current_date is None or current_date != str(q_date_df.loc[idx, 'date'])[:-2]:
            train_data = related_data.get_month_data(target_date = str(q_date_df.loc[idx, 'date'])[:-2], location = str(q_date_df.loc[idx, 'location']))
            train_data = data_preprocess.data_transform(train_data)
            X_train, y_train = data_preprocess.preprocess(train_data)
            current_date = str(q_date_df.loc[idx, 'date'])[:-2]
            current_location = q_date_df.loc[idx, 'location']
            if model is not None:   del model
            model = regression_nn.build_model(X_train, y_train)


        recent_df = related_data.get_date_data(target_date = str(q_date_df.loc[idx, 'date']), location = q_date_df.loc[idx, 'location'], day_diff = 5)
        input_df = related_data.merge_by_time(recent_df, date = str(q_date_df.loc[idx, 'date']))\
        
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
        result_df = regression_nn.predict(model, X_test, date = str(q_date_df.loc[idx, 'date']), location =  int(q_date_df.loc[idx, 'location']))
        ans_df = pd.concat([ans_df, result_df], axis = 0, ignore_index = True) if ans_df.shape[0] > 0 else result_df

        ans_df.to_csv(os.path.join(const.SUBMISSION_FOLDER, 'output.csv'), index=False, encoding="utf-8-sig")