import os
import pandas as pd
import datetime
import numpy as np

try:    import const
except: from . import const


def get_month_data(target_date, location):
    '''
    example:
        (1) target_data == '20240106', location = '01'
            ret_df will contain all data(collected from 'AvgDATA_01.csv') whose year is 2024 and date is 1

        (2) target_data == '20240706', location = '02', day_diff = 8
            ret_df will contain all data(collected from 'AvgDATA_02.csv') whose year is 2024 and date is 7
    '''

    ret_df = pd.DataFrame(columns = const.avg_data_columns)
    read_df = pd.read_csv(os.path.join(const.GENERATE_AVG_FOLDER, f'AvgDATA_{int(location):02d}.csv'), encoding = 'utf-8')

    for idx in read_df.index:
        if str(read_df.loc[idx, 'Serial'])[:6] == target_date[:6]:
            ret_df.loc[ret_df.shape[0], :] = read_df.loc[idx, :]

    return ret_df

def get_date_data(target_date, location, day_diff):
    '''
    example:
        (1) target_data == '20240106', location = '01', day_diff = 5
            ret_df will contain all data(collected from 'AvgDATA_01.csv') whose date from '20240101' to '20240106' 

        (2) target_data == '20240706', location = '02', day_diff = 8
            ret_df will contain all data(collected from 'AvgDATA_02.csv') whose date from '20240628' to '20240706' 
    '''

    ret_df = pd.DataFrame(columns = const.avg_data_columns)
    read_df = pd.read_csv(os.path.join(const.GENERATE_AVG_FOLDER, f'AvgDATA_{int(location):02d}.csv'), encoding = 'utf-8')

    target_date = datetime.datetime(year = int(target_date[:4]), 
                                    month = int(target_date[4:6]), 
                                    day = int(target_date[6:8]))
    for idx in read_df.index:
        data_date = datetime.datetime(year = int(str(read_df.loc[idx, 'Serial'])[:4]), 
                                      month = int(str(read_df.loc[idx, 'Serial'])[4:6]), 
                                      day = int(str(read_df.loc[idx, 'Serial'])[6:8]))

        if target_date >= data_date and (target_date - data_date).days <= day_diff:
            ret_df.loc[ret_df.shape[0], :] = read_df.loc[idx, :]
    ret_df.loc[:, 'Serial'] = ret_df.loc[:, 'Serial'].astype(np.int64)
    return ret_df

def merge_by_time(recent_df, date):
    recent_df.loc[:, 'Time'] = recent_df.apply(lambda row: int(str(row['Serial'])[8:10])*60 + int(str(row['Serial'])[10:12]), axis=1)
    # remove 'WindSpeed(m/s)' 
    input_df = recent_df[(recent_df['Time'] >= 9 * 60) & (recent_df['Time'] < 17 * 60)]
    input_df = input_df.drop(columns = ['Serial', 'WindSpeed(m/s)', 'Power(mW)']) 
    input_df = input_df.groupby(['Time']).mean()
    
    input_df = input_df.assign(
        # Year = int(date[:4]),
        # Month = int(date[4:6]),
        Day = int(date[6:8]),
        # Hour = input_df.loc[:, 'Time'].astype(int) // 60,
        # Minute = np.mod(input_df.loc[:, 'Time'].astype(int), 60),
    )
    # input_df = input_df.drop(columns = ['Time'])
    return input_df

if __name__ == '__main__':
    pass
