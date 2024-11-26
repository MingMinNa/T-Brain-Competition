import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import sys
import re

from utils import const

class Time:
    def __init__(self, str_datetime):
        split_datetime = str_datetime.split(' ')
        self.date = split_datetime[0]
        time_region = split_datetime[1].split(':')[:2]
        time_region[1] = f"{(int(time_region[1]) // 10) * 10:02d}"
        self.time_region = ':'.join(time_region)

    def get_date(self):
        return self.date
    def get_time_region(self):
        return self.time_region
    def get_datetime(self):
        return f"{self.get_date()} {self.get_time_region()}"
    
    def __repr__(self):
        return self.get_datetime()
    
class DataCollector:

    def __init__(self, date):
        self.collection = dict()
        self.date = date

    def insert_data(self, time_obj, data):
        if time_obj.get_time_region() not in self.collection:
            self.collection[time_obj.get_time_region()] = pd.DataFrame(columns = 'LocationCode,DateTime,WindSpeed(m/s),Pressure(hpa),Temperature(°C),Humidity(%),Sunlight(Lux),Power(mW)'.split(','))
        next_row = self.collection[time_obj.get_time_region()].shape[0]
        for col in self.collection[time_obj.get_time_region()].columns:
            self.collection[time_obj.get_time_region()].loc[next_row, col] = data[col]
    
    def get_ave_data(self, func, location):
        if func == 'AVG':   self.collection = {time_region:data for time_region, data in self.collection.items() if data.shape[0] >= 10}
        
        if len(self.collection) == 0: return None

        ret_df = pd.DataFrame([
                [
                    f"{self.get_date().replace('-', '')}{time_region.replace(':', '')}{location:02d}",
                    np.round(np.mean(self.collection[time_region].loc[:, 'WindSpeed(m/s)']), 2),
                    np.round(np.mean(self.collection[time_region].loc[:, 'Pressure(hpa)']), 2),
                    np.round(np.mean(self.collection[time_region].loc[:, 'Temperature(°C)']), 2),
                    np.round(np.mean(self.collection[time_region].loc[:, 'Humidity(%)']), 2),
                    np.round(np.mean(self.collection[time_region].loc[:, 'Sunlight(Lux)']), 2),
                    np.round(np.mean(self.collection[time_region].loc[:, 'Power(mW)']), 2),
                ]
                for time_region in self.collection
            ], columns = 'Serial,WindSpeed(m/s),Pressure(hpa),Temperature(°C),Humidity(%),Sunlight(Lux),Power(mW)'.split(','))
        return ret_df
    
    def get_date(self):
        return self.date

def merge_data(dest_folder, raw_data_folder, additional_data_folder):
    '''
    Merge files in raw_data_folder and additional_data_folder
    and generate new files in dest_folder
    '''
    raw_filenames = os.listdir(raw_data_folder)
    additional_filenames = os.listdir(additional_data_folder)
    for raw_data_filename in tqdm(raw_filenames):
        if raw_data_filename.replace('.', '_2.') not in additional_filenames:
            df = pd.read_csv(os.path.join(raw_data_folder, raw_data_filename), encoding = 'utf-8')
            df.to_csv(os.path.join(dest_folder, raw_data_filename), index = False)
        else:
            raw_df = pd.read_csv(os.path.join(raw_data_folder, raw_data_filename), encoding = 'utf-8')
            additional_df = pd.read_csv(os.path.join(additional_data_folder, raw_data_filename.replace('.', '_2.')), encoding = 'utf-8')
            result_df = pd.concat([raw_df, additional_df], axis = 0, ignore_index = True)
            result_df.drop_duplicates(subset = ['DateTime'], inplace = True)
            result_df.sort_values(by = ['DateTime'], ascending = True, inplace = True)
            result_df.to_csv(os.path.join(dest_folder, raw_data_filename), index = False)
    return

def make_AVG_TrainData(dest_folder, src_folder):

    for i in tqdm(range(1, len(os.listdir(src_folder)) + 1)):
        src_filename = f"L{i}_Train.csv"

        read_df = pd.read_csv(os.path.join(src_folder, src_filename))
        write_df = pd.DataFrame(columns = const.avg_data_columns)
        collector = None
        for idx in read_df.index:
            data = read_df.loc[idx, :]
            time_obj = Time(data.loc['DateTime'])
            if collector is None:   collector = DataCollector(time_obj.get_date())
            if collector.get_date() == time_obj.get_date():
                collector.insert_data(time_obj, data)
            else:
                ret_df = collector.get_ave_data('AVG', i)
                if isinstance(ret_df, pd.DataFrame):  write_df = pd.concat([write_df, ret_df], ignore_index = True) if write_df.shape[0] > 0 else ret_df
                del collector
                collector = DataCollector(time_obj.get_date())

        if collector is not None:
            ret_df = collector.get_ave_data('AVG', i)
            if isinstance(ret_df, pd.DataFrame):  write_df = pd.concat([write_df, ret_df], ignore_index = True) if write_df.shape[0] > 0 else ret_df
            del collector

        write_df.to_csv(os.path.join(dest_folder, f"AvgDATA_{i:02d}.csv"), index = False)

    return

def make_IncompleteAVG_TrainData(dest_folder, src_folder):

    for i in tqdm(range(1, len(os.listdir(src_folder)) + 1)):
        src_filename = f"L{i}_Train.csv"

        read_df = pd.read_csv(os.path.join(src_folder, src_filename))
        write_df = pd.DataFrame(columns = const.avg_data_columns)
        collector = None
        for idx in read_df.index:
            data = read_df.loc[idx, :]
            time_obj = Time(data.loc['DateTime'])
            if collector is None:   collector = DataCollector(time_obj.get_date())
            if collector.get_date() == time_obj.get_date():
                collector.insert_data(time_obj, data)
            else:
                ret_df = collector.get_ave_data('AVG', i)
                if isinstance(ret_df, pd.DataFrame):  write_df = pd.concat([write_df, ret_df], ignore_index = True) if write_df.shape[0] > 0 else ret_df
                del collector
                collector = DataCollector(time_obj.get_date())

        if collector is not None:
            ret_df = collector.get_ave_data('AVG', i)
            if isinstance(ret_df, pd.DataFrame):  write_df = pd.concat([write_df, ret_df], ignore_index = True) if write_df.shape[0] > 0 else ret_df
            del collector

        write_df.to_csv(os.path.join(dest_folder, f"IncompleteAvgDATA_{i:02d}.csv"), index = False)

    return

def merge_additional_features(src_folder, features_folder):
    
    for file_name in tqdm(os.listdir(src_folder)):
        location = int(re.search(r'(\d+)', file_name).group())
        target_df = pd.read_csv(os.path.join(src_folder, file_name))
        features_df = pd.read_csv(os.path.join(features_folder, f'AdditionalTrainData_{location:02d}.csv'))
        features_df['Serial'] = features_df['Serial'].astype(str)
        features_df.set_index('Serial', inplace = True)

        for idx in target_df.index:
            for col in features_df.columns:
                try:
                    target_df.loc[idx, col] = features_df.loc[str(target_df.loc[idx, 'Serial']), col]
                except:
                    target_df.loc[idx, col] = None
        target_df.to_csv(os.path.join(src_folder, file_name), index = False)
    return 

def sort_q_date(q_date_path):

    q_date = pd.read_csv(q_date_path)
    q_date.sort_values(by = ['location', 'date'], inplace = True)
    q_date.to_csv(q_date_path, index = False)

    return

def init_argparser():
    parser = argparse.ArgumentParser(description = "此程式用於處理給定資料，並將資料以 10 分鐘為單位記錄數值平均")

    parser.add_argument('-M', '--Merge', action="store_true", help = "將原始資料和額外資料做合併")
    parser.add_argument('-A', '--AVG', action="store_true", help = "從 Merge_TrainingData 中，生成 TrainData(AVG)")
    parser.add_argument('-I', '--IncompleteAVG', action="store_true", help="從 Merge_TrainingData 中，生成 TrainData(IncompleteAVG)")
    parser.add_argument('-S', '--Sort', action="store_true", help="執行 q_date.csv 的排序")
    parser.add_argument('-F', '--Features', action="store_true", help="將額外特徵整合進 TrainData(AVG) 和 TrainData(IncompleteAVG)")

    return parser.parse_args()

if __name__ == '__main__':

    args = init_argparser()

    if args.Merge == True:
        sys.stdout.write('------------ Merge Data ------------\n')
        merge_data(dest_folder = const.GENERATE_MERGE_FOLDER, 
                   raw_data_folder = const.GIVEN_RAW_DATA_FOLDER, 
                   additional_data_folder = const.GIVEN_ADDITIONAL_DATA_FOLDER)
    
    if args.AVG == True:
        sys.stdout.write('---------------- AVG ---------------\n')
        make_AVG_TrainData(dest_folder = const.GENERATE_AVG_FOLDER, 
                           src_folder = const.GENERATE_MERGE_FOLDER)


    if args.IncompleteAVG == True:
        sys.stdout.write('---------- Incomplete AVG ----------\n')
        make_IncompleteAVG_TrainData(dest_folder = const.GENERATE_IncompleteAVG_FOLDER, 
                                     src_folder = const.GENERATE_MERGE_FOLDER)

    if args.Sort == True:
        sys.stdout.write('---------- Sort q_date ----------\n')
        sort_q_date(q_date_path = os.path.join(const.SUBMISSION_FOLDER, 'q_date.csv'))
    
    if args.Features == True:
        sys.stdout.write('--------- Merge Features ---------\n')
        merge_additional_features(src_folder = os.path.join(const.GENERATE_AVG_FOLDER), 
                                  features_folder = os.path.join(const.GENERATE_FOLDER, 'Additional_Features'))
        merge_additional_features(src_folder = os.path.join(const.GENERATE_IncompleteAVG_FOLDER), 
                                  features_folder = os.path.join(const.GENERATE_FOLDER, 'Additional_Features'))
    
    quit()