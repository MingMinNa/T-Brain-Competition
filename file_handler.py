import os
import pandas as pd
import numpy as np
from tqdm import tqdm

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

def merge_data(DEST_FOLDER, RAW_DATA_FOLDER, ADDITIONAL_DATA_FOLDER):
    '''
    Merge files in RAW_DATA_FOLDER and ADDITIONAL_DATA_FOLDER
    and generate new files in DEST_FOLDER
    '''
    raw_filenames = os.listdir(RAW_DATA_FOLDER)
    additional_filenames = os.listdir(ADDITIONAL_DATA_FOLDER)
    for raw_data_filename in raw_filenames:
        if raw_data_filename.replace('.', '_2.') not in additional_filenames:
            df = pd.read_csv(os.path.join(RAW_DATA_FOLDER, raw_data_filename), encoding = 'utf-8')
            df.to_csv(os.path.join(DEST_FOLDER, raw_data_filename), index = False)
        else:
            raw_df = pd.read_csv(os.path.join(RAW_DATA_FOLDER, raw_data_filename), encoding = 'utf-8')
            additional_df = pd.read_csv(os.path.join(ADDITIONAL_DATA_FOLDER, raw_data_filename.replace('.', '_2.')), encoding = 'utf-8')
            result_df = pd.concat([raw_df, additional_df], axis = 0, ignore_index = True)
            result_df.drop_duplicates(subset = ['DateTime'], inplace = True)
            result_df.sort_values(by = ['DateTime'], ascending = True, inplace = True)
            result_df.to_csv(os.path.join(DEST_FOLDER, raw_data_filename), index = False)
    return


def make_AVG_TrainData(DEST_FOLDER, SRC_FOLDER):

    for i in tqdm(range(1, len(os.listdir(SRC_FOLDER)) + 1)):
        src_filename = f"L{i}_Train.csv"

        read_df = pd.read_csv(os.path.join(SRC_FOLDER, src_filename))
        write_df = pd.DataFrame(columns = 'Serial,WindSpeed(m/s),Pressure(hpa),Temperature(°C),Humidity(%),Sunlight(Lux),Power(mW)'.split(','))
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

        write_df.to_csv(os.path.join(DEST_FOLDER, f"AvgDATA_{i:02d}.csv"), index = False)

    return

def make_IncompleteAVG_TrainData(DEST_FOLDER, SRC_FOLDER):

    for i in tqdm(range(1, len(os.listdir(SRC_FOLDER)) + 1)):
        src_filename = f"L{i}_Train.csv"

        read_df = pd.read_csv(os.path.join(SRC_FOLDER, src_filename))
        write_df = pd.DataFrame(columns = 'Serial,WindSpeed(m/s),Pressure(hpa),Temperature(°C),Humidity(%),Sunlight(Lux),Power(mW)'.split(','))
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

        write_df.to_csv(os.path.join(DEST_FOLDER, f"IncompleteAvgDATA_{i:02d}.csv"), index = False)

    return

if __name__ == '__main__':

    DEST_FOLDER = os.path.join(const.TRAINING_FOLDER, 'Generate', '36_Merged_TrainingData')
    RAW_DATA_FOLDER = os.path.join(const.TRAINING_FOLDER, 'Given', '36_TrainingData')
    ADDITIONAL_DATA_FOLDER = os.path.join(const.TRAINING_FOLDER, 'Given', '36_TrainingData_Additional_V2')
    
    # merge_data(DEST_FOLDER, RAW_DATA_FOLDER, ADDITIONAL_DATA_FOLDER)

    DEST_FOLDER = os.path.join(const.TRAINING_FOLDER, 'Generate', 'Average_Data','TrainData(AVG)')
    SRC_FOLDER = os.path.join(const.TRAINING_FOLDER, 'Generate', '36_Merged_TrainingData')

    # make_AVG_TrainData(DEST_FOLDER, SRC_FOLDER)

    DEST_FOLDER = os.path.join(const.TRAINING_FOLDER, 'Generate', 'Average_Data','TrainData(IncompleteAVG)')
    SRC_FOLDER = os.path.join(const.TRAINING_FOLDER, 'Generate', '36_Merged_TrainingData')

    make_IncompleteAVG_TrainData(DEST_FOLDER, SRC_FOLDER)

    quit()