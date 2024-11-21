import os
import torch


def __get_parent_folder(file_path, level = 1):
    current_path = file_path
    for i in range(level):
        current_path = os.path.dirname(current_path)
    return current_path

PROJECT_FOLDER = __get_parent_folder(os.path.abspath(__file__), 2)
TRAINING_FOLDER = os.path.join(PROJECT_FOLDER, 'Training_Data')
SUBMISSION_FOLDER = os.path.join(PROJECT_FOLDER, 'submission')

# Generate_folder
GENERATE_MERGE_FOLDER = os.path.join(TRAINING_FOLDER, 'Generate', '36_Merged_TrainingData')
GENERATE_AVG_FOLDER = os.path.join(TRAINING_FOLDER, 'Generate', 'Average_Data', 'TrainData(AVG)')
GENERATE_IncompleteAVG_FOLDER = os.path.join(TRAINING_FOLDER, 'Generate', 'Average_Data', 'TrainData(IncompleteAVG)')

# Given_folder
GIVEN_RAW_DATA_FOLDER = os.path.join(TRAINING_FOLDER, 'Given', '36_TrainingData')
GIVEN_ADDITIONAL_DATA_FOLDER = os.path.join(TRAINING_FOLDER, 'Given', '36_TrainingData_Additional_V2')
GIVEN_AVG_FOLDER = os.path.join(TRAINING_FOLDER, 'Given', 'Average_Data', 'TrainData(AVG)')
GIVEN_IncompleteAVG_FOLDER = os.path.join(TRAINING_FOLDER, 'Given', 'Average_Data', 'TrainData(IncompleteAVG)')

raw_data_columns = 'LocationCode,DateTime,WindSpeed(m/s),Pressure(hpa),Temperature(°C),Humidity(%),Sunlight(Lux),Power(mW)'.split(',')
avg_data_columns = 'Serial,WindSpeed(m/s),Pressure(hpa),Temperature(°C),Humidity(%),Sunlight(Lux),Power(mW)'.split(',')
ans_df_columns = '序號,答案'.split(',')

device = 'cuda' if torch.cuda.is_available() else 'cpu'