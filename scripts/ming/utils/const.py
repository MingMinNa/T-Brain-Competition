import os
import torch


def __get_parent_folder(file_path, level = 1):
    current_path = file_path
    for i in range(level):
        current_path = os.path.dirname(current_path)
    return current_path

PROJECT_FOLDER = __get_parent_folder(os.path.abspath(__file__), 4)
TRAINING_FOLDER = os.path.join(PROJECT_FOLDER, 'Training_Data', 'ming')
SUBMISSION_FOLDER = os.path.join(PROJECT_FOLDER, 'submission', 'ming')

# Generate_folder
GENERATE_FOLDER = os.path.join(TRAINING_FOLDER, 'Generate')
GENERATE_MERGE_FOLDER = os.path.join(GENERATE_FOLDER, '36_Merged_TrainingData')
GENERATE_AVG_FOLDER = os.path.join(GENERATE_FOLDER, 'Average_Data', 'TrainData(AVG)')
GENERATE_IncompleteAVG_FOLDER = os.path.join(GENERATE_FOLDER, 'Average_Data', 'TrainData(IncompleteAVG)')

# Given_folder
GIVEN_FOLDER = os.path.join(TRAINING_FOLDER, 'Given')
GIVEN_RAW_DATA_FOLDER = os.path.join(GIVEN_FOLDER, '36_TrainingData')
GIVEN_ADDITIONAL_DATA_FOLDER = os.path.join(GIVEN_FOLDER, '36_TrainingData_Additional_V2')
GIVEN_AVG_FOLDER = os.path.join(GIVEN_FOLDER, 'Average_Data', 'TrainData(AVG)')
GIVEN_IncompleteAVG_FOLDER = os.path.join(GIVEN_FOLDER, 'Average_Data', 'TrainData(IncompleteAVG)')

ans_df_columns = '序號,答案'.split(',')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
