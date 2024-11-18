import os


def __get_parent_folder(file_path, level = 1):
    current_path = file_path
    for i in range(level):
        current_path = os.path.dirname(current_path)
    return current_path

PROJECT_FOLDER = __get_parent_folder(os.path.abspath(__file__), 2)
TRAINING_FOLDER = os.path.join(PROJECT_FOLDER, 'Training_Data')
