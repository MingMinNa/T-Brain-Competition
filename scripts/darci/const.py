import os

def __get_parent_folder(file_path, level = 1):
    current_path = file_path
    for i in range(level):
        current_path = os.path.dirname(current_path)
    return current_path

PROJECT_FOLDER = __get_parent_folder(os.path.abspath(__file__), 3)
TRAINING_FOLDER = os.path.join(PROJECT_FOLDER, 'Training_Data', 'darci')
SUBMISSION_FOLDER = os.path.join(PROJECT_FOLDER, 'submission', 'darci')

MODELS_FOLDER = os.path.join(PROJECT_FOLDER, 'save_models', 'darci')


def build_model_folder():
    if not os.path.exists(MODELS_FOLDER):
        os.mkdir(MODELS_FOLDER)
    if not os.path.exists(os.path.join(MODELS_FOLDER, 'darci')):
        os.mkdir(os.path.join(MODELS_FOLDER, 'darci'))