import os


def create_training_path_file(file_name: str) -> str:
    _training_path_file = os.path.join(os.getcwd(), "training", file_name + "-training.csv")
    return _training_path_file


def create_test_path_file(file_name: str) -> str:
    _test_path_file = os.path.join(os.getcwd(), "test", file_name + "-test.csv")
    return _test_path_file


def create_conf_path_file(file_name: str) -> str:
    _conf_path_file = os.path.join(os.getcwd(), "training", file_name + ".pickle")
    return _conf_path_file
