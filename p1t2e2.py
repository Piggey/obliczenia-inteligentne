from sklearn.model_selection import train_test_split
from utils import load_data


def analyze_knn():
    pass


def analyze_svm():
    pass


def analyze_mlp():
    pass


def experiment_two():
    # load datasets
    dataset_2_2 = load_data("./data/2_2.csv")
    dataset_2_3 = load_data("./data/2_3.csv")

    # split 0.8 train / 0.2 test
    # [0] -> X_train; [1] -> X_test; [2] -> y_train; [3] -> y_test;
    dataset_2_2 = train_test_split(dataset_2_2[0], dataset_2_2[0], test_size=0.2, random_state=42)
    dataset_2_3 = train_test_split(dataset_2_3[0], dataset_2_3[0], test_size=0.2, random_state=42)


experiment_two()
