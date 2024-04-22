from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split

from project_2_task_1.custom_mlp import CustomDataset


def experiment_one():
    dataset_iris = load_iris(return_X_y=True)
    dataset_wine = load_wine(return_X_y=True)
    dataset_breast_cancer = load_breast_cancer(return_X_y=True)

    dataset_iris = CustomDataset(dataset_iris)
    dataset_wine = CustomDataset(dataset_wine)
    dataset_breast_cancer = CustomDataset(dataset_breast_cancer)

    train_dataset_iris, test_dataset_iris = random_split(dataset_iris, [0.8, 0.2])
    train_dataset_wine, test_dataset_wine = random_split(dataset_wine, [0.8, 0.2])
    train_dataset_breast_cancer, test_dataset_breast_cancer = random_split(dataset_breast_cancer, [0.8, 0.2])

    print(train_dataset_iris[0])


experiment_one()
