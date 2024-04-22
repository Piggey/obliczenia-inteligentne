from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from torchvision import datasets, transforms
from project_2_task_1.custom_mlp import CustomDataset


def experiment_one():
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset_iris = load_iris(return_X_y=True)
    dataset_wine = load_wine(return_X_y=True)
    dataset_breast_cancer = load_breast_cancer(return_X_y=True)

    dataset_iris = CustomDataset(dataset_iris)
    dataset_wine = CustomDataset(dataset_wine)
    dataset_breast_cancer = CustomDataset(dataset_breast_cancer)

    train_dataset_iris, test_dataset_iris = random_split(dataset_iris, [0.8, 0.2])
    train_dataset_wine, test_dataset_wine = random_split(dataset_wine, [0.8, 0.2])
    train_dataset_breast_cancer, test_dataset_breast_cancer = random_split(dataset_breast_cancer, [0.8, 0.2])
    train_mnist = datasets.MNIST('../data', train=True, download=True, transform=mnist_transform)
    test_mnist = datasets.MNIST('../data', train=False, transform=mnist_transform)

    # TODO: MNIST ekstrakcja - spłaszczenia do wektora 784 elementów
    # TODO: MNIST 2x ekstrakcja - spłaszczenia do wektora 2 elementów (cech) (po jednym sposobie na osobe)
    # TODO: MNIST 2x ekstrakcja - spłaszczenia do wektora z małą liczbą elementów (cech) (po jednym sposobie na osobe)


experiment_one()
