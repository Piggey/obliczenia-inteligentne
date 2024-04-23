import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from project_2_task_1.custom_mlp import CustomDataset, CustomMLP, model_accuracy
import torch
import torch.optim as optim
from torch import nn
from sklearn.decomposition import PCA


flatten_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: torch.flatten(x)),
])


def train_model(model, loss_function, optimizer, epochs, train_dataloader, test_dataloader):
    for epoch in range(epochs):
        model.train()
        loss = -1

        for batch in train_dataloader:
            inputs, labels = batch
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"epoch {epoch + 1} finished; accuracy {model_accuracy(model, test_dataloader)}; loss {loss}")


def dataloader_to_X_y(dataloader):
    X = []
    y = []

    for Xs, ys in dataloader:
        X.append(Xs)
        y.append(ys)

    return X, y


def pca_2_components_analysis():
    train_mnist = datasets.MNIST('../data', train=True, download=True, transform=flatten_transform)
    test_mnist = datasets.MNIST('../data', train=False, transform=flatten_transform)
    train_mnist_dataloader = DataLoader(dataset=train_mnist, batch_size=128, shuffle=False)
    test_mnist_dataloader = DataLoader(dataset=test_mnist, batch_size=128, shuffle=False)

    # Split train dataset into images and labels, create Principal Component Analysis object, fit and use it
    train_images, train_labels = next(iter(train_mnist_dataloader))
    pca = PCA(n_components=2)
    pca.fit(train_images.numpy())
    train_images_2d = torch.tensor(pca.transform(train_images.numpy()))
    train_mnist = CustomDataset((train_images_2d, train_labels))
    train_mnist_dataloader = DataLoader(dataset=train_mnist, batch_size=128, shuffle=False)

    # Split test dataset into images and labels, apply Principal Component Analysis trained earlier
    test_images, test_labels = next(iter(test_mnist_dataloader))
    test_images_2d = torch.tensor(pca.transform(test_images.numpy()))
    test_mnist = CustomDataset((test_images_2d, test_labels))
    test_mnist_dataloader = DataLoader(dataset=test_mnist, batch_size=128, shuffle=False)

    # Create and train model
    model = CustomMLP(2, 16, 10)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, loss_function, optimizer, 500, train_mnist_dataloader, test_mnist_dataloader)


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

    # MNIST ekstrakcja - spłaszczenia do wektora 784 elementów


    # TODO: MNIST 2x ekstrakcja - spłaszczenia do wektora 2 elementów (cech) (po jednym sposobie na osobe)
    # Artur
    pca_2_components_analysis()

    # TODO: MNIST 2x ekstrakcja - spłaszczenia do wektora z małą liczbą elementów (cech) (po jednym sposobie na osobe)



experiment_one()
