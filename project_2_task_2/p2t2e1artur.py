import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt, cm
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class TwoFeaturesArturCNN(nn.Module):
    def __init__(self):
        super(TwoFeaturesArturCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)  # 32x28x28
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x12x12
        self.conv2 = nn.Conv2d(32, 64, 5)  # 64x10x10
        self.pool2 = nn.MaxPool2d(2, 2)  # 64x4x4
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.fc3 = nn.Linear(2, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 64 * 4 * 4)  # flatten
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        two_features = x
        x = self.fc3(x)
        return x, two_features


class ArturCNN(nn.Module):
    def __init__(self):
        super(ArturCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)  # 32x28x28
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x12x12
        self.conv2 = nn.Conv2d(32, 64, 5)  # 64x10x10
        self.pool2 = nn.MaxPool2d(2, 2)  # 64x4x4
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 64 * 4 * 4)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def model_accuracy(model, dataloader, is_two_features):
    model.eval()

    total_correct = 0
    total_instances = 0

    #  iterating through batches
    with torch.no_grad():
        for inputs, labels in dataloader:
            if is_two_features:
                predictions, two_features = model(inputs)
            else:
                predictions = model(inputs)

            classifications = torch.argmax(predictions, dim=1)
            correct_predictions = sum(classifications == labels).item()
            total_correct += correct_predictions
            total_instances += len(inputs)

    return round(total_correct / total_instances, 3)


def train_model(model, loss_function, optimizer, epochs, train_dataloader, test_dataloader, is_two_features):
    for epoch in range(epochs):
        model.train()
        loss = -1

        for batch in train_dataloader:
            inputs, labels = batch
            if is_two_features:
                outputs, two_features = model(inputs)
            else:
                outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"epoch {epoch + 1} finished; accuracy {model_accuracy(model, test_dataloader, is_two_features)}; loss {loss}")


def analyze_two_features_cnn():
    train_mnist = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST('../data', train=False, transform=transform)
    train_mnist_dataloader = DataLoader(dataset=train_mnist, batch_size=8192, shuffle=False)
    test_mnist_dataloader = DataLoader(dataset=test_mnist, batch_size=8192, shuffle=False)

    # Create and train model
    model = TwoFeaturesArturCNN()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_model(model, loss_function, optimizer, 0, train_mnist_dataloader, test_mnist_dataloader, True)

    # TODO: Plot decision boundary


def analyze_n_features_cnn():
    train_mnist = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST('../data', train=False, transform=transform)
    train_mnist_dataloader = DataLoader(dataset=train_mnist, batch_size=8192, shuffle=False)
    test_mnist_dataloader = DataLoader(dataset=test_mnist, batch_size=8192, shuffle=False)

    # Create and train model
    model = ArturCNN()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_model(model, loss_function, optimizer, 20, train_mnist_dataloader, test_mnist_dataloader, False)

    # TODO: Create multiple models and find the best one


def experiment_one():
    analyze_two_features_cnn()
    # analyze_n_features_cnn()


experiment_one()
