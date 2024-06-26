import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt, cm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class TwoFeaturesMNISTArturCNN(nn.Module):
    def __init__(self):
        super(TwoFeaturesMNISTArturCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)  # 8x28x28
        self.pool1 = nn.MaxPool2d(2, 2)  # 8x12x12
        self.conv2 = nn.Conv2d(8, 4, 5)  # 4x10x10
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x4x4
        self.fc1 = nn.Linear(4 * 4 * 4, 8)
        self.fc2 = nn.Linear(8, 2)
        self.fc3 = nn.Linear(2, 10)

    def forward(self, x, x_two_features=False):
        if not x_two_features:
            x = nn.functional.relu(self.conv1(x))
            x = self.pool1(x)
            x = nn.functional.relu(self.conv2(x))
            x = self.pool2(x)
            x = x.view(-1, 4 * 4 * 4)  # flatten
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            two_features = x
            x = self.fc3(x)
            return x, two_features
        else:
            x = self.fc3(x)
            return x, x


class TwoFeaturesCIFAR10ArturCNN(nn.Module):
    def __init__(self):
        super(TwoFeaturesCIFAR10ArturCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 9)  # 8x24x24
        self.pool1 = nn.MaxPool2d(2, 2)  # 8x12x12
        self.conv2 = nn.Conv2d(8, 4, 5)  # 4x10x10
        self.pool2 = nn.MaxPool2d(2, 2)  # 4x5x5
        self.fc1 = nn.Linear(4 * 4 * 4, 8)
        self.fc2 = nn.Linear(8, 2)
        self.fc3 = nn.Linear(2, 10)

    def forward(self, x, x_two_features=False):
        if not x_two_features:
            x = nn.functional.relu(self.conv1(x))
            x = self.pool1(x)
            x = nn.functional.relu(self.conv2(x))
            x = self.pool2(x)
            x = x.view(-1, 4 * 4 * 4)  # flatten
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            two_features = x
            x = self.fc3(x)
            return x, two_features
        else:
            x = self.fc3(x)
            return x, x


class ArturMNISTCNN(nn.Module):
    def __init__(self):
        super(ArturMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ArturCIFAR10CNN(nn.Module):
    def __init__(self):
        super(ArturCIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def plot_confusion_matrix_from_dataloader(model, dataloader, classes, is_two_features):
    y_true = np.array([])
    y_pred = np.array([])

    for inputs, labels in dataloader:
        if is_two_features:
           y_pred = np.append(y_pred, np.argmax(model(inputs)[0].detach().numpy(), axis=1))
        else:
            y_pred = np.append(y_pred, np.argmax(model(inputs).detach().numpy(), axis=1))
        y_true = np.append(y_true, labels)

    conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
    display.plot()
    plt.show()


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

    if is_two_features:
        for batch in test_dataloader:
            inputs, labels = batch
            outputs, two_features = model(inputs)
            for i in range(10):
                plt.scatter(-999, -999, alpha=1, label=str(i), cmap=cm.tab10)
            plt.scatter(two_features.detach().numpy()[:, 0], two_features.detach().numpy()[:, 1], c=np.argmax(outputs.detach().numpy(), axis=1)[:], alpha=1, cmap=cm.tab10, s=50, marker="o")
            plt.xlim([0, 20])
            plt.ylim([0, 20])
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.legend()
            plt.show()


def analyze_two_features_mnist_cnn():
    train_mnist = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST('../data', train=False, transform=transform)
    train_mnist_dataloader = DataLoader(dataset=train_mnist, batch_size=8192, shuffle=False)
    test_mnist_dataloader = DataLoader(dataset=test_mnist, batch_size=8192, shuffle=False)

    # Create and train model
    model = TwoFeaturesMNISTArturCNN()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_model(model, loss_function, optimizer, 50, train_mnist_dataloader, test_mnist_dataloader, True)

    # Plot decision boundary
    x_min, x_max = 0, 100
    y_min, y_max = 0, 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

    # Create predictions for each element of grid in decision boundary
    with torch.no_grad():
        predictions = np.argmax(model(grid_tensor, True)[0].detach().numpy(), axis=1)
        unique, counts = np.unique(predictions, return_counts=True)

    # Add dummy points to get labels on the final plot
    for i in range(10):
        plt.scatter(-999, -999, alpha=1, label=str(i), cmap=cm.tab10)

    plt.scatter(xx, yy, c=predictions, alpha=1, cmap=cm.tab10, s=50, marker="s", edgecolor='none')
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

    plot_confusion_matrix_from_dataloader(model, test_mnist_dataloader, np.arange(10), True)

    return model


def analyze_two_features_cifar10_cnn():
    cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    train_mnist = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    test_mnist = datasets.CIFAR10('../data', train=False, transform=transform)
    train_mnist_dataloader = DataLoader(dataset=train_mnist, batch_size=4096, shuffle=False)
    test_mnist_dataloader = DataLoader(dataset=test_mnist, batch_size=4096, shuffle=False)

    # Create and train model
    model = TwoFeaturesCIFAR10ArturCNN()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_model(model, loss_function, optimizer, 50, train_mnist_dataloader, test_mnist_dataloader, True)

    # Plot decision boundary
    x_min, x_max = 0, 20
    y_min, y_max = 0, 20
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

    # Create predictions for each element of grid in decision boundary
    with torch.no_grad():
        predictions = np.argmax(model(grid_tensor, True)[0].detach().numpy(), axis=1)
        unique, counts = np.unique(predictions, return_counts=True)

    # Add dummy points to get labels on the final plot
    for c in cifar10_classes:
        plt.scatter(-999, -999, alpha=1, label=c, cmap=cm.tab10)

    plt.scatter(xx, yy, c=predictions, alpha=1, cmap=cm.tab10, s=50, marker="s", edgecolor='none')
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

    plot_confusion_matrix_from_dataloader(model, test_mnist_dataloader, cifar10_classes, True)

    return model


def analyze_n_features_mnist_cnn():
    train_mnist = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST('../data', train=False, transform=transform)
    train_mnist_dataloader = DataLoader(dataset=train_mnist, batch_size=8192, shuffle=False)
    test_mnist_dataloader = DataLoader(dataset=test_mnist, batch_size=8192, shuffle=False)

    # Create and train model
    model = ArturMNISTCNN()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, loss_function, optimizer, 50, train_mnist_dataloader, test_mnist_dataloader, False)

    plot_confusion_matrix_from_dataloader(model, test_mnist_dataloader, np.arange(10), False)

    return model


def analyze_n_features_cifar10_cnn():
    cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    train_mnist = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    test_mnist = datasets.CIFAR10('../data', train=False, transform=transform)
    train_mnist_dataloader = DataLoader(dataset=train_mnist, batch_size=128, shuffle=False)
    test_mnist_dataloader = DataLoader(dataset=test_mnist, batch_size=128, shuffle=False)

    # Create and train model
    model = ArturCIFAR10CNN()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train_model(model, loss_function, optimizer, 10, train_mnist_dataloader, test_mnist_dataloader, False)

    plot_confusion_matrix_from_dataloader(model, test_mnist_dataloader, cifar10_classes, False)

    return model


def experiment_one():
    # two_features_mnist_model = analyze_two_features_mnist_cnn()
    # torch.save(two_features_mnist_model.state_dict(), "two_features_mnist_artur.pt")
    #
    # n_features_mnist_model = analyze_n_features_mnist_cnn()
    # torch.save(n_features_mnist_model.state_dict(), "n_features_mnist_artur.pt")

    # two_features_cifar10_model = analyze_two_features_cifar10_cnn()
    # torch.save(two_features_cifar10_model.state_dict(), "two_features_cifar10_artur.pt")

    n_features_cifar10_model = analyze_n_features_cifar10_cnn()
    torch.save(n_features_cifar10_model.state_dict(), "n_features_cifar10_artur.pt")


experiment_one()
