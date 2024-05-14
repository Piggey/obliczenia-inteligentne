import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np

# Przygotowanie danych
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

LEARN_RATE = 0.001
NUM_EPOCHS = 10

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=TRANSFORM)
TRAIN_LOADER = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=TRANSFORM)
TEST_LOADER = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

class TwoFeaturesCNNDawid(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 2)  # Warstwa kończąca się 2-elementowym wektorem cech
        self.fc3 = nn.Linear(2, 10)    # Warstwa Linear dla klasyfikacji cyfr MNIST

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))  # Uwaga: Nie stosujemy funkcji aktywacji na warstwie końcowej dla cech
        x = self.fc3(x)
        return x

    def forward_conv(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))  # Uwaga: Nie stosujemy funkcji aktywacji na warstwie końcowej dla cech
        return x


def train(model, criterion, optimizer):
    train_accuracy = []
    test_accuracy = []
    best_accuracy = 0.0
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in TRAIN_LOADER:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_accuracy = correct_train / total_train
        train_accuracy.append(epoch_train_accuracy)

        # Test modelu
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in TEST_LOADER:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

            epoch_test_accuracy = correct_test / total_test
            test_accuracy.append(epoch_test_accuracy)

            if epoch_test_accuracy > best_accuracy:
                best_accuracy = epoch_test_accuracy
                best_model = model.state_dict()

        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {running_loss / len(TRAIN_LOADER)}, '
              f'Train Accuracy: {epoch_train_accuracy}, Test Accuracy: {epoch_test_accuracy}')

    print(f'Best Test Accuracy: {best_accuracy}')
    return train_accuracy, test_accuracy, best_model

def visualize_decision_boundary(model, loader, outfile):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, target in loader:
            output = model.forward_conv(inputs)
            features.extend(output.squeeze().numpy())
            labels.extend(target.numpy())

    features = np.array(features)
    labels = np.array(labels)

    plt.figure(figsize=(10, 6))
    for i in range(10):
        plt.scatter(features[labels == i, 0], features[labels == i, 1], label=str(i))

    plt.title('Decision Boundary Visualization')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig(outfile)
    plt.close()

def analyze_2_components():
    model = TwoFeaturesCNNDawid()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    train_accuracy, test_accuracy, best_model = train(model, criterion, optimizer)

    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(test_accuracy, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('accuracy scores; MNIST dataset; n_components=2')
    plt.savefig('acc_scores_mnist_n_components=2.png')
    plt.close()

    # Testowanie najlepszego modelu
    model.load_state_dict(best_model)
    model.eval()

    # Predykcja na zbiorze testowym
    pred_labels = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in TEST_LOADER:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            pred_labels.extend(predicted.numpy())
            true_labels.extend(labels.numpy())

    # Macierz pomyłek
    cm = confusion_matrix(true_labels, pred_labels)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(true_labels))
    display.plot()
    plt.savefig('confusion_matrix_mnist_n_components=2.png')
    plt.close()

    visualize_decision_boundary(model, TEST_LOADER, 'decision_boundary_mnist_n_components=2.png')

# [[1, 0.45211666666666667, 0.4587], [2, 0.5237, 0.5246], [3, 0.6100333333333333, 0.6084], [4, 0.68645, 0.6786], [5, 0.7465666666666667, 0.7405]]

if __name__ == '__main__':
    analyze_2_components()