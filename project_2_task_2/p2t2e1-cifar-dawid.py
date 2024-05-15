import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np

# Przygotowanie danych
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

TRAIN_SET = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=TRANSFORM)
TRAIN_LOADER = torch.utils.data.DataLoader(TRAIN_SET, batch_size=4,
                                          shuffle=True, num_workers=2)

TEST_SET = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=TRANSFORM)
TEST_LOADER = torch.utils.data.DataLoader(TEST_SET, batch_size=4,
                                         shuffle=False, num_workers=2)

NUM_EPOCHS = 10

# Definicja modelu
class CNNCifarDawid(nn.Module):
    def __init__(self):
        super(CNNCifarDawid, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.fc4 = nn.Linear(2, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def forward_conv(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = CNNCifarDawid()

# Definicja funkcji straty i optymizatora
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer):
    train_accuracy = []
    test_accuracy = []
    best_accuracy = 0.0

    # Trenowanie sieci
    for epoch in range(NUM_EPOCHS):  # iteracja przez zestawy danych wielokrotnie
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, (inputs, labels) in enumerate(TRAIN_LOADER):
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # wydruk statystyk
            if i % 2000 == 1999:    # wydruk co 2000 mini-pakietów danych
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        epoch_train_accuracy = correct_train / total_train
        train_accuracy.append(epoch_train_accuracy)


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

        print(f'{epoch=}; {epoch_train_accuracy=}; {epoch_test_accuracy=}')

    print('Trenowanie zakończone')
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

train_accuracy, test_accuracy, best_model = train(model, criterion, optimizer)

plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(test_accuracy, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('accuracy scores; CIFAR10 dataset')
plt.savefig('acc_scores_cifar10_2d.png')
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
plt.savefig('confusion_matrix_cifar10_2d.png')
plt.close()

visualize_decision_boundary(model, TEST_LOADER, 'decision_boundary_cifar_2d.png')