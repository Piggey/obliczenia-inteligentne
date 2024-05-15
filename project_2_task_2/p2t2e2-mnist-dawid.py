import torch
import torch.nn as nn
import torch.utils as utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, criterion, optimizer):
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(TRAIN_LOADER):
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 100 == 99:    
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

def test_accuracy(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

LEARN_RATE = 0.001
NUM_EPOCHS = 5

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=TRANSFORM)

K = 1000
subsample_train_indices = torch.randperm(len(trainset))[:K]

TRAIN_LOADER = data.DataLoader(trainset, batch_size=32, num_workers=2, sampler=data.SubsetRandomSampler(subsample_train_indices))

testset = datasets.MNIST(root='./data', train=False, download=True, transform=TRANSFORM)
TEST_LOADER = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

if __name__ == '__main__':
    accuracies = []
    best_mean_accuracy = 0

    for i in range(10):
        model = CNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

        train(model, criterion, optimizer)
        accuracy = test_accuracy(model, TEST_LOADER)
        accuracies.append(accuracy)
        print(f'pass {i}; {accuracy=}')

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    if mean_accuracy > best_mean_accuracy:
        best_mean_accuracy = mean_accuracy
        torch.save(model.state_dict(), 'mnist-dawid.model')

    print(f'{mean_accuracy=}; {std_accuracy=}')

## BRAK AUGMENTACJI
# wszystkie dane
# mean_accuracy=0.98993; std_accuracy=0.0017262966141425573

# 100
# mean_accuracy=0.39083999999999997; std_accuracy=0.057972565925616934

# 200
# mean_accuracy=0.69212; std_accuracy=0.03140601216327855

# 1000
# mean_accuracy=0.94406; std_accuracy=0.008767690687974788 