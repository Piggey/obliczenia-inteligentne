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
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
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

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=TRANSFORM)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=TRANSFORM) 

TRAIN_LOADER = data.DataLoader(trainset, batch_size=32, num_workers=2)
TEST_LOADER = data.DataLoader(testset, batch_size=32, num_workers=2)

K = 1000
subsample_train_indices = torch.randperm(len(trainset))[:K]
TRAIN_LOADER = data.DataLoader(trainset, batch_size=32, num_workers=2, sampler=data.SubsetRandomSampler(subsample_train_indices))

if __name__ == '__main__':
    accuracies = []
    best_mean_accuracy = 0

    for i in range(5):
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
        torch.save(model.state_dict(), 'cifar-dawid.model')

    print(f'{mean_accuracy=}; {std_accuracy=}')

## BRAK AUGMENTACJI
# wszystkie dane
# mean_accuracy=0.61288; std_accuracy=0.007779562969730392

# 100
# mean_accuracy=0.14438; std_accuracy=0.030731573340784225

# 200
# mean_accuracy=0.18646000000000001; std_accuracy=0.011798067638388931

# 1000
# mean_accuracy=0.3006; std_accuracy=0.013007074997861735