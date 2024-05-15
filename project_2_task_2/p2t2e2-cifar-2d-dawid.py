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

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

LEARN_RATE = 0.001
NUM_EPOCHS = 5

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=TRANSFORM)

K = 100
subsample_train_indices = torch.randperm(len(trainset))[:K]
train_loader_original_subset = data.DataLoader(trainset, batch_size=32, sampler=data.SubsetRandomSampler(subsample_train_indices))

transform_augmented = transforms.Compose([
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset_augmented = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_augmented)

K = 1000
subsample_train_indices = torch.randperm(len(trainset))[:K]
train_loader_augmented_subset = data.DataLoader(trainset_augmented, batch_size=32, sampler=data.SubsetRandomSampler(subsample_train_indices))

TRAIN_LOADER = data.DataLoader(trainset, batch_size=32, num_workers=2, sampler=data.SubsetRandomSampler(subsample_train_indices))

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=TRANSFORM)
TEST_LOADER = data.DataLoader(testset, batch_size=32, shuffle=True)

if __name__ == '__main__':
    accuracies = []
    best_mean_accuracy = 0

    for i in range(1):
        model = CNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

        train(model, criterion, optimizer)
        accuracy = test_accuracy(model, TEST_LOADER)
        accuracies.append(accuracy)
        print(f'pass {i}; {accuracy=}')

        visualize_decision_boundary(model, train_loader_original_subset, 'decision_boundary_100_original_cifar.png')
        visualize_decision_boundary(model, train_loader_augmented_subset, 'decision_boundary_100_augmented_cifar.png')

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    if mean_accuracy > best_mean_accuracy:
        best_mean_accuracy = mean_accuracy
        torch.save(model.state_dict(), 'cifar-dawid.model')

    print(f'{mean_accuracy=}; {std_accuracy=}')

## BRAK AUGMENTACJI
# wszystkie dane
# mean_accuracy=0.17402; std_accuracy=0.14804

# 100
# mean_accuracy=0.10688; std_accuracy=0.013311558886922299

# 200
# mean_accuracy=0.09702000000000002; std_accuracy=0.005960000000000004

# 1000
# mean_accuracy=0.12858; std_accuracy=0.03162621697263206
