import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import numpy as np

class CNN(nn.Module):
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

trainset_original = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=TRANSFORM)

K = 2000
subsample_train_indices = torch.randperm(len(trainset_original))[:K]

transform_augmented = transforms.Compose([
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset_augmented = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_augmented)

trainset_combined = torch.utils.data.ConcatDataset([trainset_original, trainset_augmented])
TRAIN_LOADER = torch.utils.data.DataLoader(trainset_combined, batch_size=32, num_workers=2, sampler=torch.utils.data.SubsetRandomSampler(subsample_train_indices))

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=TRANSFORM)
TEST_LOADER = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

if __name__ == '__main__':
    accuracies = []
    best_mean_accuracy = 0

    for i in range(10):
        model = CNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train(model, criterion, optimizer)
        accuracy = test_accuracy(model, TEST_LOADER)
        accuracies.append(accuracy)
        print(f'pass {i}; {accuracy=}')

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    if mean_accuracy > best_mean_accuracy:
        best_mean_accuracy = mean_accuracy
        torch.save(model.state_dict(), '2d-mnist-augmented-dawid.model')

    print(f'{mean_accuracy=}; {std_accuracy=}')


## BRAK AUGMENTACJI
# wszystkie dane
# mean_accuracy=0.5063000000000001; std_accuracy=0.3937564018527191

# 100
# mean_accuracy=0.10489000000000001; std_accuracy=0.015133040011841643

# 200
# mean_accuracy=0.14056999999999997; std_accuracy=0.04664054137764698

# 1000
# mean_accuracy=0.22397; std_accuracy=0.11511936457434083

# AUGMENTACJA
# wszystkie dane
# mean_accuracy=0.5474600000000001; std_accuracy=0.4339635450127119

# 100
# mean_accuracy=0.14025; std_accuracy=0.04873898337060387

# 200
# mean_accuracy=0.16297999999999999; std_accuracy=0.06934079318842552

# 1000
# mean_accuracy=0.19356; std_accuracy=0.12185078744103378
 