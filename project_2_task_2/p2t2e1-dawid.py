from typing import Union
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch
import torchvision.datasets as datasets
import torch.utils.data as data

device = torch.device('cpu')

transform = transforms.Compose([
    transforms.ToTensor(),  # Konwersja obrazu PIL na tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalizacja obrazu
])

learning_rate = 0.001
num_epochs = 10

class TwoFeaturesCNNDawid(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, device=device), # Wejście: 1 kanał (obraz szarości), wyjście: 32 kanały, kernel: 3x3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(32, 64, kernel_size=3, device=device), # Wejście: 32 kanały, wyjście: 64 kanały, kernel: 3x3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )
        self.two_features = nn.Sequential(
            nn.Linear(64*5*5, 128, device=device), # Wejście: 64*5*5 (rozmiar obrazu po przetworzeniu przez dwie warstwy konwolucyjne), wyjście: 128
            nn.Linear(128, 2, device=device), # Wejście: 128, wyjście: 2 (2 cechy)
        )
        self.classify = nn.Linear(2, 10, device=device)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        # Spłaszczanie tensora dla warstw w pełni połączonych
        x = x.view(-1, 64*5*5)
        x = self.two_features(x)
        two_features = x
        x = self.classify(x)

        return x, two_features


model = TwoFeaturesCNNDawid()

# Pobranie danych treningowych
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = data.DataLoader(trainset, batch_size=1024, shuffle=True)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = data.DataLoader(testset, batch_size=1024, shuffle=True)

# Definicja funkcji straty i optymizatora
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Trenowanie sieci
num_epochs = 5
def accuracy(model: nn.Module, testloader: data.DataLoader) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


accuracies = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        optimizer.zero_grad()

        # Forward pass
        outputs, two_features = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass i aktualizacja wag
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print('[%d, %2d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    acc_train = accuracy(model, trainloader)
    acc_test = accuracy(model, testloader)
    print(f'{epoch=}; {acc_train=}; {acc_test=}')
    accuracies.append([epoch + 1, acc_train, acc_test])

print(accuracies)


# [[1, 0.45211666666666667, 0.4587], [2, 0.5237, 0.5246], [3, 0.6100333333333333, 0.6084], [4, 0.68645, 0.6786], [5, 0.7465666666666667, 0.7405]]