import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoFeaturesArturCNN(nn.Module):
    def __init__(self):
        super(TwoFeaturesArturCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 2)
        self.fc2 = nn.Linear(2, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 3 * 3 * 64)
        x = F.relu(self.fc1(x))
        two_features = x
        x = self.fc2(x)

        return F.log_softmax(x, dim=1), two_features


class ArturCNN(nn.Module):
    def __init__(self):
        super(ArturCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 3 * 3 * 64)
        x = F.relu(self.fc1(x))
        two_features = x
        x = self.fc2(x)

        return F.log_softmax(x, dim=1), two_features


def model_accuracy(model, dataloader):
    model.eval()

    total_correct = 0
    total_instances = 0

    #  iterating through batches
    with torch.no_grad():
        for inputs, labels in dataloader:
            classifications = torch.argmax(model(inputs), dim=1)
            correct_predictions = sum(classifications == labels).item()
            total_correct += correct_predictions
            total_instances += len(inputs)

    return round(total_correct / total_instances, 3)


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


def analyze_two_features_cnn():
    pass


def analyze_n_features_cnn():
    pass


def experiment_one():
    pass
