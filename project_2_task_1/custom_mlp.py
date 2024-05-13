import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class CustomDataset(Dataset):
    def __init__(self, scikit_dataset):
        self.data = scikit_dataset  # with return_X_y = true, is tuple (data, target)

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]


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


flatten_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: torch.flatten(x)),
])