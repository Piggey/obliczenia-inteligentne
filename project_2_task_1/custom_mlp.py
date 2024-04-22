import torch.nn as nn
from torch.utils.data import Dataset


class CustomMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(CustomMLP, self).__init__()
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

