import torch.nn as nn
import torch.nn.functional as F


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

