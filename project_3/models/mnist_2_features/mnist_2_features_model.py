import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from skimage import feature

# Załaduj dane MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Konwersja danych do numpy array
X_train = train_dataset.data.numpy()
y_train = train_dataset.targets.numpy()
X_test = test_dataset.data.numpy()
y_test = test_dataset.targets.numpy()

# Ekstrakcja cech
def extract_features(images):
    features = []
    for img in images:
        # Średnia jasność
        brightness = np.mean(img)
        
        # Ilość konturów
        edges = feature.canny(img / 255.0).astype(int)
        contour_count = np.sum(edges)
        
        features.append([brightness, contour_count])
    return np.array(features)

X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)

# Normalizacja danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

# Konwersja danych do tensorów PyTorch
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Zbuduj model MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(2, 16)
        self.hidden2 = nn.Linear(16, 16)
        self.output = nn.Linear(16, 10)
    
    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

model = MLP()

# Hiperparametry
learning_rate = 0.01
num_epochs = 20

# Funkcja kosztu i optymalizator
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Trening
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Oceń dokładność modelu
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = accuracy_score(y_test_tensor, predicted)
    print(f'Accuracy on test set: {accuracy:.4f}')

# Zapisz model do pliku
torch.save(model.state_dict(), 'project_3/models/mnist_2_features/mlp_mnist_2_features_model.pth')
print('Model saved to mlp_mnist_2_features_model.pth')
