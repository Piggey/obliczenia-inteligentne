import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Zbuduj model MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(13, 10)
        self.hidden2 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 3)
    
    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

if __name__ == '__main__':
  # Załaduj dane
  wine = load_wine()
  X = wine.data
  y = wine.target

  # Podziel dane na zbiór treningowy i testowy
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Znormalizuj dane
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # Konwersja danych do tensorów PyTorch
  X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
  y_train_tensor = torch.tensor(y_train, dtype=torch.long)
  X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
  y_test_tensor = torch.tensor(y_test, dtype=torch.long)



  model = MLP()

  # Hiperparametry
  learning_rate = 0.01
  num_epochs = 100

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

      if (epoch+1) % 10 == 0:
          print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

  # Oceń dokładność modelu
  with torch.no_grad():
      test_outputs = model(X_test_tensor)
      _, predicted = torch.max(test_outputs, 1)
      accuracy = accuracy_score(y_test_tensor, predicted)
      print(f'Accuracy on test set: {accuracy:.4f}')

  # Zapisz model do pliku
  torch.save(model.state_dict(), 'project_3/models/wine/mlp_wine_model.pth')
  print('Model saved to mlp_wine_model.pth')
