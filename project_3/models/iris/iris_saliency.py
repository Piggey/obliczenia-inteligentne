import torch
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import Saliency
from iris_model import MLP

model = MLP()
model.load_state_dict(torch.load('project_3/models/iris/mlp_iris_model.pth'))
model.eval()

# Załaduj dane iris
iris = load_iris()
X = iris.data
y = iris.target

# Normalizacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Konwersja danych do tensorów PyTorch
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Wybierz po jednej próbce z każdej klasy
unique_classes = np.unique(y)
sample_indices = [np.where(y == cls)[0][0] for cls in unique_classes]

# Wybierz te próbki do analizy
inputs = X_tensor[sample_indices]
labels = y_tensor[sample_indices]

# Stwórz obiekt Saliency
saliency = Saliency(model)

# Oblicz mapy saliency dla wybranych próbek
saliency_attr = saliency.attribute(inputs, target=labels)

# Konwersja wyników na numpy array dla łatwiejszej wizualizacji
saliency_attr = saliency_attr.detach().numpy()

# Nazwy cech
feature_names = iris.feature_names

# Wykresy dla każdej próbki
for i in range(len(inputs)):
    plt.figure(figsize=(12, 4))
    plt.bar(range(inputs.shape[1]), saliency_attr[i], align='center')
    plt.xticks(range(inputs.shape[1]), feature_names)
    plt.title(f'Saliency Map for Sample {i+1} (Class: {iris.target_names[labels[i]]})')
    plt.xlabel('Features')
    plt.ylabel('Saliency')
    plt.savefig(f'saliency_map_{iris.target_names[labels[i]]}.png')
    plt.show()
