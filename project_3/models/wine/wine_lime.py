import torch
import torch.nn as nn
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import Lime
from wine_model import MLP

model = MLP()
model.load_state_dict(torch.load('project_3/models/wine/mlp_wine_model.pth'))
model.eval()

# Załaduj dane wine
wine = load_wine()
X = wine.data
y = wine.target

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

# Stwórz obiekt Lime
lime = Lime(model)

# Przeprowadź analizę Lime dla wybranych próbek
explanations = []
for i in range(len(inputs)):
    exp = lime.attribute(inputs[i].unsqueeze(0), target=labels[i], n_samples=1000)
    explanations.append(exp)

# Nazwy cech
feature_names = wine.feature_names

# Wykresy dla każdej próbki
for i in range(len(inputs)):
    exp = explanations[i].detach().numpy()
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(exp[0])), exp[0], align='center')
    plt.xticks(range(len(exp[0])), feature_names, rotation=45)
    plt.title(f'Lime Explanation for Sample {i+1} (Class: {wine.target_names[labels[i]]})')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.savefig(f'project_3/models/wine/lime/lime_map_{wine.target_names[labels[i]]}.png')
    # plt.show()
