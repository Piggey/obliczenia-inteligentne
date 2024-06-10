import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from breast_cancer_model import MLP

model = MLP()
model.load_state_dict(torch.load('project_3/models/breast_cancer/mlp_breast_cancer_model.pth'))
model.eval()

# Załaduj dane breast cancer
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

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

# Stwórz obiekt Integrated Gradients
ig = IntegratedGradients(model)

# Przeprowadź analizę Integrated Gradients dla wybranych próbek
explanations = []
for i in range(len(inputs)):
    exp = ig.attribute(inputs[i].unsqueeze(0), target=labels[i], n_steps=200)
    explanations.append(exp)

# Nazwy cech
feature_names = breast_cancer.feature_names

# Wykresy dla każdej próbki
for i in range(len(inputs)):
    exp = explanations[i].detach().numpy()
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(exp[0])), exp[0], align='center')
    plt.xticks(range(len(exp[0])), feature_names, rotation=45)
    plt.title(f'Integrated Gradients for Sample {i+1} (Class: {breast_cancer.target_names[labels[i]]})')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.savefig(f'project_3/models/breast_cancer/integrated_gradients/integrated_gradients_map_{breast_cancer.target_names[labels[i]]}.png')
    # plt.show()
