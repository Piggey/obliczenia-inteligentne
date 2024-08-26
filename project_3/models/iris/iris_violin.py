from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from captum.attr import IntegratedGradients

from iris_model import MLP

# Wczytaj dane Iris
iris = load_iris()
X = iris['data']
y = iris['target']
feature_names = iris['feature_names']
label_names = iris['target_names']

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

X_sample = torch.tensor(X_test, dtype=torch.float32)
y_sample = torch.tensor(y_test, dtype=torch.long)

model = MLP()
model.load_state_dict(torch.load('project_3/models/iris/mlp_iris_model.pth'))
model.eval()

# Przeprowadź analizę Integrated Gradients
ig = IntegratedGradients(model)
attributions = ig.attribute(X_sample, target=y_sample, n_steps=50)

# Konwersja wyników do formatu numpy
attributions = attributions.detach().numpy()

# Przygotuj dane do wizualizacji
df = pd.DataFrame(X_sample.numpy(), columns=feature_names)
df['label'] = y_sample.numpy()
attr_df = pd.DataFrame(attributions, columns=feature_names)
attr_df['label'] = y_sample.numpy()

# Utwórz wykresy wiolinowe dla atrybutów (ważności cech)
for i, feature in enumerate(feature_names):
    plt.figure(figsize=(10, 6))
    data = [attr_df[attr_df['label'] == label][feature] for label in np.unique(y_sample)]
    plt.violinplot(data, showmeans=True)
    plt.title(f'Violin Plot of {feature} Importance')
    plt.xlabel('Class')
    plt.ylabel(f'{feature} Importance')
    plt.xticks(ticks=[1, 2, 3], labels=label_names)
    plt.savefig(f'iris_atrybuty_{feature}.png')
    plt.close()
    # plt.show()
