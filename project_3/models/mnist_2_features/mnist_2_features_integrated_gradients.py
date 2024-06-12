import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from skimage import feature

# Definiujemy klasę MLP
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

# Załaduj model
model = MLP()
model.load_state_dict(torch.load('project_3/models/mnist_2_features/mlp_mnist_2_features_model.pth'))
model.eval()

# Przygotuj dane MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
testset = MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

# Funkcja wyodrębniająca cechy
def extract_features(images):
    features = []
    for img in images:
        img = img.numpy().squeeze()
        # Średnia jasność
        brightness = np.mean(img)
        
        # Ilość konturów
        edges = feature.canny(img / 255.0).astype(int)
        contour_count = np.sum(edges)
        
        features.append([brightness, contour_count])
    return torch.tensor(features, dtype=torch.float32)

# Wybierz próbkę do analizy
dataiter = iter(testloader)
images, labels = dataiter._next_data()
image = images[0]
label = labels[0]

# Wyodrębnij cechy
features = extract_features(images)

# Przeprowadź analizę Integrated Gradients
ig = IntegratedGradients(model)
attributions = ig.attribute(features, target=label, n_steps=50)

# Konwersja wyników do formatu numpy
attributions = attributions.squeeze().cpu().detach().numpy()
features = features.squeeze().cpu().detach().numpy()

# Nazwy cech
feature_names = ['Mean Brightness', 'Contour Count']

# Wizualizacja wyników
plt.figure(figsize=(6, 4))
plt.bar(range(len(attributions)), attributions, align='center')
plt.xticks(range(len(attributions)), feature_names)
plt.title(f'Integrated Gradients for Sample (Class: {label.item()})')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.savefig(f'project_3/models/mnist_2_features/integrated_gradients/integrated_gradients_{label.item()}.png')
# plt.show()
