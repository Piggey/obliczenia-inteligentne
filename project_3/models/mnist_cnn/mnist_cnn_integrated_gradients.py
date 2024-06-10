import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import IntegratedGradients
from mnist_cnn_model import CNN

# Załaduj wytrenowany model
model = CNN()
model.load_state_dict(torch.load('project_3/models/mnist_cnn/mnist_cnn_model.pth'))
model.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=True)

# Wybierz próbkę do analizy
dataiter = iter(testloader)
images, labels = dataiter._next_data()
image = images[0]
label = labels[0]

# Przeprowadź analizę Integrated Gradients
ig = IntegratedGradients(model)
attributions = ig.attribute(image.unsqueeze(0), target=label, n_steps=50)

# Konwersja wyników do formatu numpy
attributions = attributions.squeeze().cpu().detach().numpy()
image = image.squeeze().cpu().detach().numpy()

# Wizualizacja wyników
plt.figure(figsize=(10, 4))

# Oryginalny obraz
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

# Integrated Gradients
plt.subplot(1, 2, 2)
plt.title('Integrated Gradients')
plt.imshow(attributions, cmap='hot', interpolation='nearest')

plt.savefig('project_3/models/mnist_cnn/integrated_gradients/integrated_gradients.png')
plt.show()
