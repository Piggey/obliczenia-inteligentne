import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Przygotowanie danych
transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Wybór losowego przykładu z datasetu
index = np.random.randint(len(trainset))
image, _ = trainset[index]

# Tworzenie przekształcenia za pomocą transforms.RandomAffine
random_affine = transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2))
transformed_image = random_affine(image)

# Konwersja obrazków do formatu numpy
image = image.numpy()
transformed_image = transformed_image.numpy()

# Wyświetlenie obrazków
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(np.transpose(image, (1, 2, 0)))  # Przekształcenie obrazu z formatu (C, H, W) do (H, W, C)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(np.transpose(transformed_image, (1, 2, 0)))  # Przekształcenie obrazu z formatu (C, H, W) do (H, W, C)
axes[1].set_title('Transformed Image')
axes[1].axis('off')

plt.savefig('random-affine-cifar.png')
plt.close()