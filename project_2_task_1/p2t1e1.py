import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from custom_mlp import CustomDataset, CustomMLP, model_accuracy, train_model, flatten_transform
import torch
import torch.optim as optim
from torch import nn
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from matplotlib import cm

from utils import plot_voronoi_diagram


def flatten_analysis():
    train_mnist = datasets.MNIST('../data', train=True, download=True, transform=flatten_transform)
    test_mnist = datasets.MNIST('../data', train=False, transform=flatten_transform)
    train_mnist_dataloader = DataLoader(dataset=train_mnist, batch_size=128, shuffle=False)
    test_mnist_dataloader = DataLoader(dataset=test_mnist, batch_size=128, shuffle=False)

    # Create and train model
    model = CustomMLP(784, 10, 32)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    train_model(model, loss_function, optimizer, 15, train_mnist_dataloader, test_mnist_dataloader)

    # Plot confusion matrix
    test_mnist_dataloader = DataLoader(dataset=test_mnist, batch_size=len(test_mnist), shuffle=False)
    test_images, test_labels = next(iter(test_mnist_dataloader))
    conf_matrix = confusion_matrix(test_labels, np.argmax(model(test_images).detach().numpy(), axis=1), labels=np.unique(test_labels))
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(test_labels))
    display.plot()
    plt.title(f'Confusion matrix for test MNIST dataset reduced using PCA with n_components=2')
    plt.show()


def pca_2_components_analysis():
    train_mnist = datasets.MNIST('data', train=True, download=True, transform=flatten_transform)
    test_mnist = datasets.MNIST('data', train=False, transform=flatten_transform)
    train_mnist_dataloader = DataLoader(dataset=train_mnist, batch_size=len(train_mnist), shuffle=False)
    test_mnist_dataloader = DataLoader(dataset=test_mnist, batch_size=len(test_mnist), shuffle=False)

    # Split train dataset into images and labels, create Principal Component Analysis object, fit and use it
    train_images, train_labels = next(iter(train_mnist_dataloader))
    pca = PCA(n_components=2, random_state=42)
    pca.fit(train_images.numpy())
    train_images_2d = torch.tensor(pca.transform(train_images.numpy()))
    train_mnist = CustomDataset((train_images_2d, train_labels))
    train_mnist_dataloader = DataLoader(dataset=train_mnist, batch_size=128, shuffle=False)

    # Split test dataset into images and labels, apply Principal Component Analysis trained earlier
    test_images, test_labels = next(iter(test_mnist_dataloader))
    test_images_2d = torch.tensor(pca.transform(test_images.numpy()))
    test_mnist = CustomDataset((test_images_2d, test_labels))
    test_mnist_dataloader = DataLoader(dataset=test_mnist, batch_size=128, shuffle=False)

    # Create and train model
    model = CustomMLP(2, 10, 32)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    train_model(model, loss_function, optimizer, 15, train_mnist_dataloader, test_mnist_dataloader)

    # Show voronoi for 500 of test data
    plot_voronoi_diagram(test_images_2d[:500], test_labels[:500], np.argmax(model(test_images_2d).detach().numpy()[:500], axis=1),  "PCA with n_components=2 for 500 examples from test dataset")

    # Plot decision boundary
    x_min, x_max = test_images_2d[:, 0].min() - 1, test_images_2d[:, 0].max() + 1
    y_min, y_max = test_images_2d[:, 1].min() - 1, test_images_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

    # Create predictions for each element of grid in decision boundary
    with torch.no_grad():
        predictions = np.argmax(model(grid_tensor).detach().numpy(), axis=1)

    # Add dummy points to get labels on the final plot
    for i in range(10):
        plt.scatter(-999, -999, alpha=1, label=str(i), cmap=cm.tab10)

    # Add grid as square points because contourf didn't work
    plt.scatter(xx, yy, c=predictions, alpha=1, cmap=cm.tab10, s=50, marker="s", edgecolor='none')
    plt.scatter(test_images_2d[:500, 0], test_images_2d[:500, 1], c=test_labels[:500], edgecolors='k', s=20, cmap=cm.tab10)
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Decision boundary plot for test MNIST dataset reduced using PCA with n_components=2')
    plt.legend()
    plt.show()

    # Plot confusion matrix
    conf_matrix = confusion_matrix(test_labels, np.argmax(model(test_images_2d).detach().numpy(), axis=1), labels=np.unique(test_labels))
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(test_labels))
    display.plot()
    plt.title(f'Confusion matrix for test MNIST dataset reduced using PCA with n_components=2')
    plt.show()


def lda_n_components_analysis():
    train_mnist = datasets.MNIST('../data', train=True, download=True, transform=flatten_transform)
    test_mnist = datasets.MNIST('../data', train=False, transform=flatten_transform)
    train_mnist_dataloader = DataLoader(dataset=train_mnist, batch_size=len(train_mnist), shuffle=False)
    test_mnist_dataloader = DataLoader(dataset=test_mnist, batch_size=len(test_mnist), shuffle=False)

    # Split train dataset into images and labels, create Linear Discriminant Analysis object, fit and use it
    train_images, train_labels = next(iter(train_mnist_dataloader))
    lda = LDA()
    lda.fit(train_images, train_labels)
    train_images_lda = torch.tensor(lda.transform(train_images.numpy()), dtype=torch.float32)
    train_mnist_dataset = CustomDataset((train_images_lda, train_labels))
    train_mnist_dataloader = DataLoader(dataset=train_mnist_dataset, batch_size=128, shuffle=False)

    # Split test dataset into images and labels, apply Linear Discriminant Analysis fitted earlier
    test_images, test_labels = next(iter(test_mnist_dataloader))
    test_images_lda = torch.tensor(lda.transform(test_images.numpy()), dtype=torch.float32)
    test_mnist_dataset = CustomDataset((test_images_lda, test_labels))
    test_mnist_dataloader = DataLoader(dataset=test_mnist_dataset, batch_size=128, shuffle=False)

    n_features = test_images_lda.shape[1]
    print(f"N features after LDA = {n_features}")

    # Create and train model
    model = CustomMLP(test_images_lda.shape[1], 10, 32)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    train_model(model, loss_function, optimizer, 15, train_mnist_dataloader, test_mnist_dataloader)

    # Plot confusion matrix
    conf_matrix = confusion_matrix(test_labels, np.argmax(model(test_images_lda).detach().numpy(), axis=1), labels=np.unique(test_labels))
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(test_labels))
    display.plot()
    plt.title(f'Confusion matrix for test MNIST dataset reduced using LDA with n_features={n_features}')
    plt.show()

def ipca_n_components_analysis():
    train_mnist = datasets.MNIST('data', train=True, download=True, transform=flatten_transform)
    test_mnist = datasets.MNIST('data', train=False, transform=flatten_transform)
    train_mnist_dataloader = DataLoader(dataset=train_mnist, batch_size=len(train_mnist), shuffle=False)
    test_mnist_dataloader = DataLoader(dataset=test_mnist, batch_size=len(test_mnist), shuffle=False)

    # Split train dataset into images and labels, create Linear Discriminant Analysis object, fit and use it
    train_images, train_labels = next(iter(train_mnist_dataloader))
    lda = IncrementalPCA(n_components=28)
    lda.fit(train_images, train_labels)
    train_images_lda = torch.tensor(lda.transform(train_images.numpy()), dtype=torch.float32)
    train_mnist_dataset = CustomDataset((train_images_lda, train_labels))
    train_mnist_dataloader = DataLoader(dataset=train_mnist_dataset, batch_size=128, shuffle=False)

    # Split test dataset into images and labels, apply Linear Discriminant Analysis fitted earlier
    test_images, test_labels = next(iter(test_mnist_dataloader))
    test_images_lda = torch.tensor(lda.transform(test_images.numpy()), dtype=torch.float32)
    test_mnist_dataset = CustomDataset((test_images_lda, test_labels))
    test_mnist_dataloader = DataLoader(dataset=test_mnist_dataset, batch_size=128, shuffle=False)

    # Create and train model
    model = CustomMLP(28, 10, 32)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005)
    train_model(model, loss_function, optimizer, 15, train_mnist_dataloader, test_mnist_dataloader)

    # Plot confusion matrix
    conf_matrix = confusion_matrix(test_labels, np.argmax(model(test_images_lda).detach().numpy(), axis=1), labels=np.unique(test_labels))
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(test_labels))
    display.plot()
    plt.title(f'Test MNIST dataset; IncrementalPCA; n_components=28')
    plt.savefig("conf_matrix_mnist_ipca_n_components=28.png")

def ipca_2_components_analysis():
    train_mnist = datasets.MNIST('data', train=True, download=True, transform=flatten_transform)
    test_mnist = datasets.MNIST('data', train=False, transform=flatten_transform)
    train_mnist_dataloader = DataLoader(dataset=train_mnist, batch_size=len(train_mnist), shuffle=False)
    test_mnist_dataloader = DataLoader(dataset=test_mnist, batch_size=len(test_mnist), shuffle=False)

    # Split train dataset into images and labels, create Principal Component Analysis object, fit and use it
    train_images, train_labels = next(iter(train_mnist_dataloader))
    ipca = IncrementalPCA(n_components=2)
    ipca.fit(train_images.numpy())
    train_images_2d = torch.tensor(ipca.transform(train_images.numpy()), dtype=torch.float32)
    train_mnist = CustomDataset((train_images_2d, train_labels))
    train_mnist_dataloader = DataLoader(dataset=train_mnist, batch_size=128, shuffle=False)

    # Split test dataset into images and labels, apply Principal Component Analysis trained earlier
    test_images, test_labels = next(iter(test_mnist_dataloader))
    test_images_2d = torch.tensor(ipca.transform(test_images.numpy()), dtype=torch.float32)
    test_mnist = CustomDataset((test_images_2d, test_labels))
    test_mnist_dataloader = DataLoader(dataset=test_mnist, batch_size=128, shuffle=False)

    # Create and train model
    model = CustomMLP(2, 10, 32)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005)
    train_model(model, loss_function, optimizer, 15, train_mnist_dataloader, test_mnist_dataloader)

    # Show voronoi for 500 of test data
    plot_voronoi_diagram(test_images_2d[:500], test_labels[:500], np.argmax(model(test_images_2d).detach().numpy()[:500], axis=1),  "FastICA with n_components=2 for 500 examples from test dataset")

    # Plot decision boundary
    x_min, x_max = test_images_2d[:, 0].min() - 1, test_images_2d[:, 0].max() + 1
    y_min, y_max = test_images_2d[:, 1].min() - 1, test_images_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

    # Create predictions for each element of grid in decision boundary
    with torch.no_grad():
        predictions = np.argmax(model(grid_tensor).detach().numpy(), axis=1)

    # Add dummy points to get labels on the final plot
    for i in range(10):
        plt.scatter(-999, -999, alpha=1, label=str(i), cmap=cm.tab10)

    # Add grid as square points because contourf didn't work
    plt.scatter(xx, yy, c=predictions, alpha=1, cmap=cm.tab10, s=50, marker="s", edgecolor='none')
    plt.scatter(test_images_2d[:500, 0], test_images_2d[:500, 1], c=test_labels[:500], edgecolors='k', s=20, cmap=cm.tab10)
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Test MNIST dataset; IncrementalPCA; n_components=2')
    plt.legend()
    # plt.show()
    plt.savefig("decision_boundary_mnist_ipca_n_components=2.png")
    plt.close()

    # Plot confusion matrix
    conf_matrix = confusion_matrix(test_labels, np.argmax(model(test_images_2d).detach().numpy(), axis=1), labels=np.unique(test_labels))
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(test_labels))
    display.plot()
    plt.title('Test MNIST dataset; IncrementalPCA; n_components=2')
    plt.savefig("conf_matrix_mnist_ipca_n_components=2.png")
    plt.close()


def experiment_one():
    # # Visualize MNIST dataset
    # mnist = datasets.MNIST('data', download=False, train=False, transform=transforms.ToTensor())
    # mnist_dataloader = DataLoader(dataset=mnist, batch_size=100, shuffle=False)
    # images, labels = next(iter(mnist_dataloader))

    # fig, axs = plt.subplots(10, 10)
    # for i in range(10):
    #     for j in range(10):
    #         axs[i, j].imshow(images[i * 10 + j].reshape(28, 28), cmap=cm.gray)
    #         axs[i, j].axis('off')

    # plt.tight_layout(pad=0.1)
    # plt.show()

    # MNIST ekstrakcja - spłaszczenia do wektora 784 elementów
    # flatten_analysis()

    # TODO: MNIST 2x ekstrakcja - spłaszczenia do wektora 2 elementów (cech) (po jednym sposobie na osobe)
    # Artur - Principal Component Analysis
    # pca_2_components_analysis()

    # Dawid
    ipca_2_components_analysis()

    # TODO: MNIST 2x ekstrakcja - spłaszczenia do wektora z małą liczbą elementów (cech) (po jednym sposobie na osobe)
    # Artur - Linear Discriminant Analysis (9 cech)
    # lda_n_components_analysis()

    # Dawid
    # ipca_n_components_analysis()


experiment_one()
