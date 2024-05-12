# best model is selected based on the accuracy
import numpy as np
import torch
from matplotlib import pyplot as plt, cm
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from project_2_task_1.custom_mlp import CustomMLP, model_accuracy, flatten_transform, train_model, CustomDataset


def select_best_model(hidden_layer_sizes, input_size, output_size, epochs, train_dataloader, test_dataloader, dataset_name):
    best_model = CustomMLP(input_size, output_size, 1)
    best_hls = -1
    best_accuracy = -1
    test_accuracies = []
    train_accuracies = []

    for hls in hidden_layer_sizes:
        print(f"Training MLP on {dataset_name} dataset; hls={hls}")
        model = CustomMLP(input_size, output_size, hls)

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        train_model(model, loss_function, optimizer, epochs, train_dataloader, test_dataloader)

        test_accuracy = model_accuracy(model, test_dataloader)
        test_accuracies.append(test_accuracy)

        train_accuracy = model_accuracy(model, train_dataloader)
        train_accuracies.append(train_accuracy)

        if test_accuracy > best_accuracy:
            best_model = model
            best_accuracy = test_accuracy
            best_hls = hls


    plt.plot(hidden_layer_sizes, test_accuracies, '--bo', label="Accuracy over test dataset")
    plt.plot(hidden_layer_sizes, train_accuracies, '--ro', label="Accuracy over train dataset")
    plt.xlabel("hls")
    plt.ylabel("accuracy")
    plt.ylim([0, 1])
    plt.title(f"Comparison of test data accuracy to hidden layer size on {dataset_name} dataset; Best hls={best_hls} acc={best_accuracy}")
    plt.legend()
    plt.show()

    return best_model


def prepare_flatten_dataloaders():
    train_mnist_flatten = datasets.MNIST('../data', train=True, download=True, transform=flatten_transform)
    test_mnist_flatten = datasets.MNIST('../data', train=False, transform=flatten_transform)

    train_mnist_dataloader = DataLoader(dataset=train_mnist_flatten, batch_size=2048, shuffle=False)
    test_mnist_dataloader = DataLoader(dataset=test_mnist_flatten, batch_size=2048, shuffle=False)

    return train_mnist_dataloader, test_mnist_dataloader


def prepare_pca_dataloaders():
    train_mnist = datasets.MNIST('../data', train=True, download=True, transform=flatten_transform)
    test_mnist = datasets.MNIST('../data', train=False, transform=flatten_transform)
    train_mnist_dataloader = DataLoader(dataset=train_mnist, batch_size=len(train_mnist), shuffle=False)
    test_mnist_dataloader = DataLoader(dataset=test_mnist, batch_size=len(test_mnist), shuffle=False)

    # Split train dataset into images and labels, create Principal Component Analysis object, fit and use it
    train_images, train_labels = next(iter(train_mnist_dataloader))
    pca = PCA(n_components=2, random_state=42)
    pca.fit(train_images.numpy())
    train_images_2d = torch.tensor(pca.transform(train_images.numpy()))
    train_mnist = CustomDataset((train_images_2d, train_labels))
    train_mnist_dataloader = DataLoader(dataset=train_mnist, batch_size=2048, shuffle=False)

    # Split test dataset into images and labels, apply Principal Component Analysis trained earlier
    test_images, test_labels = next(iter(test_mnist_dataloader))
    test_images_2d = torch.tensor(pca.transform(test_images.numpy()))
    test_mnist = CustomDataset((test_images_2d, test_labels))
    test_mnist_dataloader = DataLoader(dataset=test_mnist, batch_size=2048, shuffle=False)

    return train_mnist_dataloader, test_mnist_dataloader


def prepare_lda_dataloaders():
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
    train_mnist_dataloader = DataLoader(dataset=train_mnist_dataset, batch_size=2048, shuffle=False)

    # Split test dataset into images and labels, apply Linear Discriminant Analysis fitted earlier
    test_images, test_labels = next(iter(test_mnist_dataloader))
    test_images_lda = torch.tensor(lda.transform(test_images.numpy()), dtype=torch.float32)
    test_mnist_dataset = CustomDataset((test_images_lda, test_labels))
    test_mnist_dataloader = DataLoader(dataset=test_mnist_dataset, batch_size=2048, shuffle=False)

    return train_mnist_dataloader, test_mnist_dataloader


def prepare_iris_data():
    dataset_iris = load_iris(return_X_y=True)
    classes = load_iris().target_names
    scaled_iris_X = StandardScaler().fit_transform(dataset_iris[0], dataset_iris[1])
    iris_train_X, iris_test_X, iris_train_y, iris_test_y = train_test_split(scaled_iris_X, dataset_iris[1], test_size=0.2, random_state=42)

    iris_train_X, iris_train_y = torch.tensor(iris_train_X, dtype=torch.float32), torch.tensor(iris_train_y, dtype=torch.long)
    iris_test_X, iris_test_y = torch.tensor(iris_test_X, dtype=torch.float32), torch.tensor(iris_test_y, dtype=torch.long)

    iris_train = CustomDataset((iris_train_X, iris_train_y))
    iris_test = CustomDataset((iris_test_X, iris_test_y))

    train_iris_dataloader = DataLoader(dataset=iris_train, batch_size=2048, shuffle=False)
    test_iris_dataloader = DataLoader(dataset=iris_test, batch_size=2048, shuffle=False)

    return train_iris_dataloader, test_iris_dataloader, classes


def prepare_wine_data():
    dataset_wine = load_wine(return_X_y=True)
    classes = load_wine().target_names
    scaled_wine_X = StandardScaler().fit_transform(dataset_wine[0], dataset_wine[1])
    wine_train_X, wine_test_X, wine_train_y, wine_test_y = train_test_split(scaled_wine_X, dataset_wine[1], test_size=0.2, random_state=42)

    wine_train_X, wine_train_y = torch.tensor(wine_train_X, dtype=torch.float32), torch.tensor(wine_train_y, dtype=torch.long)
    wine_test_X, wine_test_y = torch.tensor(wine_test_X, dtype=torch.float32), torch.tensor(wine_test_y, dtype=torch.long)


    wine_train = CustomDataset((wine_train_X, wine_train_y))
    wine_test = CustomDataset((wine_test_X, wine_test_y))

    train_wine_dataloader = DataLoader(dataset=wine_train, batch_size=2048, shuffle=False)
    test_wine_dataloader = DataLoader(dataset=wine_test, batch_size=2048, shuffle=False)

    return train_wine_dataloader, test_wine_dataloader, classes


def prepare_breast_cancer_data():
    dataset_breast_cancer = load_breast_cancer(return_X_y=True)
    classes = load_breast_cancer().target_names
    scaled_breast_cancer_X = StandardScaler().fit_transform(dataset_breast_cancer[0], dataset_breast_cancer[1])
    breast_cancer_train_X, breast_cancer_test_X, breast_cancer_train_y, breast_cancer_test_y = train_test_split(scaled_breast_cancer_X, dataset_breast_cancer[1], test_size=0.2, random_state=42)

    breast_cancer_train_X, breast_cancer_train_y = torch.tensor(breast_cancer_train_X, dtype=torch.float32), torch.tensor(breast_cancer_train_y, dtype=torch.long)
    breast_cancer_test_X, breast_cancer_test_y = torch.tensor(breast_cancer_test_X, dtype=torch.float32), torch.tensor(breast_cancer_test_y, dtype=torch.long)

    breast_cancer_train = CustomDataset((breast_cancer_train_X, breast_cancer_train_y))
    breast_cancer_test = CustomDataset((breast_cancer_test_X, breast_cancer_test_y))

    train_breast_cancer_dataloader = DataLoader(dataset=breast_cancer_train, batch_size=2048, shuffle=False)
    test_breast_cancer_dataloader = DataLoader(dataset=breast_cancer_test, batch_size=2048, shuffle=False)

    return train_breast_cancer_dataloader, test_breast_cancer_dataloader, classes


def plot_confusion_matrix_from_dataloader(model, dataloader, classes):
    y_true = np.array([])
    y_pred = np.array([])

    for inputs, labels in dataloader:
        y_pred = np.append(y_pred, np.argmax(model(inputs).detach().numpy(), axis=1))
        y_true = np.append(y_true, labels)

    conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
    display.plot()
    plt.show()


def plot_decision_boundary_from_dataloader(model, dataloader):
    y_true = np.array([])
    X = np.empty((0, 2))

    for inputs, labels in dataloader:
        y_true = np.append(y_true, labels)
        X = np.append(X, inputs.numpy(), axis=0)
        print(inputs.numpy())

    print(X)

    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

    # Create predictions for each element of grid in decision boundary
    with torch.no_grad():
        predictions = np.argmax(model(grid_tensor).detach().numpy(), axis=1)

    # Add dummy points to get labels on the final plot
    for i in range(10):
        plt.scatter(-999, -999, alpha=1, label=str(i), cmap=cm.tab10)

    plt.scatter(xx, yy, c=predictions, alpha=1, cmap=cm.tab10, s=50, marker="s", edgecolor='none')
    plt.scatter(X[:500, 0], X[:500, 1], c=y_true[:500], edgecolors='k', s=20, cmap=cm.tab10)
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


def experiment_two():
    train_mnist_flatten_dataloader, test_mnist_flatten_dataloader = prepare_pca_dataloaders()
    best_mnist_flatten = select_best_model([8, 32, 64, 128, 256, 512], 784, 10, 10, train_mnist_flatten_dataloader, test_mnist_flatten_dataloader, "mnist flatten")
    plot_confusion_matrix_from_dataloader(best_mnist_flatten, test_mnist_flatten_dataloader, np.arange(10))

    train_mnist_pca_dataloader, test_mnist_pca_dataloader = prepare_pca_dataloaders()
    best_mnist_pca = select_best_model([2, 8, 16, 32, 48, 64], 2, 10, 1, train_mnist_pca_dataloader, test_mnist_pca_dataloader, "mnist PCA (Artur)")
    plot_confusion_matrix_from_dataloader(best_mnist_pca, test_mnist_pca_dataloader, np.arange(10))
    plot_decision_boundary_from_dataloader(best_mnist_pca, test_mnist_pca_dataloader)

    train_mnist_lda_dataloader, test_mnist_lda_dataloader = prepare_lda_dataloaders()
    best_mnist_lda = select_best_model([4, 16, 32, 64, 96, 128], 9, 10, 10, train_mnist_lda_dataloader, test_mnist_lda_dataloader, "mnist LDA (Artur)")
    plot_confusion_matrix_from_dataloader(best_mnist_lda, test_mnist_lda_dataloader, np.arange(10))

    train_iris_dataloader, test_iris_dataloader, iris_classes = prepare_iris_data()
    best_iris = select_best_model([64, 256, 512, 1024, 2048, 4096, 8192, 16384], 4, 3, 100, train_iris_dataloader, test_iris_dataloader, "iris")
    plot_confusion_matrix_from_dataloader(best_iris, test_iris_dataloader, iris_classes)

    train_wine_dataloader, test_wine_dataloader, wine_classes = prepare_wine_data()
    best_wine = select_best_model([8, 32, 64, 128, 256, 512], 13, 3, 100, train_wine_dataloader, test_wine_dataloader, "wine")
    plot_confusion_matrix_from_dataloader(best_wine, test_wine_dataloader, wine_classes)

    train_breast_cancer_dataloader, test_breast_cancer_dataloader, breast_cancer_classes = prepare_breast_cancer_data()
    best_breast_cancer = select_best_model([8, 32, 64, 128, 256, 512], 30, 2, 100, train_breast_cancer_dataloader, test_breast_cancer_dataloader, "breast_cancer")
    plot_confusion_matrix_from_dataloader(best_breast_cancer, train_breast_cancer_dataloader, breast_cancer_classes)


experiment_two()
