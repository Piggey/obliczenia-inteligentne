# best model is selected based on the accuracy
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from project_2_task_1.custom_mlp import CustomMLP, model_accuracy, flatten_transform, train_model, CustomDataset


def select_best_model(hidden_layer_sizes, input_size, output_size, train_dataloader, test_dataloader, dataset_name):
    best_model = CustomMLP(input_size, output_size, 1)
    best_hls = -1
    best_accuracy = -1
    accuracies = []

    for hls in hidden_layer_sizes:
        print(f"Training MLP on {dataset_name} dataset; hls={hls}")
        model = CustomMLP(input_size, output_size, hls)

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        train_model(model, loss_function, optimizer, 15, train_dataloader, test_dataloader)

        accuracy = model_accuracy(model, test_dataloader)
        accuracies.append(accuracy)

        if accuracy > best_accuracy:
            best_model = model
            best_accuracy = accuracy
            best_hls = hls


    plt.plot(hidden_layer_sizes, accuracies, '--bo')
    plt.xlabel("hls")
    plt.ylabel("accuracy")
    plt.ylim([0, 1])
    plt.title(f"Comparison of test data accuracy to hidden layer size on {dataset_name} dataset; Best hls={best_hls} acc={best_accuracy}")
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
    scaled_iris_X = StandardScaler().fit_transform(dataset_iris[0], dataset_iris[1])

    iris_train_X, iris_train_y = torch.tensor(scaled_iris_X[:120], dtype=torch.float32), torch.tensor(dataset_iris[1][:120], dtype=torch.long)
    iris_test_X, iris_test_y = torch.tensor(scaled_iris_X[120:], dtype=torch.float32), torch.tensor(dataset_iris[1][120:], dtype=torch.long)

    iris_train = CustomDataset((iris_train_X, iris_train_y))
    iris_test = CustomDataset((iris_test_X, iris_test_y))

    train_iris_dataloader = DataLoader(dataset=iris_train, batch_size=2048, shuffle=False)
    test_iris_dataloader = DataLoader(dataset=iris_test, batch_size=2048, shuffle=False)

    return train_iris_dataloader, test_iris_dataloader


def prepare_wine_data():
    dataset_wine = load_wine(return_X_y=True)
    scaled_wine_X = StandardScaler().fit_transform(dataset_wine[0], dataset_wine[1])

    wine_train_X, wine_train_y = torch.tensor(scaled_wine_X[:142], dtype=torch.float32), torch.tensor(dataset_wine[1][:142], dtype=torch.long)
    wine_test_X, wine_test_y = torch.tensor(scaled_wine_X[142:], dtype=torch.float32), torch.tensor(dataset_wine[1][142:], dtype=torch.long)

    wine_train = CustomDataset((wine_train_X, wine_train_y))
    wine_test = CustomDataset((wine_test_X, wine_test_y))

    train_wine_dataloader = DataLoader(dataset=wine_train, batch_size=2048, shuffle=False)
    test_wine_dataloader = DataLoader(dataset=wine_test, batch_size=2048, shuffle=False)

    return train_wine_dataloader, test_wine_dataloader


def prepare_breast_cancer_data():
    dataset_breast_cancer = load_breast_cancer(return_X_y=True)
    scaled_breast_cancer_X = StandardScaler().fit_transform(dataset_breast_cancer[0], dataset_breast_cancer[1])

    breast_cancer_train_X, breast_cancer_train_y = torch.tensor(scaled_breast_cancer_X[:455], dtype=torch.float32), torch.tensor(dataset_breast_cancer[1][:455], dtype=torch.long)
    breast_cancer_test_X, breast_cancer_test_y = torch.tensor(scaled_breast_cancer_X[455:], dtype=torch.float32), torch.tensor(dataset_breast_cancer[1][455:], dtype=torch.long)

    breast_cancer_train = CustomDataset((breast_cancer_train_X, breast_cancer_train_y))
    breast_cancer_test = CustomDataset((breast_cancer_test_X, breast_cancer_test_y))

    train_breast_cancer_dataloader = DataLoader(dataset=breast_cancer_train, batch_size=2048, shuffle=False)
    test_breast_cancer_dataloader = DataLoader(dataset=breast_cancer_test, batch_size=2048, shuffle=False)

    return train_breast_cancer_dataloader, test_breast_cancer_dataloader


def experiment_two():
    train_mnist_flatten_dataloader, test_mnist_flatten_dataloader = prepare_pca_dataloaders()
    best_mnist_flatten = select_best_model([8, 32, 64, 128, 256, 512], 784, 10,train_mnist_flatten_dataloader, test_mnist_flatten_dataloader, "mnist flatten")

    train_mnist_pca_dataloader, test_mnist_pca_dataloader = prepare_pca_dataloaders()
    best_mnist_pca = select_best_model([2, 8, 16, 32, 48, 64], 2, 10, train_mnist_pca_dataloader, test_mnist_pca_dataloader, "mnist pca (Artur)")

    train_mnist_lda_dataloader, test_mnist_lda_dataloader = prepare_lda_dataloaders()
    best_mnist_lda = select_best_model([4, 16, 32, 64, 96, 128], 9, 10, train_mnist_lda_dataloader, test_mnist_lda_dataloader, "mnist lda (Artur)")

    train_iris_dataloader, test_iris_dataloader = prepare_iris_data()
    best_iris = select_best_model([64, 256, 512, 1024, 2048, 4096, 8192, 16384], 4, 3, train_iris_dataloader, test_iris_dataloader, "iris")

    train_wine_dataloader, test_wine_dataloader = prepare_wine_data()
    best_wine = select_best_model([8, 32, 64, 128, 256, 512], 13, 3, train_wine_dataloader, test_wine_dataloader, "wine")

    train_breast_cancer_dataloader, test_breast_cancer_dataloader = prepare_breast_cancer_data()
    best_breast_cancer = select_best_model([8, 32, 64, 128, 256, 512], 30, 2, train_breast_cancer_dataloader, test_breast_cancer_dataloader, "breast_cancer")


experiment_two()
