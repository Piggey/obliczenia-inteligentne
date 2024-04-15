# project 1; task 2; experiment 1;
import numpy as np
from utils import load_data
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay


# plot decision boundary for each dataset, for rbf and linear kernel, for C with best accuracy
def analyze_svm(X, y, dataset_name):
    # find Cs with best accuracy for rbf kernel for each dataset
    for kernel in ["rbf", "linear"]:
        best_C = -1
        best_accuracy = -1
        for C in np.arange(0.5, 100, 0.5):
            classifier = SVC(kernel=kernel, C=C)
            classifier.fit(X, y)
            y_pred = classifier.predict(X)
            accuracy = accuracy_score(y, y_pred)

            if best_accuracy < accuracy:
                best_accuracy = accuracy
                best_C = C

        print(f'For dataset {dataset_name} best accuracy = {best_accuracy}; at C = {best_C}')

        classifier = SVC(kernel=kernel, C=best_C)
        classifier.fit(X, y)
        display = DecisionBoundaryDisplay.from_estimator(classifier, X, response_method="predict", alpha=0.5)
        display.ax_.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
        plt.title(
            f'DB plot for SVC; kernel = {kernel}; Highest acc = {round(best_accuracy, 2)}; C = {best_C}; {dataset_name} dataset')
        plt.show()


# plot decision boundary for each dataset, for identity, logistic, tanh and relu activation functions, for n_neurons with best accuracy
def analyze_mlp():
    pass


def experiment_one():
    # load datasets
    dataset_2_1 = load_data("./data/2_1.csv")
    dataset_2_2 = load_data("./data/2_2.csv")
    dataset_2_3 = load_data("./data/2_3.csv")

    analyze_svm(dataset_2_1[0], dataset_2_1[1], "2_1")
    analyze_svm(dataset_2_2[0], dataset_2_2[1], "2_2")
    analyze_svm(dataset_2_3[0], dataset_2_3[1], "2_3")


experiment_one()

