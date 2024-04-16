# project 1; task 2; experiment 2;
import matplotlib.pyplot as plt
import numpy as np
from utils import load_data
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def analyze_knn(X_train, X_test, y_train, y_test, dataset_name):
    best_n_neighbors = -1
    best_accuracy = -1

    scores = []

    for n_neighbors in range(1, 15):
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        classifier.fit(X_train, y_train)

        # test over training data
        y_train_pred = classifier.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_train_pred)

        # test over testing data
        y_test_pred = classifier.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_test_pred)

        scores.append([n_neighbors, accuracy_test, accuracy_train])

        if best_accuracy < accuracy_test:
            best_accuracy = accuracy_test
            best_n_neighbors = n_neighbors

    # plot accuracy on train and test datasets over changing n_neighbors parameter
    scores = np.array(scores)
    plt.plot(scores[:, 0], scores[:, 1], label="accuracy test")
    plt.plot(scores[:, 0], scores[:, 2], label="accuracy train")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.show()
    # plt.savefig(f'{dataset_name}_knn_hyperparameters.png')
    # plt.close()
    print(f'For dataset {dataset_name} KNN best accuracy = {best_accuracy}; at n_neighbors = {best_n_neighbors}')

    # show decision boundary and confusion matrix for smallest, largest and the best n_neighbors parameter
    # smallest
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(X_train, y_train)
    display = DecisionBoundaryDisplay.from_estimator(classifier, X_test, response_method="predict", alpha=0.5)
    display.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
    plt.title(f'DB plot for smallest KNN; n_neighbors = 1; {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_smallest_db.png')
    # plt.close()

    conf_matrix = confusion_matrix(y_test, classifier.predict(X_test), labels=classifier.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classifier.classes_)
    display.plot()
    plt.title(f'Confusion matrix for smallest KNN; n_neighbors = 1; {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_smallest_cm.png')
    # plt.close()

    # largest
    classifier = KNeighborsClassifier(n_neighbors=14)
    classifier.fit(X_train, y_train)
    display = DecisionBoundaryDisplay.from_estimator(classifier, X_test, response_method="predict", alpha=0.5)
    display.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
    plt.title(f'DB plot for largest KNN; n_neighbors = 14; {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_largest_db.png')
    # plt.close()

    conf_matrix = confusion_matrix(y_test, classifier.predict(X_test), labels=classifier.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classifier.classes_)
    display.plot()
    plt.title(f'Confusion matrix for largest KNN; n_neighbors = 14; {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_largest_cm.png')
    # plt.close()

    # best
    classifier = KNeighborsClassifier(n_neighbors=best_n_neighbors)
    classifier.fit(X_train, y_train)
    display = DecisionBoundaryDisplay.from_estimator(classifier, X_test, response_method="predict", alpha=0.5)
    display.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
    plt.title(f'DB plot for best KNN; n_neighbors = {best_n_neighbors}; {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_best_db.png')
    # plt.close()

    conf_matrix = confusion_matrix(y_test, classifier.predict(X_test), labels=classifier.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classifier.classes_)
    display.plot()
    plt.title(f'Confusion matrix for best KNN; n_neighbors = {best_n_neighbors}; {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_best_cm.png')
    # plt.close()


def analyze_svm(X_train, X_test, y_train, y_test, dataset_name):
    # plot accuracy on train and test datasets over changing log(C) parameter
    analyzed_Cs = np.logspace(-2, 6, num=32, endpoint=True)

    best_C = -1
    best_accuracy = -1

    scores = []

    for C in analyzed_Cs:
        classifier = SVC(kernel="rbf", C=C)
        classifier.fit(X_train, y_train)

        # test over training data
        y_train_pred = classifier.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_train_pred)

        # test over testing data
        y_test_pred = classifier.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_test_pred)

        scores.append([C, accuracy_test, accuracy_train])

        if best_accuracy < accuracy_test:
            best_accuracy = accuracy_test
            best_C = C

    # plot accuracy on train and test datasets over changing n_neighbors parameter
    scores = np.array(scores)
    plt.plot(np.log10(scores[:, 0]), scores[:, 1], label="accuracy test")
    plt.plot(np.log10(scores[:, 0]), scores[:, 2], label="accuracy train")
    plt.xlabel("log(C)")
    plt.legend()
    plt.show()
    # plt.savefig(f'{dataset_name}_knn_hyperparameters.png')
    # plt.close()
    print(f'For dataset {dataset_name} SVM best accuracy = {best_accuracy}; at C = {best_C} or log(C) = {np.log10(best_C)}')

    # show decision boundary and confusion matrix for smallest, largest and the best n_neighbors parameter
    # smallest
    classifier = SVC(kernel="rbf", C=analyzed_Cs[0])
    classifier.fit(X_train, y_train)
    display = DecisionBoundaryDisplay.from_estimator(classifier, X_test, response_method="predict", alpha=0.5)
    display.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
    plt.title(f'DB plot for SVM smallest C; C = {round(analyzed_Cs[0], 2)}; {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_smallest_db.png')
    # plt.close()

    conf_matrix = confusion_matrix(y_test, classifier.predict(X_test), labels=classifier.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classifier.classes_)
    display.plot()
    plt.title(f'Confusion matrix for SVM smallest C; C = {round(analyzed_Cs[0], 2)}; {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_smallest_cm.png')
    # plt.close()

    # largest
    classifier = SVC(kernel="rbf", C=analyzed_Cs[-1])
    classifier.fit(X_train, y_train)
    display = DecisionBoundaryDisplay.from_estimator(classifier, X_test, response_method="predict", alpha=0.5)
    display.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
    plt.title(f'DB plot for SVM largest C; C = {round(analyzed_Cs[-1], 2)}; {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_largest_db.png')
    # plt.close()

    conf_matrix = confusion_matrix(y_test, classifier.predict(X_test), labels=classifier.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classifier.classes_)
    display.plot()
    plt.title(f'Confusion matrix for SVM largest C; C = {round(analyzed_Cs[-1], 2)}; {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_largest_cm.png')
    # plt.close()

    # best
    classifier = SVC(kernel="rbf", C=best_C)
    classifier.fit(X_train, y_train)
    display = DecisionBoundaryDisplay.from_estimator(classifier, X_test, response_method="predict", alpha=0.5)
    display.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
    plt.title(f'DB plot for SVM best C; C = {round(best_C, 2)}; {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_best_db.png')
    # plt.close()

    conf_matrix = confusion_matrix(y_test, classifier.predict(X_test), labels=classifier.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classifier.classes_)
    display.plot()
    plt.title(f'Confusion matrix for SVM best C; C = {round(best_C, 2)}; {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_best_cm.png')
    # plt.close()


def analyze_mlp(X_train, X_test, y_train, y_test, dataset_name):
    # plot accuracy on train and test datasets over changing hidden layer size
    analyzed_hls = np.arange(2, 65, 4)

    best_hls = -1
    best_accuracy = -1

    scores = []

    for hls in analyzed_hls:
        classifier = MLPClassifier(solver="sgd",
                                   activation="relu",
                                   random_state=42,
                                   max_iter=100_000,
                                   n_iter_no_change=100_000,
                                   tol=0,
                                   hidden_layer_sizes=hls
                                   )
        classifier.fit(X_train, y_train)

        # test over training data
        y_train_pred = classifier.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_train_pred)

        # test over testing data
        y_test_pred = classifier.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_test_pred)

        scores.append([hls, accuracy_test, accuracy_train])

        if best_accuracy < accuracy_test:
            best_accuracy = accuracy_test
            best_hls = hls

    scores = np.array(scores)
    plt.plot(scores[:, 0], scores[:, 1], label="accuracy test")
    plt.plot(scores[:, 0], scores[:, 2], label="accuracy train")
    plt.xlabel("hidden_layer_size")
    plt.legend()
    plt.show()
    # plt.savefig(f'{dataset_name}_knn_hyperparameters.png')
    # plt.close()
    print(f'For dataset {dataset_name} MLP best accuracy = {best_accuracy}; at hidden_layer_size = {best_hls}')

    # show decision boundary and confusion matrix for smallest, largest and the best hidden_layer_sizes
    # smallest
    classifier = MLPClassifier(solver="sgd",
                               activation="relu",
                               random_state=42,
                               max_iter=100_000,
                               n_iter_no_change=100_000,
                               tol=0,
                               hidden_layer_sizes=analyzed_hls[0]
                               )
    classifier.fit(X_train, y_train)
    display = DecisionBoundaryDisplay.from_estimator(classifier, X_test, response_method="predict", alpha=0.5)
    display.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
    plt.title(f'DB plot for MLP smallest hls; hls = {analyzed_hls[0]}; {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_smallest_db.png')
    # plt.close()

    conf_matrix = confusion_matrix(y_test, classifier.predict(X_test), labels=classifier.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classifier.classes_)
    display.plot()
    plt.title(f'Confusion matrix for MLP smallest hls; hls = {analyzed_hls[0]}; {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_smallest_cm.png')
    # plt.close()

    # largest
    classifier = MLPClassifier(solver="sgd",
                               activation="relu",
                               random_state=42,
                               max_iter=100_000,
                               n_iter_no_change=100_000,
                               tol=0,
                               hidden_layer_sizes=analyzed_hls[-1]
                               )
    classifier.fit(X_train, y_train)
    display = DecisionBoundaryDisplay.from_estimator(classifier, X_test, response_method="predict", alpha=0.5)
    display.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
    plt.title(f'DB plot for MLP largest hls; hls = {analyzed_hls[-1]}; {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_largest_db.png')
    # plt.close()

    conf_matrix = confusion_matrix(y_test, classifier.predict(X_test), labels=classifier.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classifier.classes_)
    display.plot()
    plt.title(f'Confusion matrix for MLP largest hls; hls = {analyzed_hls[-1]}; {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_largest_cm.png')
    # plt.close()

    # best
    classifier = MLPClassifier(solver="sgd",
                               activation="relu",
                               random_state=42,
                               max_iter=100_000,
                               n_iter_no_change=100_000,
                               tol=0,
                               hidden_layer_sizes=best_hls
                               )
    classifier.fit(X_train, y_train)
    display = DecisionBoundaryDisplay.from_estimator(classifier, X_test, response_method="predict", alpha=0.5)
    display.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
    plt.title(f'DB plot for MLP best hls; hls = {best_hls}; {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_best_db.png')
    # plt.close()

    conf_matrix = confusion_matrix(y_test, classifier.predict(X_test), labels=classifier.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classifier.classes_)
    display.plot()
    plt.title(f'Confusion matrix for MLP best hls; hls = {best_hls}; {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_best_cm.png')
    # plt.close()


def experiment_two():
    # load datasets
    dataset_2_2 = load_data("./data/2_2.csv")
    dataset_2_3 = load_data("./data/2_3.csv")

    # split 0.8 train / 0.2 test
    # [0] -> X_train; [1] -> X_test; [2] -> y_train; [3] -> y_test;
    dataset_2_2 = train_test_split(dataset_2_2[0], dataset_2_2[1], test_size=0.2, random_state=42)
    dataset_2_3 = train_test_split(dataset_2_3[0], dataset_2_3[1], test_size=0.2, random_state=42)

    # analyze_knn(dataset_2_2[0], dataset_2_2[1], dataset_2_2[2], dataset_2_2[3], "2_2")
    # analyze_knn(dataset_2_3[0], dataset_2_3[1], dataset_2_3[2], dataset_2_3[3], "2_3")

    # analyze_svm(dataset_2_2[0], dataset_2_2[1], dataset_2_2[2], dataset_2_2[3], "2_2")
    # analyze_svm(dataset_2_3[0], dataset_2_3[1], dataset_2_3[2], dataset_2_3[3], "2_3")

    # takes a few minutes to run for each dataset
    # analyze_svm(dataset_2_2[0], dataset_2_2[1], dataset_2_2[2], dataset_2_2[3], "2_2")
    analyze_mlp(dataset_2_3[0], dataset_2_3[1], dataset_2_3[2], dataset_2_3[3], "2_3")


experiment_two()
