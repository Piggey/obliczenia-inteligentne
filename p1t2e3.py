from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from utils import load_data, test_train_line_plot
from sklearn.svm import SVC
import numpy as np

TRAIN_SIZE = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42

KNN_N_NEIGHBORS_RANGE = range(1, 15)
SVM_C_RANGE = np.arange(1e-2, 1e6, 1e4)

def evaluate_svm(X_train, X_test, y_train, y_test, *, kernel):
    acc_scores_train = []
    acc_scores_test = []

    for c in SVM_C_RANGE:
        svm = SVC(C=c, kernel=kernel)
        svm.fit(X_train, y_train)

        y_train_pred = svm.predict(X_train)
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_scores_train.append(acc_train)

        y_test_pred = svm.predict(X_test)
        acc_test = accuracy_score(y_test, y_test_pred)
        acc_scores_test.append(acc_test)

        print(f'{c=}, {acc_test=}, {acc_train=}')

    return acc_scores_train, acc_scores_test


def evaluate_knn(X_train, X_test, y_train, y_test):
    acc_scores_train = []
    acc_scores_test = []

    for n_neighbors in KNN_N_NEIGHBORS_RANGE:
        knn = KNeighborsClassifier(n_neighbors)
        knn.fit(X_train, y_train)

        y_train_pred = knn.predict(X_train)
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_scores_train.append(acc_train)

        y_test_pred = knn.predict(X_test)
        acc_test = accuracy_score(y_test, y_test_pred)
        acc_scores_test.append(acc_test)

        print(f'{n_neighbors=}, {acc_test=}, {acc_train=}')

    return acc_scores_train, acc_scores_test

def experiment_3():
    X, y_true = load_data('data/2_2.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, 
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )


    ## K-NN
    # acc_train, acc_test = evaluate_knn(X_train, X_test, y_train, y_test)
    # test_train_line_plot(
    #     KNN_N_NEIGHBORS_RANGE,
    #     acc_train,
    #     acc_test,
    #     'Dokładność klasyfikatora K-NN dla różnych wartości n_neighbors',
    #     'Liczba sąsiadów (n_neighbors)',
    #     'Dokładność (accuracy_score)',
    # )

    # SVM
    acc_train, acc_test = evaluate_svm(X_train, X_test, y_train, y_test, kernel='rbf')
    test_train_line_plot(
        SVM_C_RANGE,
        acc_train,
        acc_test,
        'SVM dla roznych wartosci C',
        'C',
        'Dokładność (accuracy_score)',
    )

experiment_3()
