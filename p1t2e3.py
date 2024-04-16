from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from utils import load_data, plot_accuracy_scores, plot_decision_boundary, plot_confusion_matrix
from sklearn.svm import SVC
import numpy as np

TRAIN_SIZE = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42

KNN_N_NEIGHBORS_RANGE = range(1, 15)
SVM_C_RANGE = np.logspace(-2, 6, num=32, endpoint=True)
MLP_HIDDEN_LAYERS_RANGE = np.arange(2, 65, 4)

def experiment_3():
    dataset_name = 'data/2_3.csv'
    X, y_true = load_data(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, 
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # analyze_knn(KNN_N_NEIGHBORS_RANGE, X_train, X_test, y_train, y_test, dataset_name)
    # analyze_svm(SVM_C_RANGE, X_train, X_test, y_train, y_test, dataset_name, kernel='rbf')
    analyze_mlp(MLP_HIDDEN_LAYERS_RANGE, X_train, X_test, y_train, y_test, dataset_name, random_state=RANDOM_STATE)

def analyze_knn(n_neighbors_range, X_train, X_test, y_train, y_test, dataset_name):
    best_n_neighbors = -1
    best_accuracy = -1

    scores = []
    for n_neighbors in n_neighbors_range:
        knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_classifier.fit(X_train, y_train)

        # test over training data
        y_train_pred = knn_classifier.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_train_pred)

        # test over testing data
        y_test_pred = knn_classifier.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_test_pred)

        scores.append([n_neighbors, accuracy_test, accuracy_train])

        print(f'{n_neighbors=}, {accuracy_train=}, {accuracy_test=}')

        if best_accuracy < accuracy_test:
            best_accuracy = accuracy_test
            best_n_neighbors = n_neighbors

    print(f'{dataset_name=}, {best_accuracy=}, {best_n_neighbors=}')

    plot_accuracy_scores(scores, title='KNN', x_label='n_neighbors')

    # smallest
    knn_classifier = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
    plot_decision_boundary(knn_classifier, X_test, y_test, title='KNN: decision boundary, n_neighbors=1')
    plot_confusion_matrix(y_test, knn_classifier.predict(X_test), knn_classifier.classes_, title='KNN: confusion matrix, n_neighbors=1')

    # largest
    knn_classifier = KNeighborsClassifier(n_neighbors=14).fit(X_train, y_train)
    plot_decision_boundary(knn_classifier, X_test, y_test, title='KNN: decision boundary, n_neighbors=14')
    plot_confusion_matrix(y_test, knn_classifier.predict(X_test), knn_classifier.classes_, title='KNN: confusion matrix, n_neighbors=14')

    # best
    knn_classifier = KNeighborsClassifier(n_neighbors=best_n_neighbors).fit(X_train, y_train)
    plot_decision_boundary(knn_classifier, X_test, y_test, title=f'KNN: decision boundary, n_neighbors={best_n_neighbors}')
    plot_confusion_matrix(y_test, knn_classifier.predict(X_test), knn_classifier.classes_, title=f'KNN: confusion matrix, n_neighbors={best_n_neighbors}')

def analyze_svm(c_range, X_train, X_test, y_train, y_test, dataset_name, *, kernel='rbf'):
    best_C = -1
    best_accuracy = -1

    scores = []
    for C in c_range:
        svc_classifier = SVC(kernel=kernel, C=C)
        svc_classifier.fit(X_train, y_train)

        # test over training data
        y_train_pred = svc_classifier.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_train_pred)

        # test over testing data
        y_test_pred = svc_classifier.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_test_pred)

        scores.append([C, accuracy_test, accuracy_train])

        print(f'{np.log10(C)=}, {accuracy_train=}, {accuracy_test=}')

        if best_accuracy < accuracy_test:
            best_accuracy = accuracy_test
            best_C = C

    plot_accuracy_scores(scores, title='SVM', x_label='log(C)')
    print(f'{dataset_name=}, {best_accuracy=}, {np.log10(best_C)=}')

    # smallest
    svc_classifier = SVC(kernel=kernel, C=c_range[0]).fit(X_train, y_train)
    plot_decision_boundary(svc_classifier, X_test, y_test, title=f'SVM: decision boundary, C={np.log10(c_range[0])}')
    plot_confusion_matrix(y_test, svc_classifier.predict(X_test), labels=svc_classifier.classes_, title=f'SVM: confusion matrix, C={np.log10(c_range[0])}')

    # largest
    svc_classifier = SVC(kernel=kernel, C=c_range[-1]).fit(X_train, y_train)
    plot_decision_boundary(svc_classifier, X_test, y_test, title=f'SVM: decision boundary, C={np.log10(c_range[-1])}')
    plot_confusion_matrix(y_test, svc_classifier.predict(X_test), labels=svc_classifier.classes_, title=f'SVM: confusion matrix, C={np.log10(c_range[-1])}')

    # best
    svc_classifier = SVC(kernel=kernel, C=best_C).fit(X_train, y_train)
    plot_decision_boundary(svc_classifier, X_test, y_test, title=f'SVM: decision boundary, C={np.log10(best_C)}')
    plot_confusion_matrix(y_test, svc_classifier.predict(X_test), labels=svc_classifier.classes_, title=f'SVM: confusion matrix, C={np.log10(best_C)}')

def analyze_mlp(hidden_layers_range, X_train, X_test, y_train, y_test, dataset_name, *, solver='sgd', activation='relu', random_state=42):
    best_hls = -1
    best_accuracy = -1

    scores = []
    for hls in hidden_layers_range:
        mlp_classifier = MLPClassifier(
            solver=solver,
            activation=activation,
            random_state=random_state,
            max_iter=100_000,
            n_iter_no_change=100_000,
            tol=0,
            hidden_layer_sizes=hls
        )
        mlp_classifier.fit(X_train, y_train)

        # test over training data
        y_train_pred = mlp_classifier.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_train_pred)

        # test over testing data
        y_test_pred = mlp_classifier.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_test_pred)

        scores.append([hls, accuracy_test, accuracy_train])

        print(f'{hls=}, {accuracy_train=}, {accuracy_test=}')

        if best_accuracy < accuracy_test:
            best_accuracy = accuracy_test
            best_hls = hls

    plot_accuracy_scores(scores, title='MLP', x_label='hidden_layer_sizes')
    print(f'{dataset_name=}, {best_accuracy=}, {best_hls=}')

    # smallest
    mlp_classifier = MLPClassifier(
        solver=solver,
        activation=activation,
        random_state=random_state,
        max_iter=100_000,
        n_iter_no_change=100_000,
        tol=0,
        hidden_layer_sizes=hidden_layers_range[0]
    ).fit(X_train, y_train)
    plot_decision_boundary(mlp_classifier, X_test, y_test, title=f'MLP: decision boundary, hidden_layer_sizes={hidden_layers_range[0]}')
    plot_confusion_matrix(y_test, mlp_classifier.predict(X_test), mlp_classifier.classes_, title=f'MLP: confusion matrix, hidden_layer_sizes={hidden_layers_range[0]}')

    # largest
    mlp_classifier = MLPClassifier(
        solver=solver,
        activation=activation,
        random_state=random_state,
        max_iter=100_000,
        n_iter_no_change=100_000,
        tol=0,
        hidden_layer_sizes=hidden_layers_range[-1]
    ).fit(X_train, y_train)
    plot_decision_boundary(mlp_classifier, X_test, y_test, title=f'MLP: decision boundary, hidden_layer_sizes={hidden_layers_range[-1]}')
    plot_confusion_matrix(y_test, mlp_classifier.predict(X_test), mlp_classifier.classes_, title=f'MLP: confusion matrix, hidden_layer_sizes={hidden_layers_range[-1]}')

    # best
    mlp_classifier = MLPClassifier(
        solver=solver,
        activation=activation,
        random_state=random_state,
        max_iter=100_000,
        n_iter_no_change=100_000,
        tol=0,
        hidden_layer_sizes=best_hls,
    ).fit(X_train, y_train)
    plot_decision_boundary(mlp_classifier, X_test, y_test, title=f'MLP: decision boundary, hidden_layer_sizes={best_hls}')
    plot_confusion_matrix(y_test, mlp_classifier.predict(X_test), mlp_classifier.classes_, title=f'MLP: confusion matrix, hidden_layer_sizes={best_hls}')


experiment_3()
