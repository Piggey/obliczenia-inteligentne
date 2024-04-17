import numpy as np

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from utils import load_data, plot_accuracy_scores, plot_decision_boundary

TEST_SIZE = 0.2
RANDOM_STATE = 42

MLP_HIDDEN_LAYERS_RANGE = np.arange(2, 65, 4)

def experiment_4():
    dataset_name = 'data/2_3.csv'
    X, y_true = load_data(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, 
        train_size=0.8,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # analyze_mlp(X_train, X_test, y_train, y_test, dataset_name, train_size=0.8)

    test_scores, train_scores = analyze_mlp_random(X_train, X_test, y_train, y_test)
    print('train_size=0.8')
    print(f'{test_scores=}')
    print(f'{train_scores=}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, 
        train_size=0.2,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    analyze_mlp(X_train, X_test, y_train, y_test, dataset_name, train_size=0.2)

    test_scores, train_scores = analyze_mlp_random(X_train, X_test, y_train, y_test)
    print('train_size=0.2')
    print(f'{test_scores=}')
    print(f'{train_scores=}')

def analyze_mlp(X_train, X_test, y_train, y_test, dataset_name, train_size, *, hidden_layer_sizes=34, activation='relu', random_state=42):
    # dataset 2_3
    # MLP
    # train_size: oba przypadki: 0.8 i 0.2
    # wizualizacja zmian dla kolejnych epok
    # wizualizacje przebiegu granicy decyzyjnej na zbiorach treningowym i testowym dla epoki: zerowej, najlepszej i ostatniej
    # dla kazdego z przypadkow:
    # Liczbę neuronów w warstwie ukrytej należy dobrać jako tą optymalną wynikającą odpowiednio z eksperymentów drugiego i trzeciego
    best_epoch = -1
    best_accuracy = -1

    mlp = MLPClassifier(
        solver='sgd',
        activation=activation,
        random_state=random_state,
        max_iter=100_000,
        n_iter_no_change=100_000,
        tol=0,
        hidden_layer_sizes=hidden_layer_sizes,
    )

    scores = []
    for epoch in range(1, 100_000):
        mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))

        y_train_pred = mlp.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_train_pred)

        y_test_pred = mlp.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_test_pred)

        scores.append([epoch, accuracy_test, accuracy_train])

        if epoch % 1000 == 0:
            print(f'{epoch=}, {accuracy_train=}, {accuracy_test=}')

        if epoch == 1:
            plot_decision_boundary(mlp, X_train, y_train, title=f'train: decision boundary, {epoch=}, {train_size=}, {accuracy_test=:.3f}', show=False, export_filename=f'{dataset_name}-mlp-epoch-decision-boundary-{train_size=}-train-smallest.png')
            plot_decision_boundary(mlp, X_test, y_test, title=f'test: decision boundary, {epoch=}, {train_size=}, {accuracy_test=:.3f}', show=False, export_filename=f'{dataset_name}-mlp-epoch-decision-boundary-{train_size=}-{train_size=}-{train_size=}-{train_size=}-{train_size=}-{train_size=}-{train_size=}-{train_size=}-test-smallest.png')

        if best_accuracy < accuracy_test:
            best_accuracy = accuracy_test
            best_epoch = epoch
            plot_decision_boundary(mlp, X_train, y_train, title=f'train: decision boundary, {epoch=}, {train_size=}, {accuracy_test=:.3f}', show=False, export_filename=f'{dataset_name}-mlp-epoch-decision-boundary-{train_size=}-train-best.png')
            plot_decision_boundary(mlp, X_test, y_test, title=f'test: decision boundary, {epoch=}, {train_size=}, {accuracy_test=:.3f}', show=False, export_filename=f'{dataset_name}-mlp-epoch-decision-boundary-{train_size=}-test-best.png')


    print(f'{best_epoch=}, {best_accuracy=}')
    plot_accuracy_scores(scores, title=f'accuracy scores, {train_size=}', x_label='epoch', show_grid=False, show=False, export_filename=f'{dataset_name}-mlp-epoch-accuracy-scores.png')
    plot_decision_boundary(mlp, X_train, y_train, title=f'train: decision boundary, {epoch=}, {train_size=}, {accuracy_test=:.3f}', show=False, export_filename=f'{dataset_name}-mlp-epoch-decision-boundary-{train_size=}-train-biggest.png')
    plot_decision_boundary(mlp, X_test, y_test, title=f'test: decision boundary, {epoch=}, {train_size=}, {accuracy_test=:.3f}', show=False, export_filename=f'{dataset_name}-mlp-epoch-decision-boundary-{train_size=}-test-biggest.png')

def analyze_mlp_random(X_train, X_test, y_train, y_test, *, hidden_layer_sizes=34, activation='relu'):
    # - uruchomić proces treningu 10 razy z różnymi wagami początkowymi
    # - w tabeli zamieścić wartości accuracy na zbiorze testowym i treningowym dla epoki: pierwszej, najlepszej i ostatniej
    # - dla najlepszej podac tez numer epoki
    test_scores = []
    train_scores = []

    for i in range(10):
        print(f'{i=}/10')
        best_epoch = -1

        accuracy_test_first = -1
        accuracy_train_first = -1
        
        accuracy_test_best = -1
        accuracy_train_best = -1
        
        accuracy_test_last = -1
        accuracy_train_last = -1

        mlp = MLPClassifier(
            solver='sgd',
            activation=activation,
            random_state=None,
            max_iter=100_000,
            n_iter_no_change=100_000,
            tol=0,
            hidden_layer_sizes=hidden_layer_sizes,
        )

        for epoch in range(1, 100_000):
            mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))

            y_train_pred = mlp.predict(X_train)
            accuracy_train = accuracy_score(y_train, y_train_pred)

            y_test_pred = mlp.predict(X_test)
            accuracy_test = accuracy_score(y_test, y_test_pred)

            if epoch % 1000 == 0:
                print(f'{epoch=}, {accuracy_train=}, {accuracy_test=}')

            if epoch == 1:
                accuracy_train_first = accuracy_train
                accuracy_test_first = accuracy_test

            if accuracy_test_best < accuracy_test:
                best_epoch = epoch
                accuracy_train_best = accuracy_train
                accuracy_test_best = accuracy_test

            if epoch == 100_000 - 1:
                accuracy_train_last = accuracy_train
                accuracy_test_last = accuracy_test
        
        # # TEST
        # [_
        #     [best_epoch, acc_first, acc_best, acc_last]
        # ]
        test_scores.append([best_epoch, accuracy_test_first, accuracy_test_best, accuracy_test_last])
        train_scores.append([best_epoch, accuracy_train_first, accuracy_train_best, accuracy_train_last])

    return test_scores, train_scores


experiment_4()