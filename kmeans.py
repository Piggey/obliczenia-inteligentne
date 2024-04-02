from main import load_data
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d


SCALE = 1.05
COLORS = ['red', 'green', 'blue', 'purple', 'orange', 'pink', 'cyan', 'olive', 'brown', 'gray']


def plot_voronoi_diagram(X, y_true, y_pred, subtitle=None):
    # Store x1 and x2 min and max values for plot scaling
    x1_max = X[:, 0].max()
    x2_max = X[:, 1].max()
    x1_min = X[:, 0].min()
    x2_min = X[:, 1].min()

    # Append dummy points to fix border cells not being colored
    X = np.append(X, [[999, 999], [-999, 999], [999, -999], [-999, -999]], axis=0)
    y_pred = np.append(y_pred, [0.0, 0.0, 0.0, 0.0])
    if y_true is not None:
        y_true = np.append(y_true, [0.0, 0.0, 0.0, 0.0])

    vor = Voronoi(X)
    voronoi_plot_2d(vor, show_vertices=False)

    unique_labels = np.unique(y_pred)

    # Color cells to show predicted labels
    for i in range(len(unique_labels)):
        region_indices = vor.point_region[np.where(y_pred == unique_labels[i])]
        for region_index in region_indices:
            vertices = vor.regions[region_index]
            if all(v >= 0 for v in vertices):
                plt.fill(vor.vertices[vertices][:, 0], vor.vertices[vertices][:, 1], color=COLORS[i], alpha=0.3)

    # Color dots to show true labels, or show black dots if y_true is None
    if y_true is not None:
        for i in range(len(X)):
            if y_true[i] == 0.0: plt.plot(X[i, 0], X[i, 1], COLORS[0], marker='o')
            if y_true[i] == 1.0: plt.plot(X[i, 0], X[i, 1], COLORS[1], marker='o')
            if y_true[i] == 2.0: plt.plot(X[i, 0], X[i, 1], COLORS[2], marker='o')
            if y_true[i] == 3.0: plt.plot(X[i, 0], X[i, 1], COLORS[3], marker='o')
            if y_true[i] == 4.0: plt.plot(X[i, 0], X[i, 1], COLORS[4], marker='o')
            if y_true[i] == 5.0: plt.plot(X[i, 0], X[i, 1], COLORS[5], marker='o')
            if y_true[i] == 6.0: plt.plot(X[i, 0], X[i, 1], COLORS[6], marker='o')
            if y_true[i] == 7.0: plt.plot(X[i, 0], X[i, 1], COLORS[7], marker='o')
            if y_true[i] == 8.0: plt.plot(X[i, 0], X[i, 1], COLORS[8], marker='o')
            if y_true[i] == 9.0: plt.plot(X[i, 0], X[i, 1], COLORS[9], marker='o')
    else:
        plt.plot(X[:, 0], X[:, 1], 'ko', markersize=3)

    if subtitle:
        plt.title(f'Voronoi ({subtitle})')
    else:
        plt.title(f'Voronoi')

    plt.xlabel('x1')
    plt.ylabel('x2')

    # Set correct scaling (to exclude dummy points)
    plt.xlim([x1_min * SCALE, x1_max * SCALE])
    plt.ylim([x2_min * SCALE, x2_max * SCALE])
    plt.show()


def plot_silhouette_score(X, y):
    scores = [0, 0]
    n_clusters_max = 10

    for n in range(2, n_clusters_max):
        kmeans = KMeans(n_clusters=n, random_state=1, n_init="auto")
        scores.append(silhouette_score(X, kmeans.fit_predict(X)))

    plt.plot(np.arange(n_clusters_max), scores)
    plt.xlim([2, n_clusters_max - 1])
    plt.ylabel('silhouette score')
    plt.xlabel('n_clusters')
    plt.show()


def analyze_kmeans_silhouette_score(X, y, dataset_name):
    scores = [0, 0]
    n_clusters_max = 10
    X = StandardScaler().fit_transform(X)

    best_score = -1
    best_score_idx = -1
    worst_score = 1
    worst_score_idx = -1

    for n in range(2, n_clusters_max):
        kmeans = KMeans(n_clusters=n, random_state=1, n_init="auto")
        score = silhouette_score(X, kmeans.fit_predict(X))
        scores.append(score)

        if score > best_score:
            best_score = score
            best_score_idx = n

        if score < worst_score:
            worst_score = score
            worst_score_idx = n

    print(f'{dataset_name} best silhouette score: {best_score} for n_clusters={best_score_idx}')
    print(f'{dataset_name} worst silhouette score: {worst_score} for n_clusters={worst_score_idx}')

    plt.plot(np.arange(n_clusters_max), scores)
    plt.xlim([2, n_clusters_max - 1])
    plt.ylabel('silhouette score')
    plt.xlabel('n_clusters')
    plt.title(f'For {dataset_name} dataset')
    plt.show()

    if X.shape[1] == 2:
        kmeans = KMeans(n_clusters=best_score_idx, random_state=1, n_init="auto")
        plot_voronoi_diagram(X, y, kmeans.fit_predict(X), f'{dataset_name} best score; n_clusters={best_score_idx}')
        kmeans = KMeans(n_clusters=worst_score_idx, random_state=1, n_init="auto")
        plot_voronoi_diagram(X, y, kmeans.fit_predict(X), f'{dataset_name} worst score; n_clusters={worst_score_idx}')

    return (best_score, best_score_idx), (worst_score, worst_score_idx)


def experiment_one_kmeans():
    # load Iris, Wine, Breast datasets from Scikit, and artificial datasets from CSV
    dataset_iris = load_iris(return_X_y=True)
    dataset_wine = load_wine(return_X_y=True)
    dataset_breast_cancer = load_breast_cancer(return_X_y=True)
    dataset_1_1 = load_data("./data/1_1.csv")
    dataset_1_2 = load_data("./data/1_2.csv")
    dataset_1_3 = load_data("./data/1_3.csv")
    dataset_2_1 = load_data("./data/2_1.csv")
    dataset_2_2 = load_data("./data/2_2.csv")
    dataset_2_3 = load_data("./data/2_3.csv")

    # silhouette score for all datasets, voronoi for 2D
    analyze_kmeans_silhouette_score(dataset_iris[0], dataset_iris[1], "iris")
    analyze_kmeans_silhouette_score(dataset_wine[0], dataset_wine[1], "wine")
    analyze_kmeans_silhouette_score(dataset_breast_cancer[0], dataset_breast_cancer[1], "breast cancer")
    analyze_kmeans_silhouette_score(dataset_1_1[0], dataset_1_1[1], "1_1")
    analyze_kmeans_silhouette_score(dataset_1_2[0], dataset_1_2[1], "1_2")
    analyze_kmeans_silhouette_score(dataset_1_3[0], dataset_1_3[1], "1_3")
    analyze_kmeans_silhouette_score(dataset_2_1[0], dataset_2_1[1], "2_1")
    analyze_kmeans_silhouette_score(dataset_2_2[0], dataset_2_2[1], "2_2")
    analyze_kmeans_silhouette_score(dataset_2_3[0], dataset_2_3[1], "2_3")


experiment_one_kmeans()
