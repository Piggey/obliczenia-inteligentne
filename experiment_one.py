from main import load_data, plot_voronoi_diagram
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


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
