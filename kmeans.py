from main import load_data
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score, \
    v_measure_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

from utils import plot_voronoi_diagram

n_clusters_max = 10

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


# analysis for experiment one
def analyze_kmeans_silhouette_score(X, y, dataset_name):
    scores = [0, 0]

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

    print(f'(Exp 1) {dataset_name} dataset; best silhouette score: {best_score} for n_clusters={best_score_idx}')
    print(f'(Exp 1) {dataset_name} dataset; worst silhouette score: {worst_score} for n_clusters={worst_score_idx}')

    plt.plot(np.arange(n_clusters_max), scores)
    plt.xlim([2, n_clusters_max - 1])
    plt.ylabel('silhouette score')
    plt.xlabel('n_clusters')
    plt.title(f'Silhouette score for {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_sil.png')
    # plt.close()

    if X.shape[1] == 2:
        kmeans = KMeans(n_clusters=best_score_idx, random_state=1, n_init="auto")
        plot_voronoi_diagram(X, None, kmeans.fit_predict(X), f'{dataset_name} dataset; best score; n_clusters={best_score_idx}')
        kmeans = KMeans(n_clusters=worst_score_idx, random_state=1, n_init="auto")
        plot_voronoi_diagram(X, None, kmeans.fit_predict(X), f'{dataset_name} dataset; worst score; n_clusters={worst_score_idx}')


def experiment_one_kmeans():
    # load Iris, Wine, Breast datasets from Scikit, and artificial datasets from CSV
    # dataset_iris = load_iris(return_X_y=True)
    # dataset_wine = load_wine(return_X_y=True)
    # dataset_breast_cancer = load_breast_cancer(return_X_y=True)
    dataset_1_1 = load_data("./data/1_1.csv")
    dataset_1_2 = load_data("./data/1_2.csv")
    dataset_1_3 = load_data("./data/1_3.csv")
    dataset_2_1 = load_data("./data/2_1.csv")
    dataset_2_2 = load_data("./data/2_2.csv")
    dataset_2_3 = load_data("./data/2_3.csv")

    # silhouette score for all datasets, voronoi for 2D
    # analyze_kmeans_silhouette_score(dataset_iris[0], dataset_iris[1], "iris")
    # analyze_kmeans_silhouette_score(dataset_wine[0], dataset_wine[1], "wine")
    # analyze_kmeans_silhouette_score(dataset_breast_cancer[0], dataset_breast_cancer[1], "breast cancer")
    analyze_kmeans_silhouette_score(dataset_1_1[0], dataset_1_1[1], "1_1")
    analyze_kmeans_silhouette_score(dataset_1_2[0], dataset_1_2[1], "1_2")
    analyze_kmeans_silhouette_score(dataset_1_3[0], dataset_1_3[1], "1_3")
    analyze_kmeans_silhouette_score(dataset_2_1[0], dataset_2_1[1], "2_1")
    analyze_kmeans_silhouette_score(dataset_2_2[0], dataset_2_2[1], "2_2")
    analyze_kmeans_silhouette_score(dataset_2_3[0], dataset_2_3[1], "2_3")


def analyze_kmeans_classifier(X, y, dataset_name):
    adj_rand_scores = [0, 0]
    homogeneity_scores = [0, 0]
    completeness_scores = [0, 0]
    v_measure_05_scores = [0, 0]
    v_measure_10_scores = [0, 0]
    v_measure_15_scores = [0, 0]

    X = StandardScaler().fit_transform(X)
    best_score = -1
    best_score_idx = -1
    worst_score = 1
    worst_score_idx = -1

    for n in range(2, n_clusters_max):
        kmeans = KMeans(n_clusters=n, random_state=1, n_init="auto")
        predictions = kmeans.fit_predict(X)

        # calculate all metrics
        adj_rand = adjusted_rand_score(y, predictions)
        homogeneity = homogeneity_score(y, predictions)
        completeness = completeness_score(y, predictions)
        v_measure_05 = v_measure_score(y, predictions, beta=0.5)
        v_measure_10 = v_measure_score(y, predictions, beta=1.0)
        v_measure_15 = v_measure_score(y, predictions, beta=1.5)

        # append to arrays
        adj_rand_scores.append(adj_rand)
        homogeneity_scores.append(homogeneity)
        completeness_scores.append(completeness)
        v_measure_05_scores.append(v_measure_05)
        v_measure_10_scores.append(v_measure_10)
        v_measure_15_scores.append(v_measure_15)

        # get average of all metrics
        avg_score = np.average([adj_rand, homogeneity, completeness, v_measure_05, v_measure_10, v_measure_15])

        if avg_score > best_score:
            best_score = avg_score
            best_score_idx = n

        if avg_score < worst_score:
            worst_score = avg_score
            worst_score_idx = n

    print(f'(Exp 2) {dataset_name} dataset; best average scores: {best_score} for n_clusters={best_score_idx}')
    print(f'(Exp 2) {dataset_name} dataset; worst average scores: {worst_score} for n_clusters={worst_score_idx}')

    plt.plot(np.arange(n_clusters_max), adj_rand_scores, label="adjusted rand score")
    plt.plot(np.arange(n_clusters_max), homogeneity_scores, label="homogeneity score")
    plt.plot(np.arange(n_clusters_max), completeness_scores, label="completeness score")
    plt.plot(np.arange(n_clusters_max), v_measure_05_scores, label="V-measure (beta=0.5)")
    plt.plot(np.arange(n_clusters_max), v_measure_10_scores, label="V-measure (beta=1.0)")
    plt.plot(np.arange(n_clusters_max), v_measure_15_scores, label="V-measure (beta=1.5)")
    plt.legend()
    plt.xlim([2, n_clusters_max - 1])
    plt.ylabel('scores')
    plt.xlabel('n_clusters')
    plt.title(f'Scores for {dataset_name} dataset')
    plt.show()
    # plt.savefig(f'{dataset_name}_scores.png')
    # plt.close()

    if X.shape[1] == 2:
        kmeans = KMeans(n_clusters=best_score_idx, random_state=1, n_init="auto")
        plot_voronoi_diagram(X, y, kmeans.fit_predict(X), f'{dataset_name} dataset; best score; n_clusters={best_score_idx}')
        kmeans = KMeans(n_clusters=worst_score_idx, random_state=1, n_init="auto")
        plot_voronoi_diagram(X, y, kmeans.fit_predict(X), f'{dataset_name} dataset; worst score; n_clusters={worst_score_idx}')


def experiment_two_kmeans():
    dataset_1_1 = load_data("./data/1_1.csv")
    dataset_1_2 = load_data("./data/1_2.csv")
    dataset_1_3 = load_data("./data/1_3.csv")
    dataset_2_1 = load_data("./data/2_1.csv")
    dataset_2_2 = load_data("./data/2_2.csv")
    dataset_2_3 = load_data("./data/2_3.csv")

    analyze_kmeans_classifier(dataset_1_1[0], dataset_1_1[1], "1_1")
    analyze_kmeans_classifier(dataset_1_2[0], dataset_1_2[1], "1_2")
    analyze_kmeans_classifier(dataset_1_3[0], dataset_1_3[1], "1_3")
    analyze_kmeans_classifier(dataset_2_1[0], dataset_2_1[1], "2_1")
    analyze_kmeans_classifier(dataset_2_2[0], dataset_2_2[1], "2_2")
    analyze_kmeans_classifier(dataset_2_3[0], dataset_2_3[1], "2_3")


print("Experiment one")
experiment_one_kmeans()
print("Experiment two")
experiment_two_kmeans()
