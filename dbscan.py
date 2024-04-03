import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from utils import load_data, plot_voronoi_diagram

def experiment_1(datasets):
    eps_values = np.linspace(0.1, 1.9, 20)

    for dataset in datasets:
        ds_name = dataset.split("/")[-1]
        print(f'dataset: {dataset}')

        X, y_true = load_data(dataset)

        silhouette_scores, cluster_counts, worst_eps, best_eps = _calculate_silhouette_scores(X, eps_values)
        print(f'best: {best_eps}, worst: {worst_eps}')

        _plot_silhouette_scores(f'{ds_name}_scores.png', eps_values, silhouette_scores, cluster_counts)

        y_pred_worst = DBSCAN(worst_eps, min_samples=1).fit_predict(X)
        y_pred_best = DBSCAN(best_eps, min_samples=1).fit_predict(X)
        plot_voronoi_diagram(X, y_true, y_pred_worst, f'{ds_name}_worst')
        plot_voronoi_diagram(X, y_true, y_pred_best, f'{ds_name}_best')


def _calculate_silhouette_scores(X, eps_values):
    silhouette_scores = []
    cluster_counts = []
    worst_eps = 100000
    best_eps = -1000000

    for eps in eps_values:
        dbscan = DBSCAN(eps, min_samples=1)
        y_pred = dbscan.fit_predict(X)

        clusters_count = len(np.unique(y_pred))
        cluster_counts.append(clusters_count)

        score = silhouette_score(X, y_pred) if clusters_count > 1 else 0
        silhouette_scores.append(score)

        if score == max(silhouette_scores):
            best_eps = eps
        if score == min(silhouette_scores):
            worst_eps = eps
        print(f'eps: {eps}, silhouette_score: {score}')

    return silhouette_scores, cluster_counts, worst_eps, best_eps


def _plot_silhouette_scores(name, eps_values, silhouette_scores, cluster_counts):
    plt.plot(eps_values, silhouette_scores)
    for eps, score, clusters in zip(eps_values, silhouette_scores, cluster_counts):
        plt.text(eps, score, str(clusters), fontsize=8, ha='right')

    plt.title('silhouette score')
    plt.ylabel('silhouette score')
    plt.xlabel('n_clusters')
    # plt.show()
    plt.savefig(name)
    plt.close()

