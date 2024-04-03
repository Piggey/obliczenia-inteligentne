import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score, silhouette_score, v_measure_score
import matplotlib.pyplot as plt
from utils import load_data, plot_voronoi_diagram

EPS_VALUES = np.linspace(0.1, 1.9, 20)

def experiment_1(datasets):
    for dataset in datasets:
        ds_name = dataset.split("/")[-1]
        print(f'dataset: {dataset}')

        X, y_true = load_data(dataset)

        silhouette_scores, cluster_counts, worst_eps, best_eps = _calculate_silhouette_scores(X, EPS_VALUES)
        print(f'best: {best_eps}, worst: {worst_eps}')

        _plot_silhouette_scores(f'{ds_name}_scores.png', EPS_VALUES, silhouette_scores, cluster_counts)

        y_pred_worst = DBSCAN(worst_eps, min_samples=1).fit_predict(X)
        y_pred_best = DBSCAN(best_eps, min_samples=1).fit_predict(X)
        plot_voronoi_diagram(X, y_true, y_pred_worst, f'{ds_name}_worst_eps:{worst_eps:.2f}')
        plot_voronoi_diagram(X, y_true, y_pred_best, f'{ds_name}_best_eps:{best_eps:.2f}')

def experiment_2(datasets):
    for dataset in datasets:
        ds_name = dataset.split('/')[-1]

        adjusted_rand_scores = []
        homogeneity_scores = []
        completeness_scores = []
        v_measure_scores_05 = []
        v_measure_scores_10 = []
        v_measure_scores_20 = []

        worst_eps = 100000
        best_eps = -1000000
        avgs = []
        cluster_counts = []

        X, y_true = load_data(dataset)

        for eps in EPS_VALUES:
            dbscan = DBSCAN(eps, min_samples=1)
            y_pred = dbscan.fit_predict(X)

            clusters_count = len(np.unique(y_pred))
            cluster_counts.append(clusters_count)
            
            ars = adjusted_rand_score(y_true, y_pred)
            hgs = homogeneity_score(y_true, y_pred)
            cs = completeness_score(y_true, y_pred)
            vm05 = v_measure_score(y_true, y_pred, beta=0.5)
            vm10 = v_measure_score(y_true, y_pred, beta=1.0)
            vm20 = v_measure_score(y_true, y_pred, beta=2.0)
            print(f'ars: {ars}, hgs: {hgs}, cs: {cs}, vm(0.5, 1.0, 2.0): ({vm05}, {vm10}, {vm20})')

            avg = np.average([ars, hgs, cs, vm05, vm10, vm20])
            avgs.append(avg)
            if avg == max(avgs):
                best_eps = eps
            if avg == min(avgs):
                worst_eps = eps
            print(f'avg: {avg}')

            adjusted_rand_scores.append(ars)
            homogeneity_scores.append(hgs)
            completeness_scores.append(cs)
            v_measure_scores_05.append(vm05)
            v_measure_scores_10.append(vm10)
            v_measure_scores_20.append(vm20)

        print(f'best_eps: {best_eps}, worst_eps: {worst_eps}')
        _plot_classifier_scores(f'{ds_name}_scores.png', EPS_VALUES, cluster_counts, adjusted_rand_scores, homogeneity_scores, completeness_scores, v_measure_scores_05, v_measure_scores_10, v_measure_scores_20)

        y_pred_worst = DBSCAN(worst_eps, min_samples=1).fit_predict(X)
        y_pred_best = DBSCAN(best_eps, min_samples=1).fit_predict(X)
        plot_voronoi_diagram(X, y_true, y_pred_worst, f'{ds_name}_worst_eps:{worst_eps:.2f}')
        plot_voronoi_diagram(X, y_true, y_pred_best, f'{ds_name}_best_eps:{best_eps:.2f}')


def _plot_classifier_scores(name, eps_values, cluster_counts, adjusted_rand_scores, homogeneity_scores, completeness_scores, v_measure_scores_05, v_measure_scores_10, v_measure_scores_20):
    plt.plot(eps_values, adjusted_rand_scores, label='Adjusted Rand Score')
    plt.plot(eps_values, homogeneity_scores, label='Homogeneity Score')
    plt.plot(eps_values, completeness_scores, label='Completeness Score')
    plt.plot(eps_values, v_measure_scores_05, label='V Measure (beta=0.5)')
    plt.plot(eps_values, v_measure_scores_10, label='V Measure (beta=1.0)')
    plt.plot(eps_values, v_measure_scores_20, label='V Measure (beta=2.0)')

    for eps, score, count in zip(eps_values, adjusted_rand_scores, cluster_counts):
        plt.text(eps, score, str(count), fontsize=12, ha='right')

    plt.title('Classifier Scores')
    plt.ylabel('Scores')
    plt.xlabel('Eps')
    plt.legend()
    # plt.show()
    plt.savefig(name)
    plt.close()

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
    plt.xlabel('eps')
    # plt.show()
    plt.savefig(name)
    plt.close()

