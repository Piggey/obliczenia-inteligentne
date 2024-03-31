import numpy as np
import pandas
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d

from dbscan import dbs_calculate_silhouette_scores, dbs_plot_silhouette_scores, dbs_plot_voronoi

SCALE = 1.05
COLORS = ['red', 'green', 'blue', 'purple', 'orange', 'pink', 'cyan', 'olive', 'brown', 'gray']


def load_data(path):
    df = pandas.read_csv(path, delimiter=';', header=0, names=['x', 'y', 'label'])

    X = StandardScaler().fit_transform(df[['x', 'y']])
    y_true = df['label'].to_numpy()

    return X, y_true


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


DATASETS = [
  'data/1_1.csv',
  'data/1_2.csv',
  'data/1_3.csv',
  'data/2_1.csv',
  'data/2_2.csv',
  'data/2_3.csv',
]

EPS_VALUES = np.linspace(0.01, 1.99, 20)

if __name__ == '__main__':
  X, y = load_data(DATASETS[3])
  # plot_voronoi_diagram(X, y, y)
  # plot_silhouette_score(X, y)
  dbs_silhouette_scores, dbs_cluster_counts = dbs_calculate_silhouette_scores(X, EPS_VALUES)

  dbs_plot_silhouette_scores(EPS_VALUES, dbs_silhouette_scores, dbs_cluster_counts)
  dbs_plot_voronoi(X, EPS_VALUES, dbs_silhouette_scores)
