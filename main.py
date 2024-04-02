import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler

from dbscan import dbs_calculate_silhouette_scores, dbs_plot_silhouette_scores, dbs_plot_voronoi


def load_data(path):
    df = pandas.read_csv(path, delimiter=';', header=0, names=['x', 'y', 'label'])

    X = StandardScaler().fit_transform(df[['x', 'y']])
    y_true = df['label'].to_numpy()

    return X, y_true


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
