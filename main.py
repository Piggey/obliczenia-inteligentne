import numpy as np
import dbscan
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

DATASETS = [
  'data/1_1.csv',
  'data/1_2.csv',
  'data/1_3.csv',
  'data/2_1.csv',
  'data/2_2.csv',
  'data/2_3.csv',
]

if __name__ == '__main__':
  # X, y = load_data(DATASETS[3])
  # plot_voronoi_diagram(X, y, y)
  # plot_silhouette_score(X, y)
  # dbscan.experiment_1(DATASETS)
  # dbscan.experiment_2(DATASETS)

  # X, y_true = load_iris(return_X_y=True)
  # dbscan.experiment_1_rd('ex1/iris', X, y_true, np.linspace(0.1, 1.9, 20))
  # dbscan.experiment_2_rd('ex2/iris', X, y_true, np.linspace(0.1, 1.9, 20))

  # X, y_true = load_wine(return_X_y=True)
  # dbscan.experiment_1_rd('ex1/wine', X, y_true, np.linspace(30, 140, 20))
  # dbscan.experiment_2_rd('ex2/wine', X, y_true, np.linspace(30, 140, 20))

  X, y_true = load_breast_cancer(return_X_y=True)
  # dbscan.experiment_1_rd('ex1/breast_cancer', X, y_true, np.linspace(500, 1200, 30))
  dbscan.experiment_2_rd('ex2/breast_cancer', X, y_true, np.linspace(100, 500, 20))
