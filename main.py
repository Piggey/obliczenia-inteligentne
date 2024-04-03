import numpy as np
import dbscan

DATASETS = [
  'data/1_1.csv',
  # 'data/1_2.csv',
  # 'data/1_3.csv',
  # 'data/2_1.csv',
  # 'data/2_2.csv',
  # 'data/2_3.csv',
]

if __name__ == '__main__':
  # X, y = load_data(DATASETS[3])
  # plot_voronoi_diagram(X, y, y)
  # plot_silhouette_score(X, y)
  # dbscan.experiment_1(DATASETS)
  dbscan.experiment_2(DATASETS)
