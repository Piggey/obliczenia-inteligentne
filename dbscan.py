import numpy as np
from sklearn.cluster import DBSCAN 
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def dbs_calculate_silhouette_scores(X, eps_values):
  silhouette_scores = []
  cluster_counts = []
  
  for eps in eps_values:
    dbscan = DBSCAN(eps=eps)
    labels = dbscan.fit_predict(X)

    unique_clusters = len(np.unique(labels))
    if unique_clusters > 1:
      silhouette = silhouette_score(X, labels)
    else:
      silhouette = 0

    silhouette_scores.append(silhouette)
    cluster_counts.append(unique_clusters)

  return silhouette_scores, cluster_counts

def dbs_plot_silhouette_scores(eps_values, silhouette_scores, cluster_counts):
  plt.figure(figsize=(12, 6))

  plt.plot(eps_values, silhouette_scores, marker='o', label='Silhouette Score')
  for eps, score, clusters in zip(eps_values, silhouette_scores, cluster_counts):
    plt.text(eps, score, str(clusters), fontsize=8, ha='right', va='bottom')

  plt.title('Silhouette Score with Cluster Counts')
  plt.xlabel('Eps')
  plt.ylabel('Silhouette Score')
  plt.grid(True)
  plt.legend()

  plt.show()

def dbs_plot_voronoi(X, eps_values, silhouette_scores):
  best_eps_index = np.argmax(silhouette_scores)
  worst_eps_index = np.argmin(silhouette_scores)

  # Wizualizacja klastrów dla najlepszego przypadku
  _, axs = plt.subplots(1, 2, figsize=(15, 5))
  _plot_voronoi(X, DBSCAN(eps=eps_values[best_eps_index]).fit_predict(X), ax=axs[0])
  axs[0].set_title(f'Best Case - Eps: {eps_values[best_eps_index]}')
  
  # Wizualizacja klastrów dla najgorszego przypadku
  _plot_voronoi(X, DBSCAN(eps=eps_values[worst_eps_index]).fit_predict(X), ax=axs[1])
  axs[1].set_title(f'Worst Case - Eps: {eps_values[worst_eps_index]}')
  
  plt.show()

def _plot_voronoi(X, labels, ax):
  vor = Voronoi(X)
  voronoi_plot_2d(vor, show_vertices=False, ax=ax)
  for label in np.unique(labels):
    ax.scatter(X[labels == label][:, 0], X[labels == label][:, 1])
