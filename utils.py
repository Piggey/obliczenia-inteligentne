from matplotlib import pyplot as plt
import numpy as np
import pandas
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pandas.read_csv(path, delimiter=';', header=0, names=['x', 'y', 'label'])

    X = StandardScaler().fit_transform(df[['x', 'y']])
    y_true = df['label'].to_numpy()

    return X, y_true

def plot_voronoi_diagram(X, y_true, y_pred, subtitle=None):
    SCALE = 1.05
    COLORS = ['red', 'green', 'blue', 'purple', 'orange', 'pink', 'cyan', 'olive', 'brown', 'gray']

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
                plt.fill(vor.vertices[vertices][:, 0], vor.vertices[vertices][:, 1], color=COLORS[i % len(COLORS)], alpha=0.3)

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
    # plt.show()
    plt.savefig(f'{subtitle}_vor.png')
    plt.close()
