import logging

import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.neighbors import KernelDensity, NearestNeighbors

logger = logging.getLogger(__name__)


def compute_kde_density(zs):
    normalize_factor = np.mean(np.std(zs, axis=0))
    zs = zs / normalize_factor
    kde = KernelDensity(bandwidth=zs.shape[0] ** (-1 / (zs.shape[1] + 4))).fit(zs)
    log_density = kde.score_samples(zs)
    return np.exp(log_density)


def compute_density(zs, k=100, bandwidth=None, batch_size=1000):
    n, d = zs.shape
    normalize_factor = np.std(zs, axis=0)
    zs = zs / normalize_factor

    if bandwidth is None:
        bandwidth = n ** (-1 / (d + 4))

    # Compute KNN
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(zs)
    _, indices = nbrs.kneighbors(zs)

    # Remove self-indices
    indices = np.array([idx[idx != i][:k] for i, idx in enumerate(indices)])

    # Prepare output
    density = np.zeros(n)
    coef = 1 / np.sqrt((2 * np.pi) ** d)

    # Process in batches
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        batch_indices = indices[start:end]
        batch_points = zs[start:end]
        batch_neighbors = zs[batch_indices]

        # Compute normalized distances
        diffs = (batch_neighbors - batch_points[:, np.newaxis, :]) / bandwidth
        dists_square = np.sum(diffs**2, axis=2)

        # Gaussian density estimate
        density[start:end] = coef * np.exp(-0.5 * dists_square).sum(axis=1)

    return density, indices


def compute_trajectory(zs, density, start_idx, end_idx, k=100, knn_indices=None):
    if knn_indices is None:
        # compute KNN
        nbrs = NearestNeighbors(n_neighbors=k).fit(zs)
        _, indices = nbrs.kneighbors(zs)
    else:
        indices = knn_indices
        if knn_indices.shape[1] < k:
            raise Exception("knn_indices does not have enough neighbors for the specified k")
        elif knn_indices.shape[1] > k:
            logger.warning(f"Using only first {k} neighbors out of provided {knn_indices.shape[1]}")
            indices = indices[:, :k]

    # Build graph
    G = nx.Graph()
    for i in range(zs.shape[0]):
        distances = np.linalg.norm(zs[i] - zs[indices[i]], axis=1)
        for j, dist in zip(indices[i], distances):
            if i != j:
                G.add_edge(i, j, weight=2 * dist / (density[i] + density[j]))

    # Compute Shortest path
    if isinstance(start_idx, int) and isinstance(end_idx, int):
        path_indices = zs[nx.shortest_path(G, source=start_idx, target=end_idx, weight="weight")]
    else:
        path_indices = [
            zs[nx.shortest_path(G, source=start_idx[i], target=end_idx[i], weight="weight")]
            for i in range(len(start_idx))
        ]

    return path_indices


def pick_trajectory_pairs(centers, n_pairs):
    # from RECOVAR Implementation https://github.com/ma-gilles/recovar/blob/main/recovar/commands/analyze.py#L240
    pair_start = []
    pair_end = []
    X = distance_matrix(centers[:, :], centers[:, :])

    for _ in range(n_pairs // 2):
        i_idx, j_idx = np.unravel_index(np.argmax(X), X.shape)
        X[i_idx, :] = 0
        X[:, i_idx] = 0
        X[j_idx, :] = 0
        X[:, j_idx] = 0
        pair_start.append(centers[i_idx])
        pair_end.append(centers[j_idx])

    # Pick some pairs that are far in the first few principal components.
    zdim = centers.shape[-1]
    max_k = np.min([(n_pairs - n_pairs // 2), zdim])
    for k in range(max_k):
        i_idx = np.argmax(centers[:, k])
        j_idx = np.argmin(centers[:, k])
        pair_start.append(centers[i_idx])
        pair_end.append(centers[j_idx])

    return pair_start, pair_end


def find_closet_idx(data, point):
    dist = np.linalg.norm(data - point, axis=1)
    idx = np.argmin(dist)
    return data[idx], idx
