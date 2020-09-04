from sklearn.metrics import silhouette_score,calinski_harabasz_score
import numpy as np
from sklearn.utils import check_X_y
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import pairwise_distances_chunked
from sklearn.metrics.cluster.unsupervised import _silhouette_reduce, check_number_of_labels
import functools

def avg_silhouette(X, labels):
    return silhouette_score(X, labels, metric='euclidean')


def carlinski_harabasz_score(X, labels):
    return calinski_harabasz_score(X, labels)

def intra_cluster_distance(X, labels, *, metric='euclidean', **kwds):
    X, labels = check_X_y(X, labels, accept_sparse=['csc', 'csr'])

    # Check for non-zero diagonal entries in precomputed distance matrix
    if metric == 'precomputed':
        atol = np.finfo(X.dtype).eps * 100
        if np.any(np.abs(np.diagonal(X)) > atol):
            raise ValueError(
                'The precomputed distance matrix contains non-zero '
                'elements on the diagonal. Use np.fill_diagonal(X, 0).')

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples = len(labels)
    label_freqs = np.bincount(labels)
    check_number_of_labels(len(le.classes_), n_samples)

    kwds['metric'] = metric
    reduce_func = functools.partial(_silhouette_reduce,
                                    labels=labels, label_freqs=label_freqs)
    results = zip(*pairwise_distances_chunked(X, reduce_func=reduce_func,
                                              **kwds))
    intra_clust_dists, inter_clust_dists = results
    intra_clust_dists = np.concatenate(intra_clust_dists)
    inter_clust_dists = np.concatenate(inter_clust_dists)
    denom = (label_freqs - 1).take(labels, mode='clip')
    with np.errstate(divide="ignore", invalid="ignore"):
        intra_clust_dists /= denom
    return intra_clust_dists, inter_clust_dists