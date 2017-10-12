# -*- encoding: utf-8 -*-

import numpy as np
from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import NearestNeighbors


def binarize(target):
    median = np.median(target, axis=1)[:, None]
    binary = np.zeros(shape=np.shape(target))
    binary[target > median] = 1
    return binary


def affinity_matrix(graph):
    idx_aff = (graph.T != 0)
    aff = graph.copy()
    aff[idx_aff] = graph.T[idx_aff]
    return aff


def heat_kernel_matrix(aff):
    idx_nnull = (aff != 0)
    aff[idx_nnull] = np.exp(-aff[idx_nnull]/2)
    heat = aff + np.identity(np.shape(aff)[0])
    return heat


def laplacian_eigenmaps(lsa_features, n_neighbors=15, subdim=15, n_jobs=1):

    print("Fitting nearest neighbors")
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs)
    nn.fit(lsa_features)
    graph = nn.kneighbors_graph(mode="distance").toarray()

    print("Creation of heat kernel affinity matrix")
    aff = affinity_matrix(graph)
    heat = heat_kernel_matrix(aff)

    print("Spectral embedding")
    spec_emb = SpectralEmbedding(n_components=subdim, affinity="precomputed", n_jobs=n_jobs)
    eigenvectors = spec_emb.fit_transform(heat)
    return eigenvectors
