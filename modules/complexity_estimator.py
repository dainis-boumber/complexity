import numpy as np
import scipy
import random

class ComplexityEstimator:

    def __init__(self, X, y, n_windows=10):
        assert (n_windows > 0)
        self.X = X
        self.y = y
        self.seeds = np.random.random_integers(0, len(X) - 1, n_windows)
        self.tree = scipy.spatial.cKDTree(X)
        self.labels = set(y)
        self.Ks = np.arange(1, len(self.X) + 1)  # ckdTree starts counting from 1
        self.Hs = np.zeros(len(self.Ks))
        self.ws = np.ndarray((n_windows, len(self.Ks)))

        for i, k in enumerate(self.Ks):
            for j, seed in enumerate(self.seeds):
                h = self._H(k=k, seed=seed)
                self.ws[j, k-1] = h
                self.Hs[i] = np.sum(self.ws[:, k-1]) / len(self.seeds)

    def get_k_complexity(self):
        return self.Ks, self.Hs

    def get_w_complexity(self):
        return self.ws

    def get_seed(self, window):
        return self.seeds(window)

    def _nearest_neighbors(self, k, seed):
        return self.tree.query(self.X[seed, :], k=k)

    def _H(self, k, seed):
        H = 0
        d, ii = self._nearest_neighbors(k, seed)
        for c in self.labels:
            r = len(np.extract(self.y[ii] == c, self.y[ii]))/k
            if r > 0:
                H += (r * np.log2(r))
        return -H