import numpy as np
import scipy
import random

class ComplexityEstimator:

    def __init__(self, X, y, n_windows=10):
        assert (n_windows > 0)
        self.X = X
        self.y = y
        self.seeds = random.sample(range(0, len(X) - 1), n_windows)
        self.tree = scipy.spatial.cKDTree(X)
        self.labels = set(y)

        Ks = np.arange(1, len(self.X) + 1)  # ckdTree starts counting from 1
        ret = []

        for k in Ks:
            entropies = [self._H(k=k, seed=seed) for seed in self.seeds]
            h = np.sum(entropies) / np.float32(len(self.seeds))
            ret.append([k, h, [self.seeds, entropies]])

        self.coomplexity = ret

    def get_k_complexity(self):
        return self.coomplexity[0], self.coomplexity[1]

    def get_s_complexity(self):
        return self.coomplexity[2][0], self.coomplexity[2][1]

    def _nearest_neighbors(self, k, seed):
        return self.tree.query(self.X[seed, :], k=k)

    def _H(self, k, seed):
        H = 0
        d, ii = self._nearest_neighbors(k, seed)
        for c in self.labels:
            tmp = self.y[ii]
            r = np.float32(len(np.where( tmp == c )))/np.float32(k)
            if r > 0:
                H += (r * np.log2(r))
        return -H