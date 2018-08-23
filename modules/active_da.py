# Necessities
import numpy as np
import random
import scipy.spatial
import sklearn.metrics as metr
from numpy.core.multiarray import ndarray

import modules.util as u
from modules.oracle import Oracle

#untested but should work
# example usage (pseudocodish):
# cda = CADA(MNIST.X, MNIST.y)
# ixs = query(MNIST_M.X, 100)
# X_train, y_train = MNIST_M.X[ixs], MNIST_M.y[ixs]
# classifier.fit(X_train, y_train)
# classifier.predict(MNIST_M.X[-ixs])

class CADA(object):
    '''
    1.  Compute complexity measure on source domain at different levels of locality (size of neighborhood).
    2. Choose the single neighborhood size K* that keeps entropy below a predefined threshold.
    3. Sample N examples from target domain randomly.
    4. Grow a window around each example of size K*.
    5. All examples within each window are banned from sampling.
    6. Go to step 3 until no more examples are left for sampling.
    7. Create a model with the queried examples on target. (outside the scope of this class, can use any model you want)
    '''
    def __init__(self, source_X, source_y, max_entropy=0.1, f_samples=0.01, window_growth_rate=0.1, mink=1, maxk=-1, n_samples=-1):
        assert(len(source_X)==len(source_y))
        assert(f_samples <= 1.0 and max_entropy <= 1.0)
        assert (n_samples < len(source_y))
        assert(mink > 0 and maxk < len(source_y))
        #all sane

        #how many do we actually sample in Step 3
        self.source_X = source_X
        self.source_y = source_y
        #build the tree of the distribution that we do nn on
        self.tree = scipy.spatial.cKDTree(self.source_X, leafsize=32, compact_nodes=False, balanced_tree=False)
        print('build done')
        if n_samples == -1:
            self.n_samples = int(len(source_y) * f_samples)
            if self.n_samples == 0:
                self.n_samples = 1
        else:
            self.n_samples = n_samples
        #init sampling
        self.seeds = np.random.random_integers(0, len(source_X) - 1, self.n_samples)
        self.classes = set(source_y)
        stepsize = int(maxk * window_growth_rate)
        if stepsize == 0:
            stepsize = 1
        self.Ks = np.arange(mink, maxk, step=stepsize)  # ckdTree starts counting from 1
        self.Hs = np.zeros(len(self.Ks))
        self.ws = np.zeros((len(self.seeds), len(self.Ks)))
        self.K = 0

        for i, k in enumerate(self.Ks):
            for j, seed in enumerate(self.seeds):
                self.ws[j, i] = self._H(k=k, seed=seed)

            # add up entropy for each window as they grow
            self.Hs[i] = np.sum(self.ws[:, i]) / len(self.seeds)

            if self.Hs[i] > max_entropy:
                if i > 0:
                    self.K = self.Ks[i-1]
                break # done with step 2

    # returns indices into target_X

    def query(self, target_X, N):
        if N > len(target_X):
            raise AttributeError

        target_banned = np.zeros(len(target_X))
        queried_example_indices = []
        while 0 in target_banned: # step 6 check
            not_banned_ix = [i for i, banned in enumerate(target_banned) if banned != 1]
            example_indices = random.sample(not_banned_ix, min(len(not_banned_ix), N)) # step 4
            queried_example_indices.extend(example_indices)

            for example_ix in example_indices:
                _, ii = self._nearest_neighbors(self.K, example_ix) # step 4
                target_banned[ii.flatten().squeeze()] = 1 # step 5

        return queried_example_indices


    def _nearest_neighbors(self, k, seed):
        return self.tree.query(self.source_X[seed, :], k=k, n_jobs=-1)

    def _H(self, k, seed):
        H = 0
        _, ii = self._nearest_neighbors(k, seed)
        neighbors = self.source_y[ii]
        for c in self.classes:
            same_c = np.where(neighbors == c)[0]
            r = len(same_c)/float(k)
            if r > 0:
                H += (r * np.log2(r))
        return -H

