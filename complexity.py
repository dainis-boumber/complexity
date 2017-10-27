import matplotlib.pyplot as plt

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC

import modules.complexity_estimator as ce
import modules.util as u
from modules.oracle import Oracle


################################################################################################33
#scatter plot of a dataset helper
#
def plot_ds(grid_size, loc, X, y, xx, yy, title, seeds=None, colspan=1, rowspan=1):

    ax = plt.subplot2grid(grid_size, loc, rowspan=rowspan, colspan=colspan)

    ax.set_title(title)
    # Plot also the training points
    ax.scatter(X[:, 0],X[:, 1], c=y)
    # and seeds
    if seeds is not None:
        ax.scatter(X[seeds, 0], X[seeds, 1],
                   alpha=1.0, facecolors='magenta')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

#perform active learning
#
def active(classifiers, src_datasets, tgt_datasets, quota=25, plot_every_n=5):
    # USE THIS INSTEAD OF YTGT WHICH WE PRETEND TO NOT KNOW
    X_src, y_src = src_datasets[0]
    X_tgt, y_tgt = tgt_datasets[0]
    u_tgt = [None] * len(X_tgt)
    est_src = ce.ComplexityEstimator(X_src, y_src)
    est_tgt = ce.ComplexityEstimator(X_tgt, y_tgt)
    # declare Dataset instance, X is the feature, y is the label (None if unlabeled)
    X = np.vstack((X_src, X_tgt))

    h = .05  # step size in the mesh
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    figure = plt.figure(figsize=(27, 13))
    grid_size = (1+len(classifiers), 6)
    for n, classifier in enumerate(classifiers):
        model = classifier
        oracle = Oracle(X_tgt, y_tgt)
        # plot src
        plot_ds(grid_size, (0, 0), X_src, y_src, xx, yy, 'Src', est_src.seeds)
        ax = plt.subplot2grid(grid_size, (0,1), colspan=2)
        ax.set_title('Src complexity')
        Ks, Es = est_src.get_k_complexity()
        ax.plot(Ks, Es)
        #plt tgt
        plot_ds(grid_size, (0, 3), X_tgt, y_tgt, xx, yy, 'Tgt', est_tgt.seeds)
        ax = plt.subplot2grid(grid_size, (0,4), colspan=2)
        Ks, Es = est_tgt.get_k_complexity()
        ax.set_title('Tgt complexity')
        ax.plot(Ks, Es)
        w = 0
        X_known = X_src.tolist()
        y_known = y_src.tolist()
        for i in range(quota):  # loop through the number of queries
            loc, y_loc = oracle.random_query()  # let the specified QueryStrategy suggest a data to query
            u_tgt[loc] = y_loc
            X_known.append(X_tgt[loc])
            y_known.append(y_tgt[loc])
            if i == 0 or i % plot_every_n == 0 or i == quota - 1:
                model.fit(X_known, y_known)  # train model with newly-updated Dataset
                score = model.score(X_tgt, y_tgt)
                ax = plt.subplot2grid(grid_size, (n + 1, w))
                if hasattr(model, "decision_function") or len(set(y_known)) != 2:
                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                else:
                    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

                # Put the result into a color plot
                Z = Z.reshape(xx.shape)

                ax.contourf(xx, yy, Z, alpha=.3)

                # Plot also the training points
                ax.scatter(X_tgt[:, 0], X_tgt[:, 1], c=y_tgt)
                ax.set_xlim(xx.min(), xx.max())
                ax.set_ylim(yy.min(), yy.max())
                ax.set_xticks(())
                ax.set_yticks(())
                if i == 0:
                    ax.set_ylabel(u.classname(model))
                if n == 0:
                    ax.set_title('# queries=' + str(i))
                ax.set_xlabel('Accuracy='+('%.2f' % score).lstrip('0'))
                w += 1
    figure.tight_layout()
    figure.savefig(filename=('./vis/active.png'))
    plt.show()

def main():
    clfs = [SVC(), LinearSVC(), AdaBoostClassifier(), GaussianNB()]
    src_datasets = []
    tgt_datasets = []

    src_datasets.append(make_blobs(n_samples=200, centers=3, cluster_std=3.0))
    tgt_datasets.append(make_blobs(n_samples=100, centers=3, cluster_std=5.0))

    active(classifiers=clfs, src_datasets=src_datasets, tgt_datasets=tgt_datasets)
    #make_hastie_10_2

if __name__ == "__main__":
    main()
