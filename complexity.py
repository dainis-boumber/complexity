import modules.complexity_estimator as ce

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_hastie_10_2, make_moons, make_gaussian_quantiles, make_circles, make_classification, make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from modules.oracle import Oracle
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import modules.util as u
################################################################################################33


#scatter plot of a dataset helper
#
def plot_ds(grid_size, loc, X, y, xx, yy, quota, title, seeds=None, colspan=1, rowspan=1):

    ax = plt.subplot2grid(grid_size, loc, rowspan=rowspan, colspan=colspan)

    ax.set_title(title)
    # Plot also the training points
    ax.scatter(X[:, 0],X[:, 1], c=y)
    # and seeds
    if seeds is not None:
        ax.scatter(X[seeds, 0], X[seeds, 1],
                   alpha=1.0, facecolors='black')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

#perform active learning
#
def active(classifiers, X_src, X_tgt, y_src, y_tgt, quota):
    assert(quota % 5 == 0)
    ####USE THIS INSTEAD OF YTGT WHICH WE PRETEND TO NOT KNOW
    u_tgt = [None] * len(X_tgt)
    est_src = ce.ComplexityEstimator(X_src, y_src)
    est_tgt = ce.ComplexityEstimator(X_tgt, y_tgt)
    # declare Dataset instance, X is the feature, y is the label (None if unlabeled)
    X = np.vstack((X_src, X_tgt))

    h = .05  # step size in the mesh
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    figure = plt.figure(figsize=(27, 13))
    grid_size = (1+len(classifiers), 6)
    for n, classifier in enumerate(classifiers):
        model = classifier
        oracle = Oracle(X_tgt, y_tgt)
        # plot src
        plot_ds(grid_size, (0, 0), X_src,y_src,xx, yy, quota, 'Src', est_src.seeds)
        ax = plt.subplot2grid(grid_size, (0,1), colspan=2)
        ax.set_title('Src complexity')
        Ks, Es = est_src.get_k_complexity()
        ax.plot(Ks, Es)
        #plt tgt
        plot_ds(grid_size, (0, 3), X_tgt,y_tgt,xx, yy, quota, 'Tgt', est_tgt.seeds)
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
            if i % 5 == 0:
                model.fit(X_known, y_known)  # train model with newly-updated Dataset
                score = model.score(X_tgt, y_tgt)
                ax = plt.subplot2grid(grid_size, (n+1,w))
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

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
    '''
    names = ["kNN", "SVM-linear", "SVM-rbf", "Gaussian Process",
             "Decision Tree", "Random Forest", "NN", "AdaBoost",
             "NaiveBayes"]

    classifiers = [
        KNeighborsClassifier(3),
        LinearSVC(),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB()]

    classifiers = [MLPClassifier()]
    names = ['MLP']

    assert(len(classifiers) == len(names))

   '''
    nwindows = 10
    #X, y = make_classification(n_features=2, n_classes=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
    #rng = np.random.RandomState(2)
    #X += 4 * rng.uniform(size=X.shape)

    X_src, y_src = make_blobs(n_samples=200, centers = 2, cluster_std=3.0)
    X_tgt, y_tgt = make_blobs(n_samples=100, centers = 2, cluster_std=5.0)
    #X_src, y_src = make_gaussian_quantiles(n_features=10, n_classes=2)
    #X_tgt, y_tgt = hastie(n_samples=1000)
    #linearly_separable = (X, y)

    active([SVC(), LinearSVC(), AdaBoostClassifier(), GaussianNB()], X_src, X_tgt, y_src, y_tgt, 30)
    #make_hastie_10_2
    #vis(datasets=datasets, dsnames=dsnames, classifiers=classifiers, clfnames=names, nwindows=nwindows)


if __name__ == "__main__":
    main()
