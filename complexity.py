import modules.complexity_estimator as ce

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs, make_gaussian_quantiles
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

################################################################################################33

def vis(datasets, dsnames, classifiers, clfnames, nwindows):
    h = .02  # step size in the mesh
    figure = plt.figure(figsize=(27, 9))
    f1 = figure.number
    figure2 = plt.figure(figsize=(27, 9))
    f2 = figure2.number

    i = 1
    j = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        plt.figure(f1)
        # preprocess dataset, split into training and test part
        X, y = ds
        estimator = ce.ComplexityEstimator(X, y, nwindows)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))


        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        ax.set_title(dsnames[ds_cnt])
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        i += 1

        # iterate over classifiers
        for name, clf in zip(clfnames, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=.8)

            # Plot also the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)


            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1

        plt.figure(f2)
        # plot data and
        ax = plt.subplot(len(datasets), 2, j)

        ax.set_title(dsnames[ds_cnt])
        # Plot also the training points
        ax.scatter(X[:, 0], X[:, 1], c=y)
        # and seeds
        ax.scatter(X[estimator.seeds, 0], X[estimator.seeds, 1],
                   alpha=1.0, facecolors='black')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        j +=1

        ax = plt.subplot(len(datasets), 2, j)
        Ks, Es = estimator.get_k_complexity()
        if ds_cnt == 0:
            ax.set_title('Avg. Complexity')
        ax.plot(Ks, Es)
        j+=1
        '''
                ws = estimator.get_w_complexity()
                for wi, w in enumerate(ws):
                    ax = plt.subplot(len(datasets), nwindows + 2, j)
                    #ax.set_title("Window %d Seed %s" % (wi, str(X[estimator.seeds[wi]]) ))
                    ax.plot(Ks, w)
                    j+=1
        '''

    figure.tight_layout()
    figure2.tight_layout(h_pad=1.0)
    figure.savefig(filename=('./vis/'+ ''.join(dsnames)+'Classifications.png'))
    figure2.savefig(filename=('./vis/'+''.join(dsnames) + 'Complexities.png'))
    plt.show()

def plot_ds(loc, X, y, xx, yy, quota, seeds=None, colspan=1, rowspan=1):

    ax = plt.subplot2grid((2, 6), loc, rowspan=rowspan, colspan=colspan)

    ax.set_title('Src')
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



def active(X_src, X_tgt, y_src, y_tgt, quota):
    assert(quota % 5 == 0)
    ####USE THIS INSTEAD OF YTGT WHICH WE PRETEND TO NOT KNOW
    u_tgt = [None] * len(X_tgt)
    est_src = ce.ComplexityEstimator(X_src, y_src)
    est_tgt = ce.ComplexityEstimator(X_tgt, y_tgt)
    # declare Dataset instance, X is the feature, y is the label (None if unlabeled)
    model = MLPClassifier()
    oracle = Oracle(X_tgt, y_tgt)

    ##Begin plot
    X = np.vstack((X_src, X_tgt))
    y = np.vstack((y_src, y_tgt))

    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    figure = plt.figure(figsize=(27, 9))

    # plot src
    plot_ds((0, 0), X_src,y_src,xx, yy, quota, est_src.seeds)
    ax = plt.subplot2grid((2, 6), (0,1), colspan=2)
    ax.set_title('Src complexity')
    Ks, Es = est_src.get_k_complexity()
    ax.plot(Ks, Es)
    #plt tgt
    plot_ds((0, 3), X_tgt,y_tgt,xx, yy, quota, est_tgt.seeds)
    ax = plt.subplot2grid((2, 6), (0,4), colspan=2)
    Ks, Es = est_tgt.get_k_complexity()
    ax.set_title('Tgt complexity')
    ax.plot(Ks, Es)
    w = 0
    for i in range(quota):  # loop through the number of queries
        loc, y_loc = oracle.random_query()  # let the specified QueryStrategy suggest a data to query
        u_tgt[loc] = y_loc
        X_tmp = []
        y_tmp = []
        if i % 5 == 0:
            for j, yt in enumerate(u_tgt):
                if yt is not None:
                    X_tmp.append(X_tgt[j])
                    y_tmp.append(yt)
            X_tmp = np.vstack((X_src, X_tmp))
            y_tmp = np.array(y_tmp)
            y_tmp = np.hstack((y_tgt, y_tmp))
            model.fit(X_tmp, y_tmp)  # train model with newly-updated Dataset
            score = model.score(X_tgt, y_tgt)
            ax = plt.subplot2grid((2, 6), (1,w))
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

            ax.set_title(str(i))
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
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

    X_src, y_src = make_blobs(n_samples=100, centers = 3, cluster_std=3.0)
    X_tgt, y_tgt = make_blobs(n_samples=100, centers = 3, cluster_std=3.0)
    #linearly_separable = (X, y)

    active(X_src, X_tgt, y_src, y_tgt, 30)
    #make_hastie_10_2
    #vis(datasets=datasets, dsnames=dsnames, classifiers=classifiers, clfnames=names, nwindows=nwindows)


if __name__ == "__main__":
    main()
