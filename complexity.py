import modules.complexity_estimator as ee

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


################################################################################################33

def vis(datasets, classifiers, nwindows, nclfs):
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
        estimator = ee.ComplexityEstimator(X, y, nwindows)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), nclfs + 2, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        i += 1

        # iterate over classifiers
        for name, clf in classifiers:
            ax = plt.subplot(len(datasets), nclfs + 2, i)
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
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot also the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                       alpha=0.6)

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

        ax = plt.subplot(len(datasets), nwindows + 1, j)

        if ds_cnt == 0:
            ax.set_title("Complexity")

        Ks, Es, = estimator.get_k_complexity()

        ax.plot(Ks, Es)
        j+=1

        Ss, Es, = estimator.get_s_complexity()
        for s in cmp[:][:][0]:
            ax = plt.subplot(len(datasets), nwindows + 1, j)
            ax.plot(Ks, Vs[1][s])
            j+=1


    plt.tight_layout()
    plt.show()

def main():

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes"]

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

    assert(len(classifiers) == len(names))

    clfs = zip(names, classifiers)
    nwindows = 10
    #X, y = make_classification(n_features=2, n_classes=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
    #rng = np.random.RandomState(2)
    #X += 4 * rng.uniform(size=X.shape)


    #linearly_separable = (X, y)
    datasets = [make_moons(noise=0.9),
                make_circles(noise=0.9, factor=0.1),
                make_gaussian_quantiles(n_classes=2)
                ]
    #make_hastie_10_2
    vis(datasets=datasets,classifiers=clfs, nwindows=nwindows, nclfs=len(classifiers))


if __name__ == "__main__":
    main()
