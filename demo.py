import matplotlib.pyplot as plt

import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

import modules.complexity_estimator as ce
import modules.util as u


#This experiment is generic and is best used to demonstrate our approach
# May be out of date
def demo(datasets, dsnames, classifiers, nwindows):
    h = .05  # step size in the mesh
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
        for clf in classifiers:
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function") or len(set(y_test)) != 2:
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
                # Put the result into a color plot

            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=.3)
            # Plot also the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(u.classname(clf))
            ax.set_xlabel('Accuracy: %.2f' % score)
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
    figure2.tight_layout()
    figure.savefig(filename=('./vis/'+ ''.join(dsnames)+'Classifications.png'))
    figure2.savefig(filename=('./vis/'+''.join(dsnames) + 'Complexities.png'))
    plt.show()

def main():
    classifiers = [
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        KNeighborsClassifier(3),
        MLPClassifier(alpha=1),
        SVC(gamma=2, C=1),
        LinearSVC(),
        GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB()]


    nwindows = 10
    datasets=[]
    dsnames = []
    datasets.append(make_classification(n_features=2, n_classes=2, n_redundant=0, n_informative=2, n_clusters_per_class=1))
    dsnames.append('Classification')
    datasets.append(make_blobs(n_samples=100, centers = 2, cluster_std=3.0))
    dsnames.append('Blobs 2_2_3')
    #datasets.append(make_blobs(n_samples=200, centers = 3, cluster_std=5.0))
    #dsnames.append('Blobs 3_3_5')
    #datasets.append(make_gaussian_quantiles(n_features=20, n_classes=2))
    #dsnames.append('Gaussian Quantiles 20_2')
    #datasets.append(u.hastie(n_samples=1000))
    #dsnames.append('Hastie_10_2')
    datasets.append(make_moons())
    dsnames.append('Moons')
    datasets.append(make_circles())
    dsnames.append('Circles')

    demo(datasets=datasets, dsnames=dsnames, classifiers=classifiers, nwindows=nwindows)


if __name__ == "__main__":
    main()
