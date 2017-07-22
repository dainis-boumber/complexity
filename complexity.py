from sklearn.model_selection import KFold
from sklearn.metrics import *
from scipy.stats import mode
from os import listdir
from os.path import isfile, join
from sklearn import metrics
from adjustText import adjust_text
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from matplotlib.patches import Ellipse
import entropy_estimators as ee
import scipy
import random
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




def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).
    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def pdf(x, mu, sigma):
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp( -(x - mu)**2 / (2 * sigma**2) )


#plot histogram of a distribution without showing it, call plt.show() to do so
def graph(dist, pdf, color):
    plt.plot(dist, pdf, linewidth=2, color=color)

def interpolate_to_get_Px(D1, D2, Px_d1_c1, Px_d2_c2, x, py):
    return np.interp(x, D1, Px_d1_c1) * py + np.interp(x, D2, Px_d2_c2) * py

def compute_bayes(pxc, py, px):
	return pxc*py/px

def draw(data, labels, ax):
    texts = []
    for label, x, y in zip(labels, data[:, 0], data[:, 1]):
        texts.append(ax.text(x, y, label, fontsize=8))
    return texts

def nearest_neighbors(X, k, seed_idx):
    tree = scipy.spatial.cKDTree(X)
    d, i = tree.query(X[seed_idx, :], k=k)
    return i

def ratio(neighbors, c):
    count = 0
    for n in neighbors:
        if int(n) == int(c):
            count += 1
    return float(count)/float(len(neighbors))

def H(X, y, classes, k, seed_idx):
    H = 0
    ii = nearest_neighbors(X, k, seed_idx)
    for c in classes:
        r = ratio(y[ii], c)
        if r > 0:
            H += (r * np.log2(r)) 
    return -H

def get_entropies(X, y, nwindows, classes=[0, 1]):
    seeds = random.sample(range(0, len(X)-1), nwindows)
    entropies = []
    Ks = []
    for i in range(3, len(X)+1):
        Ks.append(i)
        s = 0
        for seed in seeds:
            s += H(X, y, classes=classes, k=i, seed_idx=seed)
        s = s/nwindows
        entropies.append(s)

    return Ks, entropies

################################################################################################33
h = .02  # step size in the mesh

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

#X, y = make_classification(n_features=2, n_classes=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
#rng = np.random.RandomState(2)
#X += 4 * rng.uniform(size=X.shape)

n_samples = 100

# Generate 3 blobs with 2 classes where the second blob contains
# half positive samples and half negative samples. Probability in this
# blob is therefore 0.5.
#centers = [(-5, -5), (0, 0), (5, 5)]
#X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
#                  centers=centers, shuffle=False)

#linearly_separable = (X, y)
datasets = [make_moons(noise=0.9),
            make_circles(noise=0.9, factor=0.1),
            make_gaussian_quantiles(n_classes=2)
            ]
#make_hastie_10_2
nwindows = 10

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
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
    ax = plt.subplot(len(datasets), len(classifiers) + 2, i)
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
    
    ax = plt.subplot(len(datasets), len(classifiers) + 2, i)
    
    Ks, entropies = get_entropies(X, y, nwindows)

    if ds_cnt == 0:
        ax.set_title("Complexity")

    #ax.set(xlabel="$k$", ylabel="$H$")
    ax.plot(Ks,entropies)
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 2, i)
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

plt.tight_layout()
plt.show()

