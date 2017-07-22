
### Complexity based on entropy Experiment

As a reminder, we peek N seeds and grow windows around them, 
gradually increasing them to incorporate the tntire dataset.
Number of windows where entropy is calculated is 10.

Following are the "recipes" used to generate datasets:

make_moons - the two moons one, difficult for some classifiers, we control noise

make_circiles - circles within each other, we control noise and distance from circle to circle

make_classification - produce two clusters with some control over features, by default cluster centers are on the the corners
of a hypercube.

make_gaussian_quantiles - This classification dataset is constructed by taking a multi-dimensional standard normal distribution and defining classes separated by nested concentric multi-dimensional spheres such that roughly equal numbers of samples are in each class (quantiles of the \chi^2 distribution). A generalization of classical problem introduced by Trevor Hastie in 2009. 
Hastie, R. Tibshirani and J. Friedman, “Elements of Statistical Learning Ed. 2”, Springer, 2009.



###Figure 1

datasets = [make_moons(noise=0.0),
            make_circles(noise=0.0, factor=1.0),
            make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
            ]

###Figure 2

datasets = [make_moons(noise=0.3),
            make_circles(noise=0.3, factor=0.7),
            make_classification(n_features=2, n_redundant=1, n_informative=1, n_clusters_per_class=1)
            ]



###Figure 3

datasets = [make_moons(noise=0.6),
            make_circles(noise=0.6, factor=0.4),
            make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=2)
            ]

###Figure 4

datasets = [make_moons(noise=0.9),
            make_circles(noise=0.9, factor=0.1),
            make_gaussian_quantiles(n_classes=2)
            ]


            

