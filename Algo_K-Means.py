# copied from K-Means
# this is unsupervised learning ie u are given a dataset and we hv to find groups
# K means is flat clustering ie are of groups are specified before
# Doubt as its not working
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np


style.use('ggplot')

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]
              ])

plt.scatter(X[:, 0], X[:, 1], s=150)
# All zeroth element of the x array (X[:, 0])
# All the first element of the x array (sec. one)
# s is size
plt.show()
colors = ['g', 'r', 'c', 'b', 'k', 'c']


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        # tol is tolerance which means how much the centroid will gonna move
        # max is the limit of iterations ie if after 300 iteration centroid is found then stop.
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}
        # empty dict. which will contain centroids
        # keys will be centroids here 2
        for i in range(self.k):
            # here k=2 hence two times
            self.centroids[i] = data[i]
            # it will assign first two centroids to first two coordinates

        for i in range(self.max_iter):

            self.classifications = {}
            # empty dict that will contain centroids and their classifications
            # we began optimization process
            # keys will be the centroids and values feature sets that contain those values
            # after every iteration classification will get clear out because no. of elements
            # or index will gonna change but centroids dict. will always hv same no. of elements

            for i in range(self.k):
                self.classifications[i] = []
                # both keys will hv value list

            for featureset in X:

                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                # Now, that's Euclidian Distance
                # it will be a list which will be populated with k no. of values, here only 2 values will be there
                # for each centroid in centroids here only 0, 1, append the distance between the zeroth
                # centroid and that featureset or data point and second element will be dist. bet
                # second centroid centroid and that data point

                classification = distances.index(min(distances))
                # take the index of min element in distances and put it in classification

                self.classifications[classification].append(featureset)

            prev_centroid = dict(self.centroids)
            # in next step we are updating centroids hence storing previous one
            for classification in self.classifications:

                self.centroids[classification] = np.average((self.classifications)[classification], axis=1)
                # this code will redifine our new centroid
                # taking average of all values that are present zeroth and oneth classification.
                # at first we will comment this, in such case our centroids will not change
            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroid[c]
                current_centroid = self.centroids[c]

                # if np.sum((current_centroid - original_centroid)/original_centroid) * 100) > self.tol:
                #     optimized = False
                # it should not be commented but not working
            if optimized == False:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker='o', color='k', linewidths=5)
#                       (abcissa,                   ordinate) for ref see type of dataset we are providing

for classification in clf.classifications:
    color = colors[classification]
    # in built method colors will return a color to color variable
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidths=5)

plt.show()
