import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np


style.use('ggplot')

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11],
              [8, 2],
              [10, 2],
              [9, 3]])

# plt.scatter(X[:, 0], X[:, 1], s=150)
# All zeroth element of the x array (X[:, 0])
# All the first element of the x array (sec. one)
# s is size
# plt.show()
colors = 10*['g', 'r', 'c', 'b', 'k', 'c']

# for steps see Mean_Shift


class MeanShift:
    def __init__(self, radius=4):
        self.radius = radius
        # self.radius_norm_step = radius_norm_step

    def fit(self, data):
        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]
            # means every data pt. is a centroid
        while True: # infinite loop
            new_centroids = []  # this list will contain all the new centroids
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    if np.linalg.norm(featureset-centroid) < self.radius:
                        # we r finding the euclidian distance bet feature and centroid and if its less than
                        # radius the it is in the circle so append it to in_bandwidth list
                        in_bandwidth.append(featureset)

                new_centroid = np.average(in_bandwidth, axis=0)  # it is a variable not a list
                new_centroids.append(tuple(new_centroid))
                # we converted it to tuple because numpy array and tuple has different attributes
                # we are going to use tuple attributes to reference some things

            uniques = sorted(list(set(new_centroids)))
            # here tuple will help in finding set ie unique elements although numpy has also a func named
            # unique but that will find all unique values values not elements

            prev_centroid = dict(centroids)
            # this will only copy the attributes from centroids ie values not keys

            centroids = {}
            # clearing it all

            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroid[i]):
                    optimized = False
                if not optimized:
                    break
            # in case when previous centroids and current centroids are equal that means they are not shifting
            # hence they optimized so we can get out of this while loop
            # our new centroids will become prev centroids in next iteration
            if optimized:
                break

        self.centroids = centroids

    def predict(self, data):
        pass


clf = MeanShift()
clf.fit(X)

centroids = clf.centroids

plt.scatter(X[:, 0], X[:, 1], s=150)
for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)
plt.show()
# till now we hv made the algo but there are some flaws in it like if we increase the value of bandwidth
# there is only one centroid and for 2 data sets are themselves are becoming centroids
# how we are going to give radius without knowing the data set thats the problem, we hv to automate this task
# we can give very large radius and then penalise all outliers