# this is unsupervised learning ie u are given a dataset and we hv to find groups
# K means is flat clustering ie are of groups are specified before
# This prog will not work as method names has changed in original documentation
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans

style.use('ggplot')

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

# plt.scatter(X[:, 0], X[:, 1], s=150)
# All zeroth element of the x array (X[:, 0])
# All the first element of the x array (sec. one)
# s i size
# plt.show()

clf = KMeans(n_clusters=2)
clf.fit(X)
# as its flat i given the no. of clusters to be formed is 2

# its the training of our classifier on X

centroids = clf.cluster_centers_
print(centroids)
# it will be an array of list which will give the coordinates of clusters here 2 coordinates will be given
labels = clf.labels_
# it will be a list equal to length of X, and contain label to the corresponding dataset
# for ex. for [1, 2] label will be 0, so 0 will be placed in that position

colors = ['g.', 'r.', 'c.', 'b.', 'k.', 'c.']

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
# colors[labels[i]] will output a color, in this case only two grps are there hence two types of color.
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5)
# will plot two 'cross' in graph and coordinates will be taken from centroids
plt.show()
