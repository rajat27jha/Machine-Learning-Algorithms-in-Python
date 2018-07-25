# here machine finds out how many no. of clusters are required
# hierarchical clustering
# step 1: assume every data pt. as cluster center
# step 2: take any radius/bandwidth around a data point say 1 and apply to each and every cluster center
# step 3: take the mean of all the data pts. which are falling inside the circle of assumed radius.
#         and say it as a new cluster center
# step 4: repeat the same process for the newly obtained cluster center, rad will be same, again we
#         take all cluster centers that are falling in, find mean, assume as a new cluster center
# step 5: repeat until no new cluster centers are inbound the center ie cluster center is not moving
#         and at this pt. it is said to be optimized (this is known as convergence)

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")

centers = [[1, 1, 1], [5, 5, 5], [3, 10, 10]]

X, _ = make_blobs(n_samples=100, centers=centers, cluster_std=1.5)
# arg 1: no. of samples, arg 2: fixed center locations, arg 3:standard deviation among pts
# returns x ie samples and Y ie _ ie int no. for cluster membership of each sample

ms = MeanShift()
ms.fit(X)
labels = ms.labels_
# list of labels of each point
# print(labels)
cluster_centers = ms.cluster_centers_

print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters:", n_clusters_)

colors = 10*['r', 'g', 'b', 'c', 'k', 'y', 'm']
fig = plt.figure()
# inside figure add_subplot is present, it ts done to use that
# it is also used when two or graphs are used simultaneously
ax = fig.add_subplot(111, projection='3d')
# 111 is for viewing only, chnge that to 211 for more clarification
for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
           marker="x", color='k', s=150, linewidths=5, zorder=10)

plt.show()