# for Euclidean dist. formula or how to calculate it see video 15
# this is the algorithm of K nearest neighbors.

from math import sqrt
import numpy as np
# numpy has actually a built in func that calculates euclidean dist and its much faster there

import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
# by its help we are gonna do votes
import warnings

style.use('fivethirtyeight')

# plot1 = [1, 3]
# plot2 = [2, 5]
# euclidean_distance = sqrt((plot1[0] - plot2[0])**2 + (plot1[1] - plot2[1])**2)
# print(euclidean_distance)

dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
# in our dataset k and r are labels and other are their features belonging to them

new_feature = [5, 7]
# this feature is gonna fit to which label, we are gonna find out

# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0], ii[1], s=100, color=i)
# First Line: for each element in dataset,
# Sec line: for each element in first label ie 1, 2 etc.
# Third line: plot those coordinates in graph, arguments s is size color is like for i=0
# color will be for ex red and diff for rest
# This for loop can be written easily in one line as

# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_feature[0], new_feature[1])
# plt.show()


def k_nearest_neighbor(data, predict, k=3):
    # data is the total no of labels
    # predict is the feature
    # k=3 means three nearest points and their label
    if len(data) >= k:
        warnings.warn("K is set to a value less than total voting groups!")
    distances = []
    for groups in data:
        for features in data[groups]:
            # explanation same as above
            # euclidean_dictance = np.sqrt(np.sum(np.array(features)-np.array(predict)))
            # features array and predict array are directly subracted and work as same way as above
            euclidean_dictance = np.linalg.norm(np.array(features) - np.array(predict))
            # same as written above
            distances.append([euclidean_dictance, groups])

    votes = [i[1] for i in sorted(distances)[:k]]
    # distances will be a list of list, in which in every list in distances first element will be
    # euclidean_distance and sec. will be its corresponding grp or label
    # sorted will sort the list acc. to dist.
    # votes will be a list in which only k elements will be there (here 3)
    # i[1] means index 1 ie grp of inside list of distances
    # top 3 grups will enter votes which has least dist from predict

    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    # it is an ordinary array
    # in counter method input will be a list and method will calculate the frequency of each element
    # in most common first parameter will be how much we want ie top 1 or top 2 frequency
    # most common returns a sorted list of tuple in which first element is the grp and sec. ele.
    # is its freq. so [0][0] means i want only highest frequency grp
    return vote_result


result = k_nearest_neighbor(dataset, new_feature, k=3)
print(result)

[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_feature[0], new_feature[1], color='r')
plt.show()
