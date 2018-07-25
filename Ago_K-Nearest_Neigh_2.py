# Algo is same but in this we apply real world data set  and test our algorithm's accuracy

from math import sqrt
import numpy as np
# numpy has actually a built in func that calculates euclidean dist and its much faster there

from collections import Counter
# by its help we are gonna do votes
import warnings
import pandas as pd
import random


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

    # print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    # it is an ordinary array
    # in counter method input will be a list and method will calculate the frequency of each element
    # in most common first parameter will be how much we want ie top 1 or top 2 frequency
    # most common returns a sorted list of tuple in which first element is the grp and sec. ele.
    # is its freq. so [0][0] means i want only highest frequency grp
    return vote_result


df = pd.read_csv("breast-cancer-wisconsin.data.txt")
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

full_data = df.astype(float).values.tolist()
# when we replaced '?', some values in data set changed to char, in data set everything should be
# a float or int otherwise problem
# so thats why we hv to convert everything to float
# print(full_data[:10])
# full data will be a list of list

random.shuffle(full_data)
# shuffling of our data without losing the connection bet features and labels

test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
# this will be our format for both sets. 2 and 4 are keys because they are labels.

train_data = full_data[:-int(test_size*len(full_data))]
# train data is 80% of the total data thats what done here
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
    # train_set[This will be the index where it'll gonna append, it will se that ith list and -1 means last column
    # whose value will be 2 or 4 ].append(appending that whole list except last)

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbor(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

print('Accuracy: ', correct/total)

