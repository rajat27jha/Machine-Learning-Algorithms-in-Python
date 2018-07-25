# first we download the data set from ucl Ml site, where we downloaded breast cancer orig data
# and the second file will give the info about that data ex features or which no. is what
# in file all missing or bad data is implied with '?' so we hv to take care of that also

import numpy as np

from sklearn import preprocessing, cross_validation, neighbors
# we hv covered preprocessing, cross_validation, and neighbours will be like find all neighbouring
# data pts. around a selected data pt

import pandas as pd

df = pd.read_csv(r'D:\Py Work ML\breast-cancer-wisconsin.data.txt')
# this method in pandas will read our downloaded data set file and put it in df

df.replace('?', -99999, inplace=True)
# inplace is important is imp cause default value is False which will not modify our data set
# -99999 is used because most algos will treat it as outlier
# if we drop those missing data instead of replacing it,we'll like lose our 50% of data and that doesnt considered good.

df.drop(['id'], 1, inplace=True)
# we also hv to drop the id column cause the id column is useless data and become an outlier
# and k nearest neig. performs worst for outliers
# no. 1 is given to specify the axis of modification
# if we do not drop this column accuracy will be depreciated greatly

X = np.array(df.drop(['class'], 1))
# X is for features and y is for labels
# X will contain everything leaving class

y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
# this is how we shuffle the data and separate it into training and testing chunks

clf = neighbors.KNeighborsClassifier()
# and here we apply the algo of k nearest for our clf

clf.fit(X_train, y_train)
# training of our classifier
# till now most of the things are similar as we hv seen before, thats why skikit learn is amazing

accuracy = clf.score(X_test, y_test)
print(accuracy)

# Till now we hv trained the classifier, now we are gonna make a prediction

example_measures = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])
# if i dont apply sq. brackets then an error will occur saying 'only 2 non-keyword arguments accepted'
# i donno what that means but will be resolved by applying sq. brackets

example_measures = example_measures.reshape(1, -1)
# Now i hv to do this as error was coming that 2D array expected, so i have to reshape it
# -1 means numpy will automatically figure out the dimentions of the array
# 1 means no. of rows will be 1 and no. of columns will be unknown, therefore it will convert
# what it seems to be a 2D array to 1D array

# example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
# example_measures = example_measures.reshape(len(example_measures), -1)
# here, we hv two lists for two patients, so to automatically know how much rows we want
# instead of 1 write above code

prediction = clf.predict(example_measures)
print(prediction)

# here we used scikit learn to execute K nearest neighbors, but next we will build an algo
# and then use this exact data set to compare ourselves
