# LINEAR REGRESSION
# some changes made from prev file
# Here we r going to predict some unknown data

import pandas as pd
import quandl, math, datetime
import numpy as np
# this import will allow python to use array which python does not hv it previously

from sklearn import preprocessing, cross_validation, svm
# from preprocessing we'll use scaling, which'll be done on the features so that features would be between
# -1 to +1 for increased processing, time and accuracy.
# cross_validation will be used to create our testing and training samples.
# we can use svm to do regression and thats why we will not gonna use regression in future, its
# just for the example to see how it works how simple it is to change

from sklearn.linear_model import LinearRegression
# will come in handy for using regression


df = quandl.get('WIKI/GOOGL') # df for dataframe # goto quandl.com for finding dataset,
# here going to use googlewiki dataset

# print(df.head) # to find what we are working with
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
# Those are the columns
# Now we'll define another column ie high low percent
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

# redefining our prev data frame

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# print(df.head())
# prints the modified data frame ie removing unwanted or least significant

# video 3

# Adj. Close nad rest all are features ot labels
# ie we will take for ex ten value of Adj. close and that would be the feature to predict the future
# in this we will define a label
# to find the future price the only column of price we hv is Adj. close
# so we'll gonna need more info.

forecast_col = 'Adj. Close'
# this variable will contain features but in this ex its price
df.fillna(-99999, inplace=True)
# Machine Learning cant work with nan data, so we are replacing every nan with -99999(something) with this func
# or we could get rid of the whole column but its loss of data is not recomended

forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)
# this is the main regression algorithm. Here ceil is just the ceiling func
# 0.1 is 10 percent ie we are going to predict future for ex for 10 days(.1 may be bigger than that as we hv not set the
# time frame).

df['label'] = df[forecast_col].shift(-forecast_out)
# label variable will contain predicted values
# forecast_col will contain features
# -forecast_out shift upwards every value not clearly understood


# print(df.head())
# just prints only five rows

# video four

# features and labels r defined as X and y resp.
X = np.array(df.drop(['label'], 1))
# one that is written inside parenthesis will return a new data frame and that will be stored
# in X in the form of numpy array
# we r going to predict the values for this variable where we dont hv y values
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# this will scale x before we feed it to the classifier and then we will use it in real on real data
# Its all scaled together, so its like normalised with all other data points.
# Normalisation in simple means adjusting values measured on different scales to a notionally common scale.
# and thats why to properly scale it we hv to include it with our traning data X

# X = X[:-forecast_out+1]
# start from beg till (remove forecast_out+1 starting from end) remaining
# this will give us all points because we hv shifted data 1% previously upwards
# thats why we r eliminating from end as original data has been shifted upwards
# so that we only hv values for X's where we hv values for Y
# commented out cause labels has been dropped by df.dropna we dont need it now
# not understood clearly why commented

# df.dropna(inplace=True)
df.dropna(inplace=True)
# y = np.array(df['label'])
y = np.array(df['label'])

# print(len(X), len(y))
# to make sure that we hv correct lengths
# we dont hv correct length, hv to make a change

# lets train that shit

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
# this method inputs features and labels and test size here we specify it to be .2
# means this will test on 20% of the total data
# this method shuffels 'em up keeping X's and y' connected and give output to all variables
# X_train and y_train is used to fit our classifier

# and now we need to find a classifier
clf = LinearRegression(n_jobs=10)
# by default its 1, and we increase it so that the training speed of classifier increases.
# jobs means no. thread handled by the processor in the given time.
# clf = svm.SVR() # and thats how easy to shift the algorithm from linear to svm
# clf = svm.SVR(kernel="poly") # svr has kernels which we change it from linear to poly
# svr stands for support vector regression
# and for fitting it
clf.fit(X_train, y_train)
# fit means train and score means test
accuracy = clf.score(X_test, y_test)
# we are training the classifier on diff data set and testing them in another
# so as we know it has been trained properly
# print(accuracy)
# in linear regression accuracy is to be the squared error
# forecast_out is outputting 35 ie we are 35 days in future
# this was linear regression, there are many diff kinds of algos that can be super threaded
# before we jump into deep learning we'll gonna test 'em all.
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)
