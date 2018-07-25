# this is not working, i dunno why
# in this applying k means to the titanic dataset
# we'll understand how to work with non numerical data
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import k_means
import pandas as pd
from sklearn import preprocessing


df = pd.read_excel('titanic.xls')
# will read excel sheets, loving pandas
# to do this i hv to install xlrd
# print(df.head())
# as address and sex column are non numeric we'll see how to handle it

df.drop(['body', 'name'], 1, inplace=True)
# df.convert_objects(convert_numeric=True)
# this will convert all string values in df to NaN
df.fillna(0, inplace=True)


def handel_non_numerical_data(df):
    columns = df.columns.values
    # column will be a list without commas, that is only headings, which contain all the headings of df
    # print(columns)

    for column in columns:
        # for each heading in columns
        text_digit_values = {} # empty dictionary

        def convert_to_int(val): # dummy code
            return text_digit_values[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            # if it will be a character
            columns_contents = df[columns].values.tolist()
            # it will also be a list and with that heading contents because bracket is placed.
            unique_elements = set(columns_contents)
            # this will be a set of all unique elements in columns with no repitions
            x = 0 # flag variable
            for unique in unique_elements:
                if unique not in text_digit_values:
                    text_digit_values[unique] = x
                    # this is a dict. where unique will be key and x will be value
                    x += 1

            df[column] = list(map(convert_to_int(), df[column]))
            # this is the major func called mapping func. a feature in pandas
            # it will take each element from that content apply it to convert_to_int() func
            # which will return value of that key()dict.
            # for ex. in case of male it will assign 1 and vice versa
            # and make the column as 1 0 1 0 1 0 1 1 0

        return df


df = handel_non_numerical_data(df)
df.drop('home.dest', 1, inplace=True)
print(df.head())


X = np.array(df.drop(["survived"], 1).astype(float))
# we dropped survived column because otherwise it would be cheating

y = np.array(df['survived'])

clf = k_means(n_clusters=2).fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0]==y[i]:
        correct += 1

print(correct/len(X))
