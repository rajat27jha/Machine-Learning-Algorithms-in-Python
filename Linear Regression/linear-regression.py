import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression  
import numpy as np
from sklearn.metrics import mean_squared_error

# Step 1: Load Data from CSV File
dataframe = pd.read_csv('student_scores.csv')
# this function returns a dataframe object hence we are calling it
# print(dataframe)

# Step 2: Plot the Data
X = dataframe['Hours'].values.reshape(-1, 1)  # .values will convert pandas object into numpy array
# because below it will throw an error
# we added reshape because error prompt was telling us to do so
Y = dataframe['Scores'].values.reshape(-1, 1)
# plt.plot(X,Y,'o')
# plt.show()

# Step 3: Build a Linear Regression Model
X_train, Y_train = X[0:20], Y[0:20]
X_test, Y_test = X[19:], Y[19:]
# print(X_train)
model = LinearRegression()
model.fit(X_train, Y_train)
# if we directly feed pandas dataframe to fit method it will throw an error because it wants numpy 2D array so we will
# convert it

# Step 4: Plot the Regression line
regression_line = model.predict(X)
# predict will return Y corresponding to the X
plt.plot(X, regression_line, color='black') # this will plot a best fit line where we give X and it predicts y, plots it
plt.plot(X_train, Y_train, 'o', color='red')
plt.plot(X_test, Y_test, 'o', color='blue')
# plt.show()

# Step 5: Make Predictions on Test Data
y_predictions = model.predict(X_test)

# Step 6: Estimate Error
print('MSE: ', mean_squared_error(Y_test,y_predictions))
