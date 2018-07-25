# LINEAR REGRESSION ALGORITHM
# this algo. finds out the best fit line to the provided data set
# to find the line we must know its slope and y-intercept
# and this is what we are gonna do!!
from statistics import mean
# from matplotlib import style
import numpy as np
import matplotlib.pyplot as plt
import random
# to plot graphs
plt.style.use('fivethirtyeight')
# taking any random data set
# xs = [1, 3, 4, 5, 6, 9]
# ys = [5, 4, 6, 5, 7, 5]

# # plt.plot(xs, ys)
# # to plot those lists
# plt.scatter(xs, ys)
# # to see in non connected format
# plt.show()

# In previously our data was in the form of numpy array

# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# its not necessary to specify the data type but is precautionary
# float64 data type is Double precision float ie occupies 64 bits in comp. mem. it represents
# a wide dynamic range of numeric values
# ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def create_dataset(hm, variance, step=2, correlation=False):
    # hm is how much data pts we hv to create
    # variance is like range
    # step is used so that data could be varied more, see use of step in code for more clarification
    # correlation tells that data should be varied in which dir.
    # if its pos that val and step will be added otherwise subtracted
    # correlation and step is simple used to make data set linear thats it
    # if correlation true and pos is given graph will go upwards and vice versa
    # if correlation is false step will not work and data set will be scattered all around
    # and thats not goog for linear regression
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    # xs will linearly grow ex.: 0,1,2,3....,till how much
    # this was imp as data should grow linearly for proper application of lin reg.
    # ys is also step wise increasing upwards or downwards acc. to whatever parameter
    # not like now at top and next bottem, its linearly increasing
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
    # we hv to specify data type, not necessary but in video he has written thats why i hv written


xs, ys = create_dataset(40, 40, 2, correlation='pos')


def best_fit_slope_intercept(xs, ys):
    # finding out the slope and y-intercept
    m = ((mean(xs) * mean(ys)) - mean(xs*ys)) / ((mean(xs) * mean(xs)) - (mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m, b


m, b = best_fit_slope_intercept(xs, ys)
print(m, b)
# regression_line = []
# for x in xs:
#     regression_line.append((m*x)+b)
# this whole shit can be written be written easily like this

regression_line = [((m*x)+b) for x in xs]
# y = mx + b
predict_x = 8
predict_y = ((m * predict_x) + b)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()

# till now we hv created a model for our data
# to predict any data this is how its done

# predict_x = 8
# predict_y = ((m * predict_x) + b)
# plt.scatter(predict_x, predict_y, color='g')

# Now we want to calculate how good is our best fit line
# and this is done by determining r squared error
# the sq. of dist. of pt. to the line is squared error
# it is squared because there are both negative and positive values, square will manage that
# modules is not taken because we also hv to penalise the outliers(pts. which are way far from the line)
# the code for determining the coefficient of determination or r squared is given below


def squared_error(ys_orig, ys_line):
    # only sq. error not r squared
    return sum((ys_line-ys_orig)**2)


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    # this'll create a straight line ex parallel to x cause every value is same in that list
    # since mean is unique for any list
    # we want reg. line, y_mean_line, and then sq. error of both
    # to calculate squared error use above func
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)


r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)
