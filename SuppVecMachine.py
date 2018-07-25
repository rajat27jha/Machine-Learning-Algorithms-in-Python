# Equation for a Hyperplane: X.W + B (Decision boundary) (W and b are the unknowns)
# Hyperplane for positive class support vector: Xi . W + b = 1 (spe. 'll be its support vector)
# Hyperplane for positive class support vector: Xi . W + b = -1 (spe. 'll be its support vector)
# Formula for classification of classifier: Sign(Xi . W + b)
# Optimisation objective: Minimise:||W||(magnitude)   Maximise: b (bias)
# Class * (Known features*W + b) >= 1 This should satisfy. [y(x*w+b)]
# Lagrangian problem: it is to satisfy above eq. by maintaining optimisation objective
# This is quadratic programming problem. It does not hv a single solution. there is no magical formula
# Convex Problem:

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


class Support_Vector_Machine:
    def __init__(self, visualisation=True):
        self.visualisation = visualisation
        self.colors = {1: 'r', -1: 'b'}
        if self.visualisation:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # fit
    def fit(self, data):

        self.data = data
        # {||w||:[w,b]}
        opt_dict = {}

        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1]
                      [1, -1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001]
        # extremely expensive
        b_range_multiple = 5
        b_multiple = 5

        latest_optimum = self.max_feature_value

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # we can do this because
            optimised = False
            while not optimised:
                # skipping from video 27, continuing from 34 :|
                pass
        pass

    def predict(self, features):
        #  sign (x.b = b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        return classification




data_dict = {-1: np.array([[1, 7]
                          [2, 8]
                          [3, 8]]),
             1: np.array([[5, 1]
                         [6, -1]
                         [7, 3]])}







