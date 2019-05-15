# we'll use matplotlib for graphs
import matplotlib.pyplot as plt
# x, y coordinates as a list
X = [1, 2, 3, 4]
Y = [2, 4, 6, 8]

plt.plot(X, Y, '--', color='red')
plt.title('This is a sample plot')
plt.xlabel('No. of Workshops')
plt.ylabel('No. of Students')
plt.show()
