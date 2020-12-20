

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
%matplotlib inline

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import sklearn.linear_model

I = np.identity(2)
mean = [4,11]
cov = 0.3*I

print(cov)

[[0.3 0. ]
 [0.  0.3]]

x, y = np.random.multivariate_normal(mean, cov, 100).T

plt.plot(x, y, 'x')
plt.title('Datapoints of the 1st class')
plt.axis('equal')

plt.show()

I = np.identity(2)
mean = [10,3]
cov = I

x, y = np.random.multivariate_normal(mean, cov, 100).T

plt.plot(x, y, '^')
plt.title('Datapoints of the 2nd class')
plt.axis('equal')

plt.show()

mu_vec1 = np.array([4,11])
cov_mat1 = np.array([[0.3,0],[0,0.3]])
x1_samples = np.random.multivariate_normal(mu_vec1, cov_mat1, 100)
mu_vec1 = mu_vec1.reshape(1,2).T # to 1-col vector

mu_vec2 = np.array([10,3])
cov_mat2 = np.array([[1,0],[0,1]])
x2_samples = np.random.multivariate_normal(mu_vec2, cov_mat2, 100)
mu_vec2 = mu_vec2.reshape(1,2).T


fig = plt.figure()


plt.scatter(x1_samples[:,0],x1_samples[:,1], marker='x')
plt.scatter(x2_samples[:,0],x2_samples[:,1], c= 'green', marker='^')

X = np.concatenate((x1_samples,x2_samples), axis = 0)
Y = np.array([0]*100 + [1]*100)

C = 1.0  # SVM regularization parameter
clf = svm.SVC(kernel = 'linear',  gamma=0.7, C=C )
clf.fit(X, Y)
plt.title('1st and 2nd Class Datapoints')

Text(0.5, 1.0, '1st and 2nd Class Datapoints')

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy
from sklearn import svm


mu_vec1 = np.array([4,11])
cov_mat1 = np.array([[0.3,0],[0,0.3]])
x1_samples = np.random.multivariate_normal(mu_vec1, cov_mat1, 100)
mu_vec1 = mu_vec1.reshape(1,2).T # to 1-col vector

mu_vec2 = np.array([10,3])
cov_mat2 = np.array([[1,0],[0,1]])
x2_samples = np.random.multivariate_normal(mu_vec2, cov_mat2, 100)
mu_vec2 = mu_vec2.reshape(1,2).T


fig = plt.figure()


plt.scatter(x1_samples[:,0],x1_samples[:,1], marker='x')
plt.scatter(x2_samples[:,0],x2_samples[:,1], c= 'green', marker='^')

X = np.concatenate((x1_samples,x2_samples), axis = 0)
Y = np.array([0]*100 + [1]*100)

C = 1.0  # SVM regularization parameter
clf = svm.SVC(kernel = 'linear',  gamma=0.7, C=C )
clf.fit(X, Y)

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(0, 15)
yy = a * xx - (clf.intercept_[0]) / w[1]

plt.title('Decision Boundary')
plt.plot(xx, yy, 'k-')

[<matplotlib.lines.Line2D at 0x1ba2d89e490>]

 

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import sklearn.linear_model

clf = sklearn.linear_model.LogisticRegression()
clf.fit(X, Y)

b = clf.intercept_[0]
w1, w2 = clf.coef_.T

c = -b/w2
m = -w1/w2

xmin, xmax = 0, 15
ymin, ymax = 0, 15
xd = np.array([xmin, xmax])
yd = m*xd + c
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

plt.scatter(*X[Y==0].T, s=8, alpha=0.5)
plt.scatter(*X[Y==1].T, s=8, alpha=0.5)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.ylabel(r'$x_2$')
plt.xlabel(r'$x_1$')

plt.show()

from numpy import where
from numpy import meshgrid
from numpy import arange
from numpy import hstack



min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
# define the x and y scale
x1grid = arange(min1, max1, 0.1)
x2grid = arange(min2, max2, 0.1)
# create all of the lines and rows of the grid
xx, yy = meshgrid(x1grid, x2grid)
# flatten each grid to a vector
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
# horizontal stack vectors to create x1,x2 input for the model
grid = hstack((r1,r2))
# define the model
model = LogisticRegression()
# fit the model
model.fit(X, Y2)
# make predictions for the grid
yhat = model.predict(grid)
# reshape the predictions back into a grid
zz = yhat.reshape(xx.shape)
# plot the grid of x, y and z values as a surface
pyplot.contourf(xx, yy, zz, cmap='Paired')
# create scatter plot for samples from each class
for class_value in range(2):
	# get row indexes for samples with this class
	row_ix = where(y == class_value)
	# create scatter of these samples
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')
# show the plot
pyplot.show()

 

 

