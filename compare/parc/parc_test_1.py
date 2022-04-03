# -*- coding: utf-8 -*-
"""
Example of PWA regression to fit a nonlinear function.

(C) 2021 A. Bemporad
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split
from parc import PARC

np.random.seed(0)  # for reproducibility

K = 10
separation = 'Softmax'
# separation='Voronoi'
sigma = 1
alpha = 1.0e-3
beta = 1.0e-3
softmax_maxiter = 100000
maxiter = 15

N = 1000
test_size = 0.2

nx = 2
xmin = 0
xmax = 1
ymin = 0
ymax = 1
X = np.random.rand(N, nx) * np.array([xmax - xmin, ymax - ymin]) + np.array([xmin, ymin])

nyc = 1  # number of numeric outputs
noise_frac = 0.0  # noise standard deviation (0 = no noise)

f = lambda x1, x2: np.sin(4 * x1 - 5 * (x2 - 0.5) ** 2) + 2 * x2
Y = f(X[:, 0], X[:, 1]) + noise_frac * np.random.randn(N)
categorical = False
nlevels = 8

plt.close('all')
fig, ax = plt.subplots(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.grid()

dx = (xmax - xmin) / 100.0
dy = (ymax - ymin) / 100.0
[x1, x2] = np.meshgrid(np.arange(xmin, xmax + dx, dx), np.arange(ymin, ymax + dy, dy))
z = f(x1, x2)
plt.contourf(x1, x2, z, alpha=0.6, levels=nlevels)
plt.contour(x1, x2, z, linewidths=3.0, levels=nlevels)
plt.title('level sets of y(x)', fontsize=20)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

# Get random split of training/test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

predictor = PARC(K=K, alpha=alpha, sigma=sigma, separation=separation, maxiter=maxiter,
                 cost_tol=1e-4, min_number=10, fit_on_partition=True,
                 beta=beta, verbose=1)

# Y_hat, delta_hat = predictor.predict(X_test) # predict targets

predictor.fit(X_train, Y_train, categorical, weights=np.ones(1))

score_train = predictor.score(X_train, Y_train)  # compute R2 score on training data
score_test = predictor.score(X_test, Y_test)  # compute R2 score on test data

print("\nResults:\n")
print("Training data: %6.2f %%" % (score_train * 100))
print("Test data:     %6.2f %%" % (score_test * 100))
print("--------------------\n")

Kf = predictor.K  # final number of partitions
delta = predictor.delta  # final assignment of training points to clusters
xbar = predictor.xbar  # centroids of final clusters

# Plot resulting PWA function
zpwl, _ = predictor.predict(np.hstack((x1.reshape(x1.size, 1), x2.reshape(x2.size, 1))))
zpwl = zpwl.reshape(x1.shape)

# plot level sets of PWA function
fig, ax = plt.subplots(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.grid()
ax.set_xlim([xmin, xmax])
ax.set_ylim([0, 1])
NN = x1.shape[0]
plt.contourf(x1, x2, zpwl, alpha=0.6, levels=nlevels)
plt.contour(x1, x2, zpwl, linewidths=3.0, levels=nlevels)
plt.title('PARC (K = %d)' % K, fontsize=20)

Yhtrain, _ = predictor.predict(X_train)
Yhtest, delta_test = predictor.predict(X_test)

fig, ax = plt.subplots(figsize=(8, 8))
for i in range(0, Kf):
    iD = (delta == i).ravel()
    plt.scatter(X_train[iD, 0], X_train[iD, 1], marker='*', linewidth=3,
                alpha=0.5, color=cm.tab10(i))
plt.grid()
plt.scatter(xbar[:, 0], xbar[:, 1], marker='o', linewidth=5, alpha=.5, color=(.8, .4, .4))

# Plot PWL partition
predictor.plot_partition([xmin, ymin], [xmax, ymax], fontsize=32,
                         ax=ax, alpha=.6, linestyle='-', linewidth=2.0, color=(1, 1, 1))
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

plt.title('PARC (K = %d)' % K, fontsize=20)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel(r'$x_1$', labelpad=15, fontsize=20)
ax.set_ylabel(r'$x_2$', labelpad=15, fontsize=20)
ax.set_zlabel(r'$y$', fontsize=20)
ax.scatter(X_test[:, 0], X_test[:, 1], Y_test, alpha=0.5)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
ax.plot_surface(x1, x2, z, alpha=0.5)
ax.view_init(35, -120)
plt.title('Nonlinear function', fontsize=20)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel(r'$x_1$', labelpad=15, fontsize=20)
ax.set_ylabel(r'$x_2$', labelpad=15, fontsize=20)
ax.set_zlabel(r'$y$', fontsize=20)
for i in range(0, Kf):
    iD = (delta_test == i).ravel()
    ax.scatter(X_test[iD, 0], X_test[iD, 1], Y_test[iD], marker='*',
               linewidth=3, alpha=0.5, color=cm.tab10(i))

# plot PWA function
ax.plot_surface(x1, x2, zpwl, alpha=0.5)
ax.view_init(35, -120)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.title('PARC (K = %d)' % K, fontsize=20)

#######################
# MIP Optimization
#######################
solveMIP = False
if solveMIP:
    Xmin = np.min(X, axis=0)
    Xmax = np.max(X, axis=0)
    yref = 3. * np.ones(nyc)  # desired target
    verbose = False
    solver = "CBC"
    # solver="GRB"
    x, y, region, f = predictor.optimize(Xmin, Xmax, yref, verbose=verbose, solver=solver)
    ax.scatter(x[0], x[1], y, marker='o', linewidth=10.0, color=(1, 0, 0))
    print("yhat = %5.4f, x1 = %5.4f, x2 = %5.4f" % (y, x[0], x[1]))


########################
# K-fold cross validation
########################
runCrossValidation = False
if runCrossValidation:
    bestK, results = predictor.cross_val(X_train, Y_train, categorical, Kfolds=5, Ks=[5, 10, 15])