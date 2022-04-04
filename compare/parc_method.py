# -*- coding: utf-8 -*-
"""
Example of PWA regression to fit a nonlinear function.

(C) 2021 A. Bemporad
"""

import numpy as np
from sklearn.model_selection import train_test_split
from parc.parc import PARC
import warnings
import time

warnings.filterwarnings("ignore", category=FutureWarning)



st = time.time()



separation='Voronoi'
sigma = 1
alpha = 1.0e-3
beta = 1.0e-3
maxiter = 15

data_train = np.load('data_train.npy', allow_pickle=True).item()
X_train = np.hstack((data_train['X_data'], data_train['U_data']))
Y_train = data_train['Y_data']

N_train = data_train['N']
K_train = data_train['K']

data_test = np.load('data_test.npy', allow_pickle=True).item()
X_test = np.hstack((data_test['X_data'], data_test['U_data']))
Y_test = data_test['Y_data']
N_test = data_test['N']
K_test = data_test['K']

nyc = Y_train.shape[1]

categorical = nyc * [False]

predictor = PARC(K=K_train, alpha=alpha, sigma=sigma, separation=separation, maxiter=maxiter,
                 cost_tol=1e-4, min_number=10, fit_on_partition=True,
                 beta=beta, verbose=1)


predictor.fit(X_train, Y_train, categorical, weights=np.ones(nyc))

score_train = predictor.score(X_train, Y_train)  # compute R2 score on training data
score_test = predictor.score(X_test, Y_test)  # compute R2 score on test data

print("\nResults:\n")
print("Training data:", score_train)
print("Test data:", score_test)
print('K:', K_train)
print('training_time:', time.time() - st)
print("--------------------\n")
