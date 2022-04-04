import time

import lcs_class
import numpy as np
from casadi import *
import lcs.optim as opt

n_state = 4
n_control = 2
n_lam = 16
gen_stiffness = 1
training_data_size = 5000

print('------------------------------------------------')

# generate the lcs system
lcs_mats = lcs_class.gen_lcs(n_state, n_control, n_lam, gen_stiffness)

# generate the training data
training_data = lcs_class.gen_data(lcs_mats, training_data_size, noise_level=1e-2)

print('------------------------------------------------')
X_train = training_data['x_batch']
U_train = training_data['u_batch']
Y_train = training_data['x_next_batch']
K_train = len(training_data['unique_mode_list'])
N_train = X_train.shape[0]

print('the number of training categories: ', K_train)

data = dict(X_data=X_train,
            Y_data=Y_train,
            U_data=U_train,
            n_state=n_state,
            n_control=n_control,
            n_lam=n_lam,
            K=K_train,
            N=N_train)

np.save('data_train.npy', data)

# generate testing data
testing_data_size = 1000
testing_data = lcs_class.gen_data(lcs_mats, testing_data_size, noise_level=0.0)
print('------------------------------------------------')
X_test = testing_data['x_batch']
U_test = testing_data['u_batch']
Y_test = testing_data['x_next_batch']
K_test = len(testing_data['unique_mode_list'])
N_test = X_test.shape[0]

print('the number of testing categories: ', K_test)

data = dict(X_data=X_test,
            Y_data=Y_test,
            U_data=U_test,
            n_state=n_state,
            n_control=n_control,
            n_lam=n_lam,
            K=K_test,
            N=N_test)

np.save('data_test.npy', data)
