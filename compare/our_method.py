import lcs_class
import lcs.optim as opt
import numpy as np
import time


st = time.time()

# load data
data_train = np.load('data_train.npy', allow_pickle=True).item()

n_state = data_train['n_state']
n_control = data_train['n_control']
n_lam = data_train['n_lam']
training_data_size = data_train['N']

x_batch = data_train['X_data']
u_batch = data_train['U_data']
x_next_batch = data_train['Y_data']

# establish the VN learner (violation-based method)
F_stiffness = 1
gamma = 1e-2
epsilon = 1e-2
vn_learner = lcs_class.LCS_VN(n_state=n_state, n_control=n_control, n_lam=n_lam, F_stiffness=F_stiffness)
vn_learner.diff(gamma=gamma, epsilon=epsilon, w_C=1e-6, C_ref=0, w_F=0e-6, F_ref=0)

# establish the optimizer
vn_learning_rate = 1e-3
vn_optimizier = opt.Adam()
vn_optimizier.learning_rate = vn_learning_rate

# training loop
max_iter = 50
mini_batch_size = 100
vn_curr_theta = 0.01 * np.random.randn(vn_learner.n_theta)



for iter in range(max_iter):

    all_indices = np.random.permutation(training_data_size)

    for batch in range(int(np.floor(training_data_size / mini_batch_size))):
        # mini_batch_size
        shuffle_index = all_indices[batch * mini_batch_size:(batch + 1) * mini_batch_size]
        x_mini_batch = x_batch[shuffle_index]
        u_mini_batch = u_batch[shuffle_index]
        x_next_mini_batch = x_next_batch[shuffle_index]

        # do one step for VN
        vn_mean_loss, vn_dtheta, vn_dyn_loss, vn_lcp_loss, _ = vn_learner.step(batch_x=x_mini_batch,
                                                                               batch_u=u_mini_batch,
                                                                               batch_x_next=x_next_mini_batch,
                                                                               current_theta=vn_curr_theta)
        vn_curr_theta = vn_optimizier.step(vn_curr_theta, vn_dtheta)

    print('iter:', iter, 'vn_loss: ', vn_mean_loss)



# --------------------------- testing the prediction error for PN and VN ----------------------------------#

print('#######  '
      'start testing '
      '####### ')

data_test = np.load('data_test.npy', allow_pickle=True).item()
x_test_batch = data_test['X_data']
x_next_test_batch = data_test['Y_data']
u_test_batch = data_test['U_data']

# establish the testing object
tester = lcs_class.LCS_PN(n_state=n_state, n_control=n_control, n_lam=n_lam, F_stiffness=F_stiffness)
tester.diff(w_C=0, C_ref=0, w_F=0, F_ref=0)

vn_pred_error = tester.pred_error(batch_x=x_test_batch, batch_u=u_test_batch,
                                  batch_x_next=x_next_test_batch,
                                  current_theta=vn_curr_theta)

print('Testing finished!')
print('Prediction error for VN:', vn_pred_error)
print('training_time:', time.time() - st)
