import time

from lcs import lcs_learning
import numpy as np
from casadi import *
import lcs.optim as opt


# loop for trials
n_state = 2
n_control = 4
n_lam = 10
gen_stiffness = 0.5

training_data_size = 50000
training_data_noise_level=1e-2
testing_data_size = 1000


max_iter = 10000
max_trial = 15
mini_batch_size = 200
init_magnitude = 0.1

# algorithm phyer parameter
gamma = 1e-2
epsilon = 1e-4
F_stiffness = 1.0
pn_learning_rate = 1e-3
vn_learning_rate = 5e-3

hyper_parameters = {'n_state': n_state,
                    'n_control': n_control,
                    'n_lam': n_lam,
                    'gen_stiffness': gen_stiffness,
                    'training_data_size': training_data_size,
                    'training_data_noise_level': training_data_noise_level,
                    'testing_data_size': testing_data_size,
                    'max_iter': max_iter,
                    'max_trial': max_trial,
                    'mini_batch_size': mini_batch_size,
                    'init_magnitude': init_magnitude,
                    'gamma': gamma,
                    'epsilon': epsilon,
                    'F_stiffness': F_stiffness,
                    'pn_learning_rate': pn_learning_rate,
                    'vn_learning_rate': vn_learning_rate,
                    }

# storage parameter
meta_para_list = [2, 8, 16, 32, 64, 128]
meta_para_trials = []
for meta_para in meta_para_list:

    # establish the PN learner (prediction-based method)
    pn_learner = lcs_learning.LCS_PN(n_state=meta_para, n_control=n_control, n_lam=n_lam, F_stiffness=F_stiffness)
    pn_learner.diff(w_C=1e-6, C_ref=0, w_F=0e-6, F_ref=0)
    pn_optimizier = opt.Adam()
    pn_optimizier.learning_rate = pn_learning_rate

    # establish the VN learner (violation-based method)
    vn_learner = lcs_learning.LCS_VN(n_state=meta_para, n_control=n_control, n_lam=n_lam, F_stiffness=F_stiffness)
    vn_learner.diff(gamma=gamma, epsilon=epsilon, w_C=1e-6, C_ref=0, w_F=0e-6, F_ref=0)
    vn_optimizier = opt.Adam()
    vn_optimizier.learning_rate = vn_learning_rate

    # establish the testing object
    tester = lcs_learning.LCS_PN(n_state=meta_para, n_control=n_control, n_lam=n_lam, F_stiffness=F_stiffness)
    tester.diff(w_C=0, C_ref=0, w_F=0, F_ref=0)

    # storage
    trial_result_trace = []
    for n_trial in range(max_trial):

        trial_result = {}
        print('####### '
              'current trial system and data generation '
              '####### ')

        # --------------------------- generating the system and data----------------------------------#
        # generate the lcs system
        lcs_mats = lcs_learning.gen_lcs(meta_para, n_control, n_lam, gen_stiffness)
        # lcs_mats = np.load('sys_matrix.npy', allow_pickle=True).item()
        min_sig = lcs_mats['min_sig']
        true_theta = lcs_mats['theta']
        trial_result['true_theta'] = true_theta

        print('------------------------------------------------')
        print('LCS generated!')
        print('n_state:', lcs_mats['n_state'], 'n_control:', lcs_mats['n_control'], 'n_lam:', lcs_mats['n_lam'])
        print('min_sig_lis', min_sig)
        trial_result['min_sig_list'] = min_sig

        # generate the training data
        training_data = lcs_learning.gen_data(lcs_mats, training_data_size, noise_level=training_data_noise_level)
        print('------------------------------------------------')
        print('Training data generated!')
        print('Training_data_size:', training_data_size)
        print('Mode percentage:', training_data['mode_percentage'])
        trial_result['training_data_size'] = training_data_size
        trial_result['training_data_mode_percentage'] = training_data['mode_percentage']

        # generate testing data
        testing_data = lcs_learning.gen_data(lcs_mats, testing_data_size, noise_level=0.0)
        print('------------------------------------------------')
        print('Testing data generated!')
        print('Testing_data_size:', testing_data_size)
        trial_result['testing_data_size'] = testing_data_size
        trial_result['testing_data_mode_percentage'] = testing_data['mode_percentage']
        print('')

        # --------------------------- training PN and VN ----------------------------------#
        print('###### '
              'current trial start training process '
              '###### ')

        # run training using PN and VN using mini_batch size
        pn_curr_theta = init_magnitude * np.random.randn(true_theta.size)
        vn_curr_theta = init_magnitude * np.random.randn(true_theta.size)
        pn_loss_trace = []
        pn_theta_trace = []
        vn_loss_trace = []
        vn_theta_trace = []
        # load data
        x_batch = training_data['x_batch']
        u_batch = training_data['u_batch']
        x_next_batch = training_data['x_next_batch']
        for iter in range(max_iter):
            # mini_batch_size
            shuffle_index = np.random.permutation(training_data_size)[0:mini_batch_size]
            x_mini_batch = x_batch[shuffle_index]
            u_mini_batch = u_batch[shuffle_index]
            x_next_mini_batch = x_next_batch[shuffle_index]

            # do one step size for PN
            pn_mean_loss, pn_dtheta, _ = pn_learner.step(batch_x=x_mini_batch, batch_u=u_mini_batch,
                                                         batch_x_next=x_next_mini_batch,
                                                         current_theta=pn_curr_theta)
            # store and update
            pn_loss_trace += [pn_mean_loss]
            pn_theta_trace += [pn_curr_theta]
            pn_curr_theta = pn_optimizier.step(pn_curr_theta, pn_dtheta)

            # do one step for VN
            vn_mean_loss, vn_dtheta, vn_dyn_loss, vn_lcp_loss, _ = vn_learner.step(batch_x=x_mini_batch,
                                                                                   batch_u=u_mini_batch,
                                                                                   batch_x_next=x_next_mini_batch,
                                                                                   current_theta=vn_curr_theta)
            # store
            vn_loss_trace += [vn_mean_loss]
            vn_theta_trace += [vn_curr_theta]
            vn_curr_theta = vn_optimizier.step(vn_curr_theta, vn_dtheta)

            if iter % 100 == 0:
                print('dimension of state:', meta_para,
                      '|| trial #:', n_trial,
                      '|| iter', iter,
                      '|| PN_loss:', pn_mean_loss,
                      '|| VN_loss:', vn_mean_loss,
                      '| VN_dyn_loss:', vn_dyn_loss,
                      '| VN_lcp_loss:', vn_lcp_loss,
                      )

        print('------------------------------------------------')
        print('Training finished!')
        print('Training loss for PN:', pn_loss_trace[-1])
        print('Training loss for VN:', vn_loss_trace[-1])
        trial_result['pn_training_loss'] = pn_loss_trace[-1]
        trial_result['pn_training_theta'] = pn_theta_trace[-1]
        trial_result['vn_training_loss'] = vn_loss_trace[-1]
        trial_result['vn_training_theta'] = vn_theta_trace[-1]
        print(' ')

        # --------------------------- testing the prediction error for PN and VN ----------------------------------#

        print('#######  '
              'start testing '
              '####### ')

        # testing the prediction error
        x_test_batch = testing_data['x_batch']
        u_test_batch = testing_data['u_batch']
        x_next_test_batch = testing_data['x_next_batch']

        pn_pred_error = tester.pred_error(batch_x=x_test_batch, batch_u=u_test_batch,
                                          batch_x_next=x_next_test_batch,
                                          current_theta=pn_theta_trace[-1])

        vn_pred_error = tester.pred_error(batch_x=x_test_batch, batch_u=u_test_batch,
                                          batch_x_next=x_next_test_batch,
                                          current_theta=vn_theta_trace[-1])

        print('------------------------------------------------')
        print('Testing finished!')
        print('Prediction error for PN:', pn_pred_error)
        print('Prediction error for VN:', vn_pred_error)
        trial_result['pn_prediction_error'] = pn_pred_error
        trial_result['vn_prediction_error'] = vn_pred_error
        print(' ')
        print(' ')

        trial_result_trace += [trial_result]

    print('****************************** '
          'summary of all trials'
          '****************************** ')
    pn_pred_error_trial = []
    vn_pred_error_trial = []
    for trial_result in trial_result_trace:
        pn_pred_error_trial += [trial_result['pn_prediction_error']]
        vn_pred_error_trial += [trial_result['vn_prediction_error']]

    print('current epsilon:', meta_para)
    print('Mean prediction error for PN cross trials: ', np.array(pn_pred_error_trial).mean())
    print('STD prediction error for VN cross trials: ', np.array(pn_pred_error_trial).std())
    print('Mean prediction error for PN cross trials: ', np.array(vn_pred_error_trial).mean())
    print('STD prediction error for vN cross trials: ', np.array(vn_pred_error_trial).std())

    meta_para_trials += [trial_result_trace]

# save
np.save('results',
        {
            'meta_para_list': meta_para_list,
            'meta_para_trials': meta_para_trials,
            'hyper_parameters': hyper_parameters,
        }
        )
