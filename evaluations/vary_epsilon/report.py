import numpy as np
import matplotlib.pyplot as plt

load = np.load('results.npy', allow_pickle=True).item()
load2 = np.load('results2.npy', allow_pickle=True).item()
hyper_parameters = load['hyper_parameters']
print('-----')
print('lam dim:', hyper_parameters['n_lam'],
      '  control dim:', hyper_parameters['n_control'],
      '  state dim:', hyper_parameters['n_state'],
      )
print('-----')
meta_para_list = np.array(load['meta_para_list'])
print('list of state dim:', meta_para_list)
print('-----')
print('gamma:', hyper_parameters['gamma'])
print('epsilon:', hyper_parameters['epsilon'])
print('-----')
print('noise_levels:', hyper_parameters['training_data_noise_level'])

# storage
avg_pn_test_error = []
std_pn_test_error = []
avg_vn_test_error = []
std_vn_test_error = []
avg_data_stiffness = []
std_data_stiffness = []
avg_training_data_size = []
avg_testing_data_size = []
avg_training_mode_count = []
std_training_mode_count = []
avg_n_theta=[]
# load the results list
results_list = load['meta_para_trials']
results_list2 = load2['meta_para_trials']
for i_meta_para in range(len(results_list)):

    meta_para = meta_para_list[i_meta_para]
    # current meta parameter trials
    trials_list = results_list[i_meta_para]
    trials_list2 = results_list2[i_meta_para]

    # enumerate for each trials
    pn_test_error = []
    vn_test_error = []
    data_stiffness = []
    training_data_size = []
    testing_data_size = []
    training_mode_count = []
    n_theta=[]
    for i_trial in range(len(trials_list)):
        pn_test_error += [trials_list[i_trial]['pn_prediction_error']]
        vn_test_error += [trials_list[i_trial]['vn_prediction_error']]
        data_stiffness += [trials_list[i_trial]['min_sig_list']]
        training_data_size += [trials_list[i_trial]['training_data_size']]
        testing_data_size += [trials_list[i_trial]['testing_data_size']]
        training_mode_count += [trials_list[i_trial]['training_data_mode_percentage'] * (2 ** hyper_parameters['n_lam'])]
        n_theta+=[trials_list[i_trial]['vn_training_theta'].size]
    for i_trial in range(len(trials_list2)):
        pn_test_error += [trials_list2[i_trial]['pn_prediction_error']]
        vn_test_error += [trials_list2[i_trial]['vn_prediction_error']]
        data_stiffness += [trials_list2[i_trial]['min_sig_list']]
        training_data_size += [trials_list2[i_trial]['training_data_size']]
        testing_data_size += [trials_list2[i_trial]['testing_data_size']]
        training_mode_count += [trials_list2[i_trial]['training_data_mode_percentage'] * (2 ** hyper_parameters['n_lam'])]
        n_theta+=[trials_list2[i_trial]['vn_training_theta'].size]

    print('size of pn_test_error:', len(pn_test_error))



    # compute the statistics
    avg_pn_test_error += [np.array(pn_test_error).mean()]
    avg_vn_test_error += [np.array(vn_test_error).mean()]
    std_pn_test_error += [np.array(pn_test_error).std()]
    std_vn_test_error += [np.array(vn_test_error).std()]
    avg_data_stiffness += [np.array(data_stiffness).mean()]
    std_data_stiffness += [np.array(data_stiffness).std()]
    avg_training_data_size += [np.array(training_data_size).mean()]
    avg_testing_data_size += [np.array(testing_data_size).mean()]
    avg_training_mode_count += [np.array(training_mode_count).mean()]
    std_training_mode_count += [np.array(training_mode_count).std()]
    avg_n_theta+=[np.array(n_theta).mean()]


print('------')
print('avg_pn_test_error:', avg_pn_test_error)
print('avg_vn_test_error:', avg_vn_test_error)
print('avg_mode_count:', avg_training_mode_count)


print('------')
print('avg_testing_data_size:', avg_testing_data_size)

print('------')
print('avg_n_theta:', avg_n_theta)

# -----------------------------------------------
params = {'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 18,
          'ytick.labelsize': 20,
          'legend.fontsize': 18}
plt.rcParams.update(params)

plt.figure()
# x = np.arange(len(meta_para_list))
# # plt.plot(x, pn_lam, label='Prediction-based', lw=3, marker='o', markersize=7)
# # plt.fill_between(x, pn_lam_mean-pn_lam_std, pn_lam_mean+pn_lam_std, color='tab:blue', alpha=0.4)
x=np.arange(meta_para_list.size)
plt.errorbar(x, avg_pn_test_error, yerr=std_pn_test_error, label='Prediction', lw=4, marker='o',
             markersize=7, capsize=3, elinewidth=2)
plt.errorbar(x+0.05, avg_vn_test_error, yerr=std_vn_test_error, label='Violation', lw=4, marker='o',
             markersize=7, capsize=3, elinewidth=2)
# # plt.plot(x, vn_lam, label='Violation-based', lw=3, marker='o', markersize=7)
label = ['10', '$5$', '$1$', '$0.5$', '$0.1$', '$10^{-2}$', '$10^{-3}$', '$10^{-4}$', '$10^{-5}$']
plt.xticks(x,labels=label)
plt.xlabel('$\epsilon$', labelpad=15)
plt.ylabel(r'$e_{test}$', fontsize=25)
plt.grid()
plt.legend(loc='upper right')
plt.yscale('log')
#
plt.tight_layout()
plt.show()
