import numpy as np
import matplotlib.pyplot as plt

load = np.load('results.npy', allow_pickle=True).item()
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
avg_vn_dyn_loss = []
avg_vn_lcp_loss = []
std_vn_dyn_loss = []
std_vn_lcp_loss = []
# load the results list
results_list = load['meta_para_trials']
for i_meta_para in range(len(results_list)):

    meta_para = meta_para_list[i_meta_para]
    # current meta parameter trials
    trials_list = results_list[i_meta_para]

    # enumerate for each trials
    vn_training_dyn_loss = []
    vn_training_lcp_loss = []

    n_theta = []
    for i_trial in range(len(trials_list)):
        vn_training_dyn_loss += [trials_list[i_trial]['vn_training_dyn_loss']]
        vn_training_lcp_loss += [trials_list[i_trial]['vn_training_lcp_loss']]


    # compute the statistics
    avg_vn_dyn_loss += [np.array(vn_training_dyn_loss).mean()]
    avg_vn_lcp_loss += [np.array(vn_training_lcp_loss).mean()]
    std_vn_dyn_loss += [np.array(vn_training_dyn_loss).std()]
    std_vn_lcp_loss += [np.array(vn_training_lcp_loss).std()]


print('------')
print('avg_vn_dyn_loss:', avg_vn_dyn_loss)
print('avg_vn_lcp_loss:', avg_vn_lcp_loss)
print('std_vn_dyn_loss:', std_vn_dyn_loss)
print('std_vn_lcp_loss:', std_vn_lcp_loss)


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
x = np.arange(meta_para_list.size)
plt.errorbar(x, avg_vn_dyn_loss, yerr=std_vn_dyn_loss, label='dyn violation', lw=4, marker='o',
             markersize=7, capsize=3, elinewidth=2, color='tab:red')
plt.errorbar(x + 0.05, avg_vn_lcp_loss, yerr=std_vn_lcp_loss, label='lcp violation', lw=4, marker='o',
             markersize=7, capsize=3, elinewidth=2, color='tab:green')
# # plt.plot(x, vn_lam, label='Violation-based', lw=3, marker='o', markersize=7)
label = ['10', '$5$', '$1$', '$0.5$', '$0.1$', '$10^{-2}$', '$10^{-3}$', '$10^{-4}$', '$10^{-5}$']
plt.xticks(x, labels=label)
plt.xlabel('$\epsilon$', labelpad=15)
plt.ylabel('violation loss in training', fontsize=20)
plt.grid()
plt.legend(loc='lower left')
plt.yscale('log')
#
plt.tight_layout()
plt.show()
