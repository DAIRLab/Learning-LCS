import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------
params = {'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 18,
          'ytick.labelsize': 20,
          'legend.fontsize': 18}
plt.rcParams.update(params)

# set width of bar
barWidth = 0.25

# set height of bar
K = [16, 32, 60, 105, 267, 1074, 2021]
ours = [9, 11, 14, 16, 22, 48, 59]
comp = [ 32, 73, 120, 227, 534, 2340, np.inf]
accuracy_ours = [0.0049, 0.0038, 0.0043, 0.0022, 0.003, 0.004, 0.003]
accuracy_comp = [ 0.0046, 0.010, 0.0036, 0.0016, 0.003, 0.006, np.inf]

# Set position of bar on X axis
br1 = np.arange(len(ours))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

fig, axs = plt.subplots(2, sharex=True)

# Adding Xticks

axs[0].plot(accuracy_ours, color='tab:red', marker='o', linewidth=3, label='Ours (violation)')
axs[0].plot(accuracy_comp, color='tab:green', marker='o', linewidth=3, label='PARC')
# axs[0].set_yscale('log')
axs[0].set_ylabel(r'$e_{test}$')
axs[0].legend()
axs[0].grid()

# Make the plot
axs[1].bar(br1, ours, color='tab:red', width=barWidth,
           edgecolor='grey', label='Ours (violation)')
axs[1].bar(br2, comp, color='tab:green', width=barWidth,
           edgecolor='grey', label='PARC')

axs[1].set_xlabel('Number of modes (partitions)')
axs[1].set_ylabel('Training time [s]')
axs[1].set_xticks([r + barWidth for r in range(len(ours))],
                  [ '16', '32', '60', '105', '267', '1074', '2021'])
axs[1].set_yscale('log')
axs[1].legend()

plt.grid()
plt.tight_layout()
plt.show()
