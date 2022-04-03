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
fig = plt.subplots()

# set height of bar
K=[4,8, 16, 32, 63]
ours = [6,  9, 10, 12, 15]
comp = [17, 30, 75, 113, 301]

# Set position of bar on X axis
br1 = np.arange(len(ours))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot
plt.bar(br1, ours, color='tab:red', width=barWidth,
        edgecolor='grey', label='Ours (violation)')
plt.bar(br2, comp, color='tab:green', width=barWidth,
        edgecolor='grey', label='PARC')
# plt.yscale('log')
plt.grid()


# Adding Xticks
plt.xlabel('Number of modes (partitions)')
plt.ylabel('Training time [sec]')
plt.xticks([r + barWidth for r in range(len(ours))],
           ['4', '8', '16', '32', '63'])

plt.legend()
plt.tight_layout()
plt.show()
