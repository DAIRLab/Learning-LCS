import time

from lcs import lcs_learning
import numpy as np
from casadi import *
import lcs.optim as opt

large_value = 100000


def solve_MIQP(x_data, u_data, x_next_data, n_lam):
    n_data = x_data.shape[0]
    n_x = x_data.shape[1]
    n_u = u_data.shape[1]
    n_lam = n_lam

    A = SX.sym('A', n_x, n_x)
    B = SX.sym('B', n_x, n_u)
    C = SX.sym('C', n_x, n_lam)
    dyn_offset = SX.sym('dyn_offset', n_x)
    D = SX.sym('D', n_lam, n_x)
    E = SX.sym('E', n_lam, n_u)
    F = SX.sym('F', n_lam, n_lam)
    lcp_offset = SX.sym('lcp_offset', n_lam)

    theta = vertcat(vec(A), vec(B), vec(C), vec(dyn_offset), vec(D), vec(E), vec(F), vec(lcp_offset))

    J = 0
    w = [theta]
    g = []
    discrete = theta.numel() * [False]

    for k in range(n_data):
        # define lambda
        lam_k = SX.sym('lam_'+str(k), n_lam)
        w += [lam_k]
        discrete += n_lam * [False]

        # define delta
        delta_k = SX.sym('delta_'+str(k), n_lam)
        w += [delta_k]
        discrete += n_lam * [False]

        #


    print('hahah')


# --------------------------- generating the system and data----------------------------------#
# generate the lcs system
n_state = 4
n_control = 2
n_lam = 2
gen_stiffness = 0
lcs_mats = lcs_learning.gen_lcs(n_state, n_control, n_lam, gen_stiffness)
# lcs_mats = np.load('sys_matrix.npy', allow_pickle=True).item()
min_sig = lcs_mats['min_sig']
true_theta = lcs_mats['theta']
