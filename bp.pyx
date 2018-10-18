#cython: boundscheck=False
#cython: wraparound=False

# A cython script that does belief propagation for toric code with only Pauli X error.
# See appendix of the paper for the description.
# This script can be compiled by bp_compile.py
# It is not necessary to use cython. But even the cython version is quite slow.
# One reason is currently it only use 1 core of CPU.

import numpy as np
cimport numpy as np

e_all_np = np.unpackbits(np.arange(4096, dtype=np.uint16).reshape((4096, 1)).view(np.uint8), axis=1)
e_all_np = e_all_np[:, [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]]
cdef np.uint8_t [:,:] e_all = e_all_np.copy()
# setting up a matrix contains all length-12 string with 0,1 (i.e. binary of number 0-4095)

c = np.zeros((4, 12), dtype=np.int8)
c[0, [0, 2, 6, 8]] = 1
c[1, [1, 3, 8, 10]] = 1
c[2, [2, 4, 7, 9]] = 1
c[3, [3, 5, 9, 11]] = 1
c = c.transpose()
# setting up the check matrix for computing 4 parity checks in each unit cell.
# Layout of unit cell:
#    0     1
# 6 (0) 8 (1) 10
#    2     3
# 7 (2) 9 (3) 11
#    4     5

# 0-11 are qubits, (0-3) are the parity checks.

cpdef np.float64_t [:,:] cell_inf_vp(np.int8_t [:] cell_synd, np.float64_t [:,:] boundary_state,
                                    np.float64_t [:,:] og_boundary_state, np.float64_t [:] inner_state):
    """
    inference inside a cell
    :param cell_synd: 4 parity check results, see above for layout
    :param boundary_state: incoming messages. See code below for ordering
    :param og_boundary_state: joint probability distribution for each 2-qubit boundary.
    :param inner_state: error rates for qubits not on the boundary
    :return: all messages this unit cell sends out.
    """

    # defining variables
    cdef Py_ssize_t i
    cdef int n_edge, s_edge, w_edge, e_edge, weight_otherqbit
    cdef np.float64_t p_n, p_s, p_w, p_e, total_p, p_inner
    cdef np.uint8_t[:] error

    new_boundary_state_np = np.zeros_like(boundary_state, dtype=np.float64)
    cdef np.float64_t[:, :] new_boundary_state = new_boundary_state_np

    # computing error configurations satisfy the syndrome. (solving linear equation route likely faster...)
    all_possible_synd = np.matmul(e_all, c)
    all_possible_synd %= 2
    matching_errors_idx = np.nonzero(np.all(all_possible_synd == np.array(cell_synd), axis=1))[0]

    # Go through all matching error configuration to compute outgoing messages
    # which is basically some marginal probability distribution
    for i in range(matching_errors_idx.size):
        error = e_all[matching_errors_idx[i], :]

        n_edge = error[0] + error[1] * 2
        s_edge = error[4] + error[5] * 2
        w_edge = error[6] + error[7] * 2
        e_edge = error[10] + error[11] * 2

        p_n = boundary_state[0, n_edge]
        p_s = boundary_state[1, s_edge]
        p_w = boundary_state[2, w_edge]
        p_e = boundary_state[3, e_edge]

        p_inner = inner_state[0]**error[2] * inner_state[1]**error[3] * \
                  inner_state[2]**error[8] * inner_state[3]**error[9]

        total_p = p_e * p_n * p_s * p_w * p_inner
        new_boundary_state[0, n_edge] += total_p / p_n * og_boundary_state[0,n_edge]
        new_boundary_state[1, s_edge] += total_p / p_s * og_boundary_state[1,s_edge]
        new_boundary_state[2, w_edge] += total_p / p_w * og_boundary_state[2,w_edge]
        new_boundary_state[3, e_edge] += total_p / p_e * og_boundary_state[3,e_edge]

    # normalize the outgoing messages
    new_boundary_state_np /= np.sum(new_boundary_state_np, axis=1, keepdims=True)

    return new_boundary_state


def bp_vp(np.int8_t[:,:] synd, np.float64_t[:,:,:] edge_prior, int bp_steps):
    cdef int latt_size = 16
    cdef int half_latt_size = latt_size//2
    cdef Py_ssize_t i, j, k, i_n, i_s, j_w, j_e, e1, e2, i2_period, j2_period
    cdef double prob_string, prob_nostring

    cdef np.float64_t[:,:,:,:] boundary_state = np.ones((half_latt_size, half_latt_size, 4, 4))
    cdef np.float64_t[:,:,:,:] og_boundary_state = np.ones((half_latt_size, half_latt_size, 4, 4))
    cdef np.float64_t[:,:,:] cell_inner_state = np.ones((half_latt_size, half_latt_size, 4))
    for i in range(half_latt_size):
        for j in range(half_latt_size):
            i2_period = (2*i + 2) % latt_size
            j2_period = (2*j + 2) % latt_size
            for e1 in range(2):
                for e2 in range(2):
                    # in this setting, boundary_state with e1=e2=0 has value 1
                    boundary_state[i,j,0,e1+e2*2] = edge_prior[2*i, 2*j, 1]**e1 * edge_prior[2*i, 2*j + 1, 1]**e2
                    boundary_state[i,j,2,e1+e2*2] = edge_prior[2*i, 2*j, 0]**e1 * edge_prior[2*i+1, 2*j, 0]**e2
                    boundary_state[i,j,1,e1+e2*2] = edge_prior[i2_period, 2*j, 1]**e1 * edge_prior[i2_period, 2*j + 1, 1]**e2
                    boundary_state[i,j,3,e1+e2*2] = edge_prior[2*i, j2_period, 0]**e1 * edge_prior[2*i + 1, j2_period, 0]**e2

            cell_inner_state[i, j, 0] =edge_prior[2*i+1, 2*j, 1]
            cell_inner_state[i, j, 1] =edge_prior[2*i+1, 2*j+1, 1]
            cell_inner_state[i, j, 2] =edge_prior[2*i, 2*j+1, 0]
            cell_inner_state[i, j, 3] =edge_prior[2*i+1, 2*j+1, 0]

    og_boundary_state[...] = boundary_state
    cdef np.float64_t[:,:,:,:] new_state = np.ones((half_latt_size, half_latt_size, 4, 4))
    cdef np.float64_t[:,:] cell_inf_res

    cdef np.float64_t[:,:,:] edge_prob = np.zeros((half_latt_size, half_latt_size, 2))

    for k in range(bp_steps):
        for i in range(half_latt_size):
            for j in range(half_latt_size):
                i_n = (i - 1) % half_latt_size
                i_s = (i + 1) % half_latt_size
                j_w = (j - 1) % half_latt_size
                j_e = (j + 1) % half_latt_size

                cell_state = np.stack([boundary_state[i_n, j, 1, :],
                                       boundary_state[i_s, j, 0, :],
                                       boundary_state[i, j_w, 3, :],
                                       boundary_state[i, j_e, 2, :]])
                cell_synd = np.array(synd[2*i : 2*i+2, 2*j : 2*j+2]).flatten()
                cell_inf_res = cell_inf_vp(cell_synd=cell_synd, boundary_state=cell_state,
                                           og_boundary_state=og_boundary_state[i,j],
                                           inner_state = cell_inner_state[i,j])
                new_state[i,j,:,:] = cell_inf_res[:,:]

        boundary_state[...] = new_state

    # post-processing the bp state
    for i in range(half_latt_size):
        for j in range(half_latt_size):
            i_n = (i - 1) % half_latt_size
            j_w = (j - 1) % half_latt_size

            # north edge of the cell: (1 in the last index)
            prob_string = boundary_state[i,j,0,1]*boundary_state[i_n,j,1,1]/og_boundary_state[i,j,0,1] \
                          +boundary_state[i,j,0,2]*boundary_state[i_n,j,1,2]/og_boundary_state[i,j,0,2]
            prob_nostring = boundary_state[i, j, 0, 0] * boundary_state[i_n, j, 1, 0] + boundary_state[i, j, 0, 3] * boundary_state[
                i_n, j, 1, 3] / og_boundary_state[i,j,0,3]
            edge_prob[i,j,1] = prob_string/(prob_string+prob_nostring)

            # west edge of the cell: (0 in the last index)
            prob_string = boundary_state[i, j, 2, 1] * boundary_state[i, j_w, 3, 1]/og_boundary_state[i,j,2,1] \
                          + boundary_state[i, j, 2, 2] * boundary_state[i, j_w, 3, 2]/og_boundary_state[i,j,2,2]
            prob_nostring = boundary_state[i, j, 2, 0] * boundary_state[i, j_w, 3, 0] \
                            + boundary_state[i, j, 2, 3] * boundary_state[i, j_w, 3, 3] / og_boundary_state[i,j,2,3]
            edge_prob[i, j, 0] = prob_string / (prob_string + prob_nostring)

    return edge_prob
