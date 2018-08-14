import numpy as np
from tools import periodic_generator

e_all = np.unpackbits(np.arange(4096, dtype=np.uint16).reshape((4096, 1)).view(np.uint8), axis=1)
e_all = e_all[:, [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]]

c = np.zeros((4, 12), dtype=np.int8)

c[0, [0, 2, 6, 8]] = 1
c[1, [1, 3, 8, 10]] = 1
c[2, [2, 4, 7, 9]] = 1
c[3, [3, 5, 9, 11]] = 1
c = c.transpose()

error_prob = 0.08


def cell_inf(cell_synd, cell_state):
    """
    inference inside a cell
    :return:
    """

    new_cell_state = np.zeros_like(cell_state, dtype=np.float32)
    all_possible_synd = np.matmul(e_all, c)
    all_possible_synd %= 2
    matching_errors_idx = np.nonzero(np.all(all_possible_synd == cell_synd, axis=1))[0]

    # matching_errors = e_all[matching_errors_idx]
    for i in range(matching_errors_idx.size):
        error = e_all[matching_errors_idx[i]]

        n_edge = error[0] + error[1] * 2
        s_edge = error[4] + error[5] * 2
        w_edge = error[6] + error[7] * 2
        e_edge = error[10] + error[11] * 2

        p_n = cell_state[0, n_edge]
        p_s = cell_state[1, s_edge]
        p_w = cell_state[2, w_edge]
        p_e = cell_state[3, e_edge]

        weight_otherqbit = error[[2, 3, 8, 9]].sum()

        total_p = p_e * p_n * p_s * p_w * error_prob ** weight_otherqbit
        new_cell_state[0, n_edge] += total_p / p_n * error_prob ** (error[0] + error[1])
        new_cell_state[1, s_edge] += total_p / p_s* error_prob ** (error[4] + error[5])
        new_cell_state[2, w_edge] += total_p / p_w* error_prob ** (error[6] + error[7])
        new_cell_state[3, e_edge] += total_p / p_e* error_prob ** (error[10] + error[11])

    new_cell_state /= np.sum(new_cell_state, axis=1, keepdims=True)

    return new_cell_state


def bp(synd, bp_steps):
    latt_size = 16
    half_latt_size = latt_size//2

    state = np.ones((half_latt_size, half_latt_size, 4, 4))
    state[:, :, :, [1, 2]] = error_prob
    state[:, :, :, 3] = error_prob ** 2
    new_state = np.ones((half_latt_size, half_latt_size, 4, 4))

    edge_prob = np.zeros((half_latt_size, half_latt_size, 2))

    for k in range(bp_steps):
        for i in range(half_latt_size):
            for j in range(half_latt_size):
                i_n = (i - 1) % half_latt_size
                i_s = (i + 1) % half_latt_size
                j_w = (j - 1) % half_latt_size
                j_e = (j + 1) % half_latt_size

                cell_state = np.stack([state[i_n, j, 1, :],
                                       state[i_s, j, 0, :],
                                       state[i, j_w, 3, :],
                                       state[i, j_e, 2, :]])
                assert cell_state.shape == (4, 4)

                new_state[i, j, :, :] = cell_inf(cell_synd=synd[i, j, :], cell_state=cell_state)

        state = new_state.copy()

    # post-processing the bp state
    for i in range(half_latt_size):
        for j in range(half_latt_size):
            i_n = (i - 1) % half_latt_size
            j_w = (j - 1) % half_latt_size

            # north edge of the cell:
            prob_string = (state[i,j,0,1]*state[i_n,j,1,1]+state[i,j,0,2]*state[i_n,j,1,2])/error_prob
            prob_nostring = state[i, j, 0, 0] * state[i_n, j, 1, 0] + state[i, j, 0, 3] * state[
                i_n, j, 1, 3] / error_prob**2
            edge_prob[i,j,0] = prob_string/(prob_string+prob_nostring)

            # west edge of the cell:
            prob_string = (state[i, j, 2, 1] * state[i, j_w, 3, 1] + state[i, j, 2, 2] * state[
                i, j_w, 3, 2]) / error_prob
            prob_nostring = state[i, j, 2, 0] * state[i, j_w, 3, 0] + state[i, j, 2, 3] * state[
                i, j_w, 3, 3] / error_prob ** 2
            edge_prob[i, j, 1] = prob_string / (prob_string + prob_nostring)

    return edge_prob


def test():
    pg = periodic_generator(16, 0.08, 1, 2)

    a, c = next(pg)

    i = np.squeeze(a)
    ix, iy = i.shape
    s = (ix // 2, iy // 2) + (2, 2)
    strd = np.lib.stride_tricks.as_strided
    x, y = i.strides
    subM = strd(i, shape=s, strides=(2 * x, 2 * y, x, y))

    synd = subM.reshape((ix // 2, iy // 2, 4))

    return a, bp(synd, 4), bp(synd, 10)
