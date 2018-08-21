import bp
from tools import pg4bp
import numpy as np
import pickle

def gen_bp():
    num_data = 40000
    pg = pg4bp(16, num_data, 2)
    bp_steps = 7

    bp_array = np.zeros((num_data, 8, 8, 2))

    a, c, p = next(pg)
    p = p/(1-p)
    for i in range(num_data):
        bp_array[i] = bp.bp_vp(a[i].squeeze(), p[i].squeeze(), bp_steps)
        if i %100 ==0:
            print(i)

    with open('bp_training_vp_' + str(num_data)+'_1.pkl', 'wb') as f:
        pickle.dump({'anyons': a, 'rn_current': c, 'bp': bp_array, 'p_error':p}, f)


def combine_dataset():
    with open('bp_training_vp_40000.pkl', 'rb') as f:
        data1 = pickle.load(f)

    with open('bp_training_vp_40000_2.pkl', 'rb') as f:
        data2 = pickle.load(f)

    data_comb = {}
    for k in data1:
        data_comb[k] = np.concatenate([data1[k], data2[k]])

    return data_comb


def aug_dataset(data):
    """
    :param data:
    :return: double the amount of data by transpose x and y axis
    """

    data2 = {}

    data2['anyons'] = np.swapaxes(data['anyons'], 1,2)
    data2['bp'] = np.swapaxes(data['bp'], 1, 2)[:,:,:,[1,0]]
    data2['p_error'] = np.swapaxes(data['p_error'], 1, 2)[:, :, :, [1, 0]]
    data2['rn_current'] = np.swapaxes(data['rn_current'], 1, 2)[:, :, :, [1, 0]]

    data_comb = {}
    for k in data:
        data_comb[k] = np.concatenate([data[k], data2[k]])

    with open('bp_training_vp_comb.pkl', 'wb') as f:
        pickle.dump(data_comb, f)