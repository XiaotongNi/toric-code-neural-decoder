from typing import List, Any

import tensorflow as tf
import numpy as np
import warnings

import pickle

from tensorflow import layers
from tensorflow.python.ops import control_flow_ops
from tools import periodic_generator, pg_var_error_rate
import itertools

batch_size = 50
# During early development and previous version of Tensorflow, there are some issues about handling varying batch size.
# Maybe it is not necessary as a global variable now.

conv_num_filters = 200


def parity(in_tensor, block_size):
    """
    Divide the input tensor into block_size * block_size blocks, and compute the parity of each block.
    :param in_tensor: A tensor which has shape [batch_size, l, l, 1]
    :param block_size: size of block
    :return: A tensor with shape [batch_size, l/2, l/2, 1]
    """

    parity_filter = tf.constant(1.0, shape=[block_size, block_size, 1, 1])

    parity_result = tf.mod(tf.round(tf.nn.conv2d(in_tensor, parity_filter,
                                                 strides=[1, block_size, block_size, 1], padding='VALID')), 2)
    # Likely don't need tf.round

    return parity_result


def conv2d_periodic(inputs, filters=200, kernel_size=[3, 3], activation=tf.nn.leaky_relu):
    """
    Convolution layer with periodic boundary condition. Based on tf.layers.conv2d
    :param inputs: 4-D Tensor with format "NHWC"
    :param filters: Number of filters. Same as the parameter in tf.layers.conv2d
    :param kernel_size: Kernel size. We assume kernel_size[0] % 2 == 1. Same as the parameter in tf.layers.conv2d
    :param activation: Activation function. Same as the parameter in tf.layers.conv2d
    :return: Convolution layer with periodic boundary condition.
    """
    assert kernel_size[0] % 2 == 1
    if kernel_size[0] > 1:
        p_size = kernel_size[0] // 2

        inputs = tf.concat([inputs[:, :, - p_size:, :], inputs, inputs[:, :, :p_size, :]], axis=2)

        inputs = tf.concat([inputs[:, -p_size:, :, :], inputs, inputs[:, :p_size, :, :]], axis=1)

    return layers.conv2d(inputs, filters, kernel_size, activation=activation)


def bp_net(in_tensor, training=True):
    """
    This is what we called 'belief propagation network' in the paper
    :param in_tensor: 4-D tensor
    :param training: A boolean passed to batch_normalization.
    :return: 4-D tensor
    """

    h_conv = layers.conv2d(
        inputs=in_tensor,
        filters=conv_num_filters,
        kernel_size=[2, 2],
        strides=[2, 2],
        activation=tf.nn.relu)
    # relu above should be leaky_relu for consistency.
    # This layer does the coarse-grain

    h_conv = conv2d_periodic(inputs=h_conv)
    h_conv = conv2d_periodic(inputs=h_conv, kernel_size=[1, 1])
    h_conv = conv2d_periodic(inputs=h_conv, kernel_size=[1, 1])
    h_conv = layers.batch_normalization(h_conv, training=training)

    h_conv = conv2d_periodic(inputs=h_conv)
    h_conv = conv2d_periodic(inputs=h_conv, kernel_size=[1, 1])
    h_conv = conv2d_periodic(inputs=h_conv, kernel_size=[1, 1])
    h_conv = layers.batch_normalization(h_conv, training=training)

    h_conv = conv2d_periodic(inputs=h_conv)
    h_conv = conv2d_periodic(inputs=h_conv, kernel_size=[1, 1])
    h_conv = conv2d_periodic(inputs=h_conv, kernel_size=[1, 1])
    h_conv = layers.batch_normalization(h_conv, training=training)

    h_conv = conv2d_periodic(inputs=h_conv)
    h_conv = conv2d_periodic(inputs=h_conv, kernel_size=[1, 1])
    h_conv = conv2d_periodic(inputs=h_conv, filters=2, kernel_size=[1, 1], activation=None)
    # activation is set to None so that it can more naturally represent the log scale of error rates

    return h_conv


def remove_entropy(bpo, prev_parity, logi_plhd):
    """
    The main post-processing step. If the bp_net suggests a coarse-grained edge has an error probability > 0.5,
    then we correct it and also change logical placeholder accordingly 
    :param bpo: bp_net output. Each element has range: [-inf, inf], somewhat corresponding to log p/(1-p)
    :param prev_parity: parity from previous level
    :param logi_plhd: The logical placeholder to be updated because we are applying correction
    :return: tuple (bpi, updated_logi_plhd),
        where bpi is a tensor with a format suitable as a bp_net input,
        and updated_logi_plhd is the updated logi_plhd after post-processing
    """

    bpo_g0 = tf.to_float(bpo > 0)
    # Flag the location where log p/(1-p) > 0, in other words p > 0.5
    # Convert to float so can be added to p0

    bpo_nega = -tf.abs(bpo)  # we set it <0 for consistency with bp_net input

    bpo_min = -tf.reduce_min(bpo_nega, axis=(1, 2, 3), keepdims=True) / 7
    bpo_nega = bpo_nega / bpo_min
    # Clipping bpo_nega for consistency with bp_net input,
    # because we only trained bp_net with a certain range of error rates

    p0 = parity(prev_parity, 2)
    # computing syndrome of the coarse-grained lattice

    p0 = p0 + bpo_g0[:, :, :, 0:1] + bpo_g0[:, :, :, 1:2] + tf.manip.roll(bpo_g0[:, :, :, 0:1], -1,
                                                                          axis=2) + tf.manip.roll(
        bpo_g0[:, :, :, 1:2], -1, axis=1)
    p0 = tf.mod(p0, 2)
    # Update p0 according to applying X correction on qubits where p > 0.5

    bsum0 = tf.reduce_sum(bpo_g0[:, :, 0, 0], axis=1)
    bsum1 = tf.reduce_sum(bpo_g0[:, 0, :, 1], axis=1)
    bsum = tf.stack([bsum0, bsum1], axis=-1)
    bsum = bsum + logi_plhd
    # Update logi_plhd according to applying X correction on qubits where p > 0.5

    return tf.concat([p0, bpo_nega], axis=-1), tf.mod(bsum, 2)


class ModelBP:
    """
    For training the belief propagation network
    """

    def __init__(self, sess):
        """
        :param sess: tf.Session(), is used to initialize variables. (Possibly should be done outside the class)
        """

        L = 16
        # Because bp_net only does 4 rounds of communication between neighbouring unit cells, it seems fine to do the
        #  training on L=16 lattice

        self.synd_placeholder = tf.placeholder(tf.float32, shape=[None, L, L, 3])
        self.bp_plhd = tf.placeholder(tf.float32, shape=[None, L // 2, L // 2, 2])

        with tf.variable_scope("rn_block"):
            rn_block = bp_net(self.synd_placeholder)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                updates = tf.group(*update_ops)
                rn_block = control_flow_ops.with_dependencies([updates], rn_block)
            # These codes are related to layers.batch_normalization, see Tensorflow documentation

        self.loss = tf.losses.mean_squared_error(self.bp_plhd, rn_block)
        self.adam = tf.train.AdamOptimizer(7e-4).minimize(self.loss)

        self.saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())


class BatchIteratorBP:
    """
    The batch iterator for training the belief propagation network.
    The training data comes from a belief propagation algorithm
    """

    def __init__(self, path='bp_training_vp_comb.pkl'):
        """
        :param path: location of the training data
        """
        self.test_size = 200
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.anyon_perror = np.concatenate([data['anyons'], np.log10(data['p_error'])], axis=-1)
        # data['anyons'] contains syndrome, data['p_error'] contains error rate of qubits.

        bp_limited = np.clip(data['bp'], 1e-10, 1 - 1e-10)
        # data['bp'] contains the outputs from the belief propagation algorithm
        # it is clipped so we can compute log10 without issues caused by numerical accuracy

        self.bp = np.log10(bp_limited) - np.log10(1 - bp_limited)
        self.batch_start = 0
        self.num_data = self.anyon_perror.shape[0] - self.test_size

    def test_batch(self):
        anyons_batch = self.anyon_perror[self.num_data:, :, :, :]
        bp_batch = self.bp[self.num_data:, :, :, :]

        return anyons_batch, bp_batch

    def next_batch(self):
        batch_end = self.batch_start + batch_size

        if batch_end >= self.num_data:  # temporarily solution
            self.batch_start = 0
            batch_end = self.batch_start + batch_size

        anyons_batch = self.anyon_perror[self.batch_start:batch_end, :, :, :]
        bp_batch = self.bp[self.batch_start:batch_end, :, :, :]

        self.batch_start = batch_end
        return anyons_batch, bp_batch


def pre_training_bp(m: ModelBP, sess, num_batch):
    """
    Training belief propagation network based on ModelBP and BatchIteratorBP
    :param m: An instant of class ModelBP
    :param sess: tf.Session()
    :param num_batch: how many times we run the m.adam operation
    :return: None
    """
    batch_gen = BatchIteratorBP()
    ta, tb = batch_gen.test_batch()
    for i in range(num_batch):
        a, b = batch_gen.next_batch()
        sess.run(m.adam, feed_dict={m.synd_placeholder: a, m.bp_plhd: b})

        if i % 20 == 0:
            print(sess.run(m.loss, feed_dict={m.synd_placeholder: ta, m.bp_plhd: tb}))


class Model:
    """
    Container for the decoder network
    """

    def __init__(self, sess, L):
        """
        :param sess: tf.Session()
        :param L: lattice size of toric code, limited to 16,32,64. Ignore this if your computer is powerful enough
        """
        if (np.log2(L).is_integer() == False) or L < 4:
            raise ValueError('unsupported L value')
        if L > 64:
            warnings.warn("Can be slow.")

        num_renorm_block = int(np.log2(L)) - 1

        self.synd_placeholder = tf.placeholder(tf.float32, shape=[None, L, L, 3])
        self.logical_placeholder = tf.placeholder(tf.float32, shape=[None, 2])

        with tf.variable_scope("rn_block1"):
            rn_block_output = bp_net(self.synd_placeholder, training=False)

            rn_processed, logi_updated = remove_entropy(rn_block_output, self.synd_placeholder,
                                                        self.logical_placeholder)

        for i in range(2, num_renorm_block + 1):
            # The indexing is a bit confusing right now, as it start with rn_block1
            with tf.variable_scope("rn_block" + str(i)):
                rn_block_output = bp_net(rn_processed, training=False)

                rn_processed, logi_updated = remove_entropy(rn_block_output, rn_processed[:, :, :, 0:1], logi_updated)

        # Constructing a list of dictionary for loading pre-trained belief propagation network
        dict_rnblock = []
        for i in range(1, num_renorm_block + 1):
            dict_rnblock.append({})
            rn_block_i_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "rn_block" + str(i))
            for v in rn_block_i_vars:
                dict_rnblock[-1]['rn_block' + v.op.name[9:]] = v

        rn4_flat = tf.layers.flatten(rn_processed)
        with tf.variable_scope("dense_block"):
            acti_fun = tf.nn.leaky_relu
            dl = tf.layers.dense(rn4_flat, 50, activation=acti_fun)
            dl = tf.layers.dense(dl, 50, activation=acti_fun)
            dl = tf.layers.dense(dl, 50, activation=acti_fun)
            dl = tf.layers.dense(dl, 2)

        dense_block_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dense_block")

        self.loss_logical = tf.losses.sigmoid_cross_entropy(logi_updated, dl)
        self.dense_block_trainer = tf.train.AdamOptimizer().minimize(self.loss_logical, var_list=dense_block_vars)
        self.global_trainer = tf.train.AdamOptimizer(7e-6).minimize(self.loss_logical)
        self.accu_logical = tf.contrib.metrics.accuracy((dl > 0), tf.cast(logi_updated, tf.bool))

        self.rn_block_saver = []
        for i in range(num_renorm_block):  # note that here i is from 0 to num_rn_block-1
            self.rn_block_saver.append(tf.train.Saver(var_list=dict_rnblock[i]))
        self.global_saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())


def training_glob(m: Model, sess, train_op, num_batch, L=16, p=0.08):
    """
    Training Model by applying train_op
    :param m: instance of Model
    :param sess: tf.Session()
    :param train_op: training operation: m.dense_block_trainer or m.global_trainer
    :param num_batch: number of batches to train
    :param L: toric code size
    :param p: error probability of training data
    :return: None
    """
    batch_gen = periodic_generator(L, p, batch_size, L)
    p_mat = np.ones((batch_size, L, L, 2)) * np.log10(p / (1 - p))
    # p_mat contains the error rates, although here it is constant, it is needed by the bp_net

    for i in range(num_batch):
        a, l = next(batch_gen)
        l = l.squeeze()
        synd_perror = np.concatenate([a, p_mat], axis=-1)
        o = sess.run([m.loss_logical, m.accu_logical, train_op],
                     feed_dict={m.synd_placeholder: synd_perror, m.logical_placeholder: l})

        if i % 20 == 0:
            print(o[0], o[1])


class ModelVariableRate:
    """
    Container for decoder network that trains additional site-dependent 'error rate' variables
    """

    def __init__(self, sess, L, p):
        """
        :param sess: tf.Session()
        :param L: toric code size, limited to 2^N for N>2
        :param p: how we initialize the 'error rate' variables
        """
        if (np.log2(L).is_integer() is False) or L < 4:
            raise ValueError('unsupported L value')
        if L > 64:
            warnings.warn("Can be slow.")

        num_renorm_block = int(np.log2(L)) - 1

        self.synd_placeholder = tf.placeholder(tf.float32, shape=[None, L, L, 1])
        batch_size_tensor = tf.shape(self.synd_placeholder)[0]
        p_mat = np.ones((1, L, L, 2)) * np.log10(p / (1 - p))
        var_error_rate = tf.Variable(p_mat, name='error_rate', dtype=tf.float32)  # the "error rate" variable
        tiled_error_rate = tf.tile(var_error_rate, tf.stack([batch_size_tensor, 1, 1, 1]))
        synd_perror = tf.concat([self.synd_placeholder, tiled_error_rate], axis=-1)

        self.logical_placeholder = tf.placeholder(tf.float32, shape=[None, 2])

        with tf.variable_scope("rn_block1"):
            rn_block_output = bp_net(synd_perror, training=False)

            rn_processed, logi_updated = remove_entropy(rn_block_output, self.synd_placeholder,
                                                        self.logical_placeholder)

        for i in range(2, num_renorm_block + 1):
            with tf.variable_scope("rn_block" + str(i)):
                rn_block_output = bp_net(rn_processed, training=False)

                rn_processed, logi_updated = remove_entropy(rn_block_output, rn_processed[:, :, :, 0:1], logi_updated)

        rn_block1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "rn_block1")
        rn_block_all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "rn_block")

        dict_rnblock = []
        for i in range(1, num_renorm_block + 1):
            dict_rnblock.append({})
            rn_block_i_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "rn_block" + str(i))
            for v in rn_block_i_vars:
                dict_rnblock[-1]['rn_block' + v.op.name[9:]] = v

        rn4_flat = tf.layers.flatten(rn_processed)
        with tf.variable_scope("dense_block"):
            acti_fun = tf.nn.leaky_relu
            dl = tf.layers.dense(rn4_flat, 50, activation=acti_fun)
            dl = tf.layers.dense(dl, 50, activation=acti_fun)
            dl = tf.layers.dense(dl, 50, activation=acti_fun)
            dl = tf.layers.dense(dl, 2)

        dense_block_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dense_block")

        self.loss_logical = tf.losses.sigmoid_cross_entropy(logi_updated, dl)
        self.accu_logical = tf.contrib.metrics.accuracy((dl > 0), tf.cast(logi_updated, tf.bool))

        rn_block1_vars.append(var_error_rate)
        self.rn_block1_trainer = tf.train.AdamOptimizer(7e-4).minimize(self.loss_logical, var_list=rn_block1_vars)
        # rn_block1_trainer optimize both rn_block1_vars and var_error_rate

        self.error_rate_trainer = tf.train.AdamOptimizer().minimize(self.loss_logical, var_list=[var_error_rate])

        self.global_saver = tf.train.Saver()

        rn_block_all_vars.extend(dense_block_vars)
        self.global_without_errorrate_saver = tf.train.Saver(var_list=rn_block_all_vars)
        # to load everything apart from var_error_rate
        # alternatively, can delete var_error_rate from GLOBAL_VARIABLES

        self.var_error_rate = var_error_rate
        sess.run(tf.global_variables_initializer())


def training_var_rate(m: ModelVariableRate, sess, trainer, num_batch, L=16):
    batch_gen = pg_var_error_rate(L, 50, L)
    for i in range(num_batch):
        a, l, p = next(batch_gen)
        l = l.squeeze()
        sess.run(trainer, feed_dict={m.synd_placeholder: a, m.logical_placeholder: l})

        if i % 20 == 0:
            print(
                sess.run([m.loss_logical, m.accu_logical], feed_dict={m.synd_placeholder: a, m.logical_placeholder: l}))
