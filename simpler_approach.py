from typing import List, Any

import tensorflow as tf

from tensorflow import layers
from tools import periodic_generator

batch_size = 50

conv_num_filters = 200


def filter_const_one(shape):  # for easy parity computation
    # check whether numerical accuracy will matter
    return tf.constant(1.0, shape=shape)


def parity(in_tensor, block_size):
    parity_filter = filter_const_one([block_size, block_size, 1, 1])

    # print('parity_filter', parity_filter._shape)

    rn_anyons = tf.mod(tf.round(tf.nn.conv2d(tf.slice(in_tensor, [0, 0, 0, 0], [-1, -1, -1, 1]), parity_filter,
                                             strides=[1, block_size, block_size, 1], padding='VALID')), 2)
    return rn_anyons


def conv2d_periodic(inputs, filters=200, kernel_size=[3, 3], activation=tf.nn.leaky_relu):
    assert kernel_size[0] % 2 == 1
    if kernel_size[0] > 1:
        p_size = kernel_size[0] // 2

        # first concatenate along x direction
        inputs = tf.concat([inputs[:, :, - p_size:, :], inputs, inputs[:, :, :p_size, :]], axis=2)

        # concatenate along y direction
        inputs = tf.concat([inputs[:, -p_size:, :, :], inputs, inputs[:, :p_size, :, :]], axis=1)

    return layers.conv2d(inputs, filters, kernel_size, activation=activation)


def rn_block(in_tensor, training=True):
    # assume the first layer in_tensor[:,:,:,0] corresponds to the anyon

    # in_filters= in_tensor.get_shape()[-1]

    h_conv = layers.conv2d(
        inputs=in_tensor,
        filters=conv_num_filters,
        kernel_size=[2, 2],
        strides=[2, 2],
        activation=tf.nn.relu)

    # print 'h_conv1', h_conv1._shape

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

    return h_conv



class Model:
    def __init__(self, sess, learning_rate=7e-4, training = True):

        L = 32
        self.sydn_placeholder = tf.placeholder(tf.float32, shape=[None, L, L, 1])
        self.logical_placeholder = tf.placeholder(tf.float32, shape=[None, 2])

        self.rnc_plhd = []
        for i in [2, 4, 8, 16]:
            self.rnc_plhd.append(tf.placeholder(tf.float32, shape=[None, L // i, L // i, 2]))

        self.rn_block_adam: List[Any] = []
        self.rn_block_adam_op: List[Any] = []
        self.rn_block_loss: List[Any] = []
        with tf.variable_scope("rn_block1"):
            rn_block1 = rn_block(self.sydn_placeholder, training)

            p0 = parity(self.sydn_placeholder, 2)

            rn1 = tf.concat([p0, rn_block1], 3)

        self.rn_block_loss.append(tf.losses.sigmoid_cross_entropy(self.rnc_plhd[0], rn_block1))
        self.rn_block_adam_op.append(tf.train.AdamOptimizer(learning_rate))
        self.rn_block_adam.append(self.rn_block_adam_op[-1].minimize(self.rn_block_loss[-1]))

        with tf.variable_scope("rn_block2"):
            rn_block2 = rn_block(rn1, training)
            p0 = parity(self.sydn_placeholder, 4)

            rn2 = tf.concat([p0, rn_block2], 3)

        self.rn_block_loss.append(tf.losses.sigmoid_cross_entropy(self.rnc_plhd[1], rn_block2))
        self.rn_block_adam_op.append(tf.train.AdamOptimizer(learning_rate))
        self.rn_block_adam.append(self.rn_block_adam_op[-1].minimize(self.rn_block_loss[-1]))

        with tf.variable_scope("rn_block3"):
            rn_block3 = rn_block(rn2, training)
            p0 = parity(self.sydn_placeholder, 8)

            rn3 = tf.concat([p0, rn_block3], 3)

        self.rn_block_loss.append(tf.losses.sigmoid_cross_entropy(self.rnc_plhd[2], rn_block3))
        self.rn_block_adam_op.append(tf.train.AdamOptimizer(learning_rate))
        self.rn_block_adam.append(self.rn_block_adam_op[-1].minimize(self.rn_block_loss[-1]))

        with tf.variable_scope("rn_block4"):
            rn_block4 = rn_block(rn3, training)
            p0 = parity(self.sydn_placeholder, 16)

            rn4 = tf.concat([p0, rn_block4], 3)

        self.rn_block_loss.append(tf.losses.sigmoid_cross_entropy(self.rnc_plhd[3], rn_block4))
        self.rn_block_adam_op.append(tf.train.AdamOptimizer(learning_rate))
        self.rn_block_adam.append(self.rn_block_adam_op[-1].minimize(self.rn_block_loss[-1]))

        rn4_flat = tf.layers.flatten(rn4)
        with tf.variable_scope("dense_block"):
            acti_fun = tf.nn.leaky_relu
            dl = tf.layers.dense(rn4_flat, 50, activation=acti_fun)
            dl = tf.layers.dense(dl, 50, activation=acti_fun)
            dl = tf.layers.dense(dl, 50, activation=acti_fun)
            dl = tf.layers.dense(dl, 2)

        self.loss_logical = tf.losses.sigmoid_cross_entropy(self.logical_placeholder, dl)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            self.adam_glob = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_logical)

        self.accu_logical = tf.contrib.metrics.accuracy((dl > 0), tf.cast(self.logical_placeholder, tf.bool))

        self.saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())


def pre_training(m: Model, sess, num_batch, rn_lvl, p=0.09):
    batch_gen = periodic_generator(32, p, 50, 2**rn_lvl)
    for i in range(num_batch):
        a, b = next(batch_gen)
        sess.run(m.rn_block_adam[rn_lvl-1], feed_dict={m.sydn_placeholder: a, m.rnc_plhd[rn_lvl-1]: b})

        if i % 20 == 0:
            print(sess.run(m.rn_block_loss[rn_lvl-1], feed_dict={m.sydn_placeholder: a, m.rnc_plhd[rn_lvl-1]: b}))

            
def training(m: Model, sess, num_batch, p=0.09):
    batch_gen = periodic_generator(32, p, 50, 32)
    for i in range(num_batch):
        a, b = next(batch_gen)
        b = b.squeeze()
        sess.run(m.adam_glob, feed_dict={m.sydn_placeholder: a, m.logical_placeholder: b})

        if i % 20 == 0:
            print(sess.run([m.loss_logical, m.accu_logical], feed_dict={m.sydn_placeholder: a, m.logical_placeholder: b}))