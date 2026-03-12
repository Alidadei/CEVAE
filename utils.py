import tensorflow as tf
import numpy as np
import sys


def fc_net(inp, layers, out_layers, scope, lamba=1e-3, activation=tf.nn.relu, reuse=None,
           weights_initializer=tf.initializers.glorot_uniform()):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        h = inp
        for i, units in enumerate(layers):
            weight = tf.compat.v1.get_variable(f'{scope}_dense_{i+1}_weight', 
                                   shape=[h.get_shape().as_list()[-1], units],
                                   initializer=weights_initializer,
                                   regularizer=tf.keras.regularizers.l2(lamba))
            bias = tf.compat.v1.get_variable(f'{scope}_dense_{i+1}_bias',
                                 shape=[units],
                                 initializer=tf.zeros_initializer())
            h = activation(tf.matmul(h, weight) + bias)
        
        if not out_layers:
            return h
        
        outputs = []
        for i, (outdim, act) in enumerate(out_layers):
            weight = tf.compat.v1.get_variable(f'{scope}_out_{i+1}_weight',
                                   shape=[h.get_shape().as_list()[-1], outdim],
                                   initializer=weights_initializer,
                                   regularizer=tf.keras.regularizers.l2(lamba))
            bias = tf.compat.v1.get_variable(f'{scope}_out_{i+1}_bias',
                                 shape=[outdim],
                                 initializer=tf.zeros_initializer())
            if act is None:
                output = tf.matmul(h, weight) + bias
            else:
                output = act(tf.matmul(h, weight) + bias)
            outputs.append(output)
        return outputs if len(outputs) > 1 else outputs[0]


def get_y0_y1(sess, y, f0, f1, shape=(), L=1, verbose=True):
    y0, y1 = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    ymean = y.mean()
    for l in range(L):
        if L > 1 and verbose:
            sys.stdout.write('\r Sample {}/{}'.format(l + 1, L))
            sys.stdout.flush()
        y0 += sess.run(ymean, feed_dict=f0) / L
        y1 += sess.run(ymean, feed_dict=f1) / L

    if L > 1 and verbose:
        print()
    return y0, y1


