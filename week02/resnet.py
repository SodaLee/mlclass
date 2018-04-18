import numpy as np
import tensorflow as tf


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def fc_layer(inpt, shape):
    fc_w = weight_variable(shape)
    fc_b = tf.Variable(tf.zeros([shape[1]]))
    fc_h = tf.matmul(inpt, fc_w) + fc_b

    return fc_h

def conv_layer(inpt, filter_shape, stride, keep_rate):
    out_channels = filter_shape[3]

    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    mean, var = tf.nn.moments(conv, axes=[0,1,2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels], name="gamma")
    
    batch_norm = tf.nn.batch_normalization(
        conv, mean, var, beta, gamma, 0.001)

    drop_out = tf.nn.dropout(batch_norm, keep_rate)

    out = tf.nn.relu(drop_out)

    return out

def residual_block(inpt, output_depth, keep_rate, down_sample, projection=True):
    input_depth = inpt.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1,2,2,1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')

    conv1 = conv_layer(inpt, [3, 3, input_depth, output_depth], 1, keep_rate)
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1, keep_rate)

    if input_depth != output_depth:
        if projection:
            input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth], 1, keep_rate)
        else:
            input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt

    res = conv2 + input_layer
    return res
