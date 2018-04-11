import tensorflow as tf
from resnet import fc_layer, conv_layer, residual_block

def resnet(inpt, n, keep_rate):
    if n < 20 or (n - 20) % 12 != 0:
        print("ResNet depth invalid.")
        return

    num_conv = int((n - 20) / 12 + 1)
    layers = []

    with tf.variable_scope('conv1'):
        conv1 = conv_layer(inpt, [5, 5, 3, 64], 2, keep_rate)
        layers.append(conv1)

    for i in range (num_conv):
        with tf.variable_scope('conv2_%d' % (i+1)):
            conv2_x = residual_block(layers[-1], 64, keep_rate, False)
            conv2 = residual_block(conv2_x, 64, keep_rate, False)
            layers.append(conv2_x)
            layers.append(conv2)

        assert conv2.get_shape().as_list()[1:] == [16, 16, 64]

    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv3_%d' % (i+1)):
            conv3_x = residual_block(layers[-1], 128, keep_rate, down_sample)
            conv3 = residual_block(conv3_x, 128, keep_rate, False)
            layers.append(conv3_x)
            layers.append(conv3)

        assert conv3.get_shape().as_list()[1:] == [8, 8, 128]
    
    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv4_%d' % (i+1)):
            conv4_x = residual_block(layers[-1], 256, keep_rate, down_sample)
            conv4 = residual_block(conv4_x, 256, keep_rate, False)
            layers.append(conv4_x)
            layers.append(conv4)

        assert conv4.get_shape().as_list()[1:] == [4, 4, 256]
		
    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv5_%d' % (i+1)):
            conv5_x = residual_block(layers[-1], 512, keep_rate, down_sample)
            conv5 = residual_block(conv5_x, 512, keep_rate, False)
            layers.append(conv5_x)
            layers.append(conv5)

        assert conv5.get_shape().as_list()[1:] == [2, 2, 512]

    with tf.variable_scope('fc'):
        global_pool = tf.reduce_mean(layers[-1], [1, 2])
        assert global_pool.get_shape().as_list()[1:] == [512]
        
        fc = tf.nn.relu(fc_layer(global_pool, [512, 1000]))
        out = fc_layer(fc, [1000, 10])
        layers.append(out)

    return layers[-1]
