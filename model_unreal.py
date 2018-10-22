import tensorflow as tf
import numpy as np

def fc_initializer(input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer


def conv_initializer(kernel_width, kernel_height, input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer

def _conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

def _fc_variable(weight_shape, name):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)
    
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape, initializer=fc_initializer(input_channels))
    bias   = tf.get_variable(name_b, bias_shape,   initializer=fc_initializer(input_channels))
    return weight, bias

def _conv_variable(weight_shape, name, deconv=False):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)
    
    w = weight_shape[0]
    h = weight_shape[1]
    if deconv:
        input_channels  = weight_shape[3]
        output_channels = weight_shape[2]
    else:
        input_channels  = weight_shape[2]
        output_channels = weight_shape[3]
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape,
                             initializer=conv_initializer(w, h, input_channels))
    bias   = tf.get_variable(name_b, bias_shape,
                             initializer=conv_initializer(w, h, input_channels))
    return weight, bias

def unreal_cnn(state_input, reuse=False):
    with tf.variable_scope("base_conv", reuse=reuse) as scope:
        # Weights
        W_conv1, b_conv1 = _conv_variable([8, 8, 3, 16],  "base_conv1") # 16 8x8 filters
        W_conv2, b_conv2 = _conv_variable([4, 4, 16, 32], "base_conv2") # 32 4x4 filters

        # Nodes
        h_conv1 = tf.nn.relu(_conv2d(state_input, W_conv1, 4) + b_conv1) # stride=4 => 19x19x16
        h_conv2 = _conv2d(h_conv1,     W_conv2, 2) + b_conv2 # stride=2 => 9x9x32
        return h_conv2

def disc_fc(input_feature_map, **conv_kwargs):
    activ = tf.nn.relu
    W_fc1, b_fc1 = _fc_variable([2592, 256], "base_fc1")
    W_fc2, b_fc2 = _fc_variable([256, 256], "base_fc2")
    W_fc3, b_fc3 = _fc_variable([256, 1], "base_fc3")
    
    input_feature = tf.reshape(input_feature_map, [-1, 2592])
    fc1 = tf.nn.relu(tf.matmul(input_feature, W_fc1) + b_fc1)
    fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2) + b_fc2)
    fc3 = tf.nn.relu(tf.matmul(fc2, W_fc3) + b_fc3)
    return fc3

class SourceModel():
    def __init__(self, args, X):
        # with tf.variable_scope("source"):
        with tf.variable_scope("source"):
            self.output = unreal_cnn(X)

class TargetModel():
    def __init__(self, args, X):
        with tf.variable_scope("target"):
            self.output = unreal_cnn(X)

class Discriminator():
    def __init__(self, args, M):
        with tf.variable_scope("disc"):
            self.output = disc_fc(M)
