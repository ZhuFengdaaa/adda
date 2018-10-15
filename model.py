import tensorflow as tf
import numpy as np

def conv(x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, nf]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, nf, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [nf] if one_dim_bias else [1, nf, 1, 1]
    nin = x.get_shape()[channel_ax].value
    wshape = [rf, rf, nin, nf]
    with tf.variable_scope(scope):
        w = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        b = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            b = tf.reshape(b, bshape)
        return tf.nn.conv2d(x, w, strides=strides, padding=pad, data_format=data_format) + b

def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def conv_to_fc(x):
    nh = np.prod([v.value for v in x.get_shape()[1:]])
    x = tf.reshape(x, [-1, nh])
    return x

def nature_cnn(unscaled_images, **conv_kwargs):
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
        **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))

def target_cnn(unscaled_images, **conv_kwargs):
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=7, stride=2, init_scale=np.sqrt(2),
        **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h4 = activ(conv(h3, 'c4', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h5 = activ(conv(h4, 'c5', nf=128, rf=3, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h6 = activ(conv(h5, 'c6', nf=128, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h7 = activ(conv(h6, 'c7', nf=128, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h8 = conv_to_fc(h7)
    return fc(h8, 'fc1', nh=512, init_scale=np.sqrt(2))

def disc_fc(input_feature, **conv_kwargs):
    activ = tf.nn.relu
    h1 = activ(fc(input_feature, 'fc1', nh=256, init_scale=np.sqrt(2)))
    h2 = activ(fc(h1, 'fc2', nh=128, init_scale=np.sqrt(2)))
    h3 = fc(h2, 'fc3', nh=1, init_scale=np.sqrt(2))
    return h3

class SourceModel():
    def __init__(self, args, X):
        # with tf.variable_scope("source"):
        with tf.variable_scope("a2c_model/pi"):
            self.output = nature_cnn(X)

class TargetModel():
    def __init__(self, args, X):
        with tf.variable_scope("a2c_model1/pi"):
            self.output = nature_cnn(X)

class Discriminator():
    def __init__(self, args, M):
        with tf.variable_scope("disc"):
            self.output = disc_fc(M)
