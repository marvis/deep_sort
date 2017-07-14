import tensorflow as tf
import numpy as np

sess = tf.Session()
new_saver = tf.train.import_meta_graph('networks/mars-small128.ckpt-68577.meta')
new_saver.restore(sess, 'networks/mars-small128.ckpt-68577')

#print(sess.run('conv1_1/weights:0').transpose(3,2,0,1))
#exit(0)
fp = open('deep_sort.weights', 'wb')
np.zeros(4, dtype=np.float32).tofile(fp)
def save_conv_bn(fp, name):
    print('save_conv_bn %s' % name)
    conv_weights = sess.run(name + '/weights:0').transpose(3,2,0,1)
    bn_beta = sess.run(name + '/' + name + '/bn/beta:0')
    bn_moving_mean = sess.run(name + '/' + name + '/bn/moving_mean:0')
    bn_moving_variance = sess.run(name + '/' + name + '/bn/moving_variance:0')
    bn_scale = np.ones(bn_beta.size, dtype=np.float32)
    bn_beta.tofile(fp)
    bn_scale.tofile(fp)
    bn_moving_mean.tofile(fp)
    bn_moving_variance.tofile(fp)
    conv_weights.tofile(fp)
    

def save_conv(fp, name):
    print('save_conv    %s' % name)
    conv_weights = sess.run(name + '/weights:0').transpose(3,2,0,1)
    conv_biases = sess.run(name + '/biases:0')
    conv_biases.tofile(fp)
    conv_weights.tofile(fp)

def save_proj(fp, name):
    print('save_proj    %s' % name)
    conv_weights = sess.run(name + '/weights:0').transpose(3,2,0,1)
    conv_biases = np.zeros(conv_weights.shape[3], dtype=np.float32)
    conv_biases.tofile(fp)
    conv_weights.tofile(fp)

def save_bn(fp, name):
    print('save_bn      %s' % name)
    bn_beta = sess.run(name + '/beta:0')
    bn_moving_mean = sess.run(name + '/moving_mean:0')
    bn_moving_variance = sess.run(name + '/moving_variance:0')
    bn_scale = np.ones(bn_beta.size, dtype=np.float32)
    bn_moving_mean = bn_moving_mean - bn_beta * np.sqrt(bn_moving_variance+0.001)
    bn_scale.tofile(fp)
    bn_moving_mean.tofile(fp)
    bn_moving_variance.tofile(fp)

def save_fc_bn(fp, name):
    print('save_fc_bn   %s' % name)
    fc_weights = sess.run(name + '/weights:0').transpose(1,0)
    bn_beta = sess.run(name + '/' + name + '/bn/beta:0')
    bn_moving_mean = sess.run(name + '/' + name + '/bn/moving_mean:0')
    bn_moving_variance = sess.run(name + '/' + name + '/bn/moving_variance:0')
    bn_scale = np.ones(bn_beta.size, dtype=np.float32)
    bn_beta.tofile(fp)
    bn_scale.tofile(fp)
    bn_moving_mean.tofile(fp)
    bn_moving_variance.tofile(fp)
    fc_weights.tofile(fp)

# Conv1
save_conv_bn(fp,'conv1_1') # combine bn params to conv
# Conv2
save_conv_bn(fp,'conv1_2')

# Res4
save_conv_bn(fp,'conv2_1/1')
save_conv(fp,'conv2_1/2')

# Res5
save_bn(fp,'conv2_3/bn')
save_conv_bn(fp,'conv2_3/1')
save_conv(fp,'conv2_3/2')

# Res6
save_bn(fp,'conv3_1/bn')
save_conv_bn(fp,'conv3_1/1')
save_conv(fp,'conv3_1/2')
save_proj(fp,'conv3_1/projection')

# Res7
save_bn(fp,'conv3_3/bn')
save_conv_bn(fp,'conv3_3/1')
save_conv(fp,'conv3_3/2')

# Res8
save_bn(fp,'conv4_1/bn')
save_conv_bn(fp,'conv4_1/1')
save_conv(fp,'conv4_1/2')
save_proj(fp,'conv4_1/projection')

# Res9
save_bn(fp,'conv4_3/bn')
save_conv_bn(fp,'conv4_3/1')
save_conv(fp,'conv4_3/2')

save_fc_bn(fp,'fc1')

fp.close()

