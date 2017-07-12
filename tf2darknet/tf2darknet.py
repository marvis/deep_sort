# vim: expandtab:ts=4:sw=4
import os
import errno
import argparse
import numpy as np
import cv2

import tensorflow as tf
import tensorflow.contrib.slim as slim

from collections import OrderedDict
layers = []

from prototxt import *

def get_scope(name):
    if type(name) == list:
        out = []
        for item in name:
            item = get_scope(item)
            out.append(item)
        return out
    else:
        splits = name.split('/')
        out = splits[0]
        for i in range(1, len(splits)-1):
            out = out + '/' + splits[i]
        return out
  
def add_layer(ltype, bottom, top, name=None):
    layer = OrderedDict()
    layer['type'] = ltype
    layer['bottom'] = get_scope(bottom)
    layer['top'] = get_scope(top)
    if name != None:
        layer['name'] = get_scope(name)
    else:
        layer['name'] = get_scope(top)
    layers.append(layer)

def add_conv_layer(ltype, bottom, top, name=None):
    layer = OrderedDict()
    layer['type'] = ltype
    layer['bottom'] = bottom
    layer['top'] = top
    if name != None:
        layer['name'] = name
    else:
        layer['name'] = top
    layers.append(layer)


def _batch_norm_fn(x, scope=None):
    if scope is None:
        scope = tf.get_variable_scope().name + "/bn"
    return slim.batch_norm(x, scope=scope)


def create_link(
        incoming, network_builder, scope, nonlinearity=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(stddev=1e-3),
        regularizer=None, is_first=False, summarize_activations=True):
    if is_first:
        network = incoming
    else:
        bottom = incoming.op.name
        network = _batch_norm_fn(incoming, scope=scope + "/bn")
        print("BatchNorm %s []" % (network.op.name))
        add_layer('BatchNorm', bottom, network.op.name)

        bottom = network.op.name
        network = nonlinearity(network)
        print("Activation %s [act=elu]" % (network.op.name))
        add_layer('ELU', bottom, network.op.name)

        if summarize_activations:
            tf.summary.histogram(scope+"/activations", network)

    pre_block_network = network
    post_block_network = network_builder(pre_block_network, scope)

    incoming_dim = pre_block_network.get_shape().as_list()[-1]
    outgoing_dim = post_block_network.get_shape().as_list()[-1]
    if incoming_dim != outgoing_dim:
        assert outgoing_dim == 2 * incoming_dim, \
            "%d != %d" % (outgoing_dim, 2 * incoming)
        projection = slim.conv2d(
            incoming, outgoing_dim, 1, 2, padding="SAME", activation_fn=None,
            scope=scope+"/projection", weights_initializer=weights_initializer,
            biases_initializer=None, weights_regularizer=regularizer)
        print("Conv2d %s [filters=%d, kernel_size=1, stride=2, act=None, pad=same, bn=0]" % (projection.name, outgoing_dim))
        add_layer('Convolution', incoming.op.name, projection.op.name)
        network = projection + post_block_network
        print("Add %s []" % network.name)
        add_layer('Eltwise', [projection.op.name, post_block_network.op.name], network.op.name)
    else:
        network = incoming + post_block_network
        print("Add %s []" % network.name)
        add_layer('Eltwise', [incoming.op.name, post_block_network.op.name], network.op.name)
    return network


def create_inner_block(
        incoming, scope, nonlinearity=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(1e-3),
        bias_initializer=tf.zeros_initializer(), regularizer=None,
        increase_dim=False, summarize_activations=True):
    n = incoming.get_shape().as_list()[-1]
    stride = 1
    if increase_dim:
        n *= 2
        stride = 2

    bottom = incoming.op.name
    incoming = slim.conv2d(
        incoming, n, [3, 3], stride, activation_fn=nonlinearity, padding="SAME",
        normalizer_fn=_batch_norm_fn, weights_initializer=weights_initializer,
        biases_initializer=bias_initializer, weights_regularizer=regularizer,
        scope=scope + "/1")
    print("Conv2d %s [filters=%d, kernel_size=3, stride=%d, act=elu, pad=same, bn=1]" % (incoming.name, n, stride))
    add_layer('Convolution', bottom, incoming.op.name)
    
    if summarize_activations:
        tf.summary.histogram(incoming.name + "/activations", incoming)

    bottom = incoming.op.name
    incoming = slim.dropout(incoming, keep_prob=0.6)
    print("Dropout %s [prob=0.6]" % incoming.name)
    add_layer('Dropout', bottom, incoming.op.name)

    bottom = incoming.op.name
    incoming = slim.conv2d(
        incoming, n, [3, 3], 1, activation_fn=None, padding="SAME",
        normalizer_fn=None, weights_initializer=weights_initializer,
        biases_initializer=bias_initializer, weights_regularizer=regularizer,
        scope=scope + "/2")
    print("Conv2d %s [filters=%d, kernel_size=3, stride=1, act=None, pad=same, bn=0]" % (incoming.name, n))
    add_layer('Convolution', bottom, incoming.op.name)
    return incoming


def residual_block(incoming, scope, nonlinearity=tf.nn.elu,
                   weights_initializer=tf.truncated_normal_initializer(1e3),
                   bias_initializer=tf.zeros_initializer(), regularizer=None,
                   increase_dim=False, is_first=False,
                   summarize_activations=True):

    def network_builder(x, s):
        return create_inner_block(
            x, s, nonlinearity, weights_initializer, bias_initializer,
            regularizer, increase_dim, summarize_activations)

    return create_link(
        incoming, network_builder, scope, nonlinearity, weights_initializer,
        regularizer, is_first, summarize_activations)


def _create_network(incoming, num_classes, reuse=None, l2_normalize=True,
                   create_summaries=True, weight_decay=1e-8):
    nonlinearity = tf.nn.elu
    conv_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    conv_bias_init = tf.zeros_initializer()
    conv_regularizer = slim.l2_regularizer(weight_decay)
    fc_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    fc_bias_init = tf.zeros_initializer()
    fc_regularizer = slim.l2_regularizer(weight_decay)

    def batch_norm_fn(x):
        return slim.batch_norm(x, scope=tf.get_variable_scope().name + "/bn")

    network = incoming
    bottom = network.op.name
    network = slim.conv2d(
        network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_1",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)
    print("Conv2d %s [filters=32, kernel_size=3, stride=1, act=elu, pad=same, bn=1]" % network.name)
    add_layer('Convolution', bottom, network.op.name)

    if create_summaries:
        tf.summary.histogram(network.name + "/activations", network)
        tf.summary.image("conv1_1/weights", tf.transpose(
            slim.get_variables("conv1_1/weights:0")[0], [3, 0, 1, 2]),
                         max_images=128)

    bottom = network.op.name
    network = slim.conv2d(
        network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_2",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)
    print("Conv2d %s [filters=32, kernel_size=3, stride=1, act=elu, pad=same, bn=1]" % network.name)
    add_layer('Convolution', bottom, network.op.name)
    if create_summaries:
        tf.summary.histogram(network.name + "/activations", network)

    bottom = network.op.name
    network = slim.max_pool2d(network, [3, 3], [2, 2], scope="pool1")
    print("MaxPool2d %s [kernel_size=3, stride=2]" % network.name)
    add_layer('MaxPool2d', bottom, network.op.name)

    network = residual_block(
        network, "conv2_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False, is_first=True,
        summarize_activations=create_summaries)
    network = residual_block(
        network, "conv2_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    network = residual_block(
        network, "conv3_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True,
        summarize_activations=create_summaries)
    network = residual_block(
        network, "conv3_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    network = residual_block(
        network, "conv4_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True,
        summarize_activations=create_summaries)
    network = residual_block(
        network, "conv4_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    feature_dim = network.get_shape().as_list()[-1]
    print("feature dimensionality: ", feature_dim)
    bottom = network.op.name
    network = slim.flatten(network)
    print("Flatten %s []" % network.name)
    add_layer('Flatten', bottom, network.op.name)

    bottom = network.op.name
    network = slim.dropout(network, keep_prob=0.6)
    print("Dropout %s []" % network.name)
    add_layer('Dropout', bottom, network.op.name)

    network = slim.fully_connected(
        network, feature_dim, activation_fn=nonlinearity,
        normalizer_fn=batch_norm_fn, weights_regularizer=fc_regularizer,
        scope="fc1", weights_initializer=fc_weight_init,
        biases_initializer=fc_bias_init)
    print("FC %s []" % network.name)
    add_layer('InnerProduct', network.op.name, network.op.name)

    features = network

    if l2_normalize:
        # Features in rows, normalize axis 1.
        features = slim.batch_norm(features, scope="ball", reuse=reuse)
        feature_norm = tf.sqrt(
            tf.constant(1e-8, tf.float32) +
            tf.reduce_sum(tf.square(features), [1], keep_dims=True))
        features = features / feature_norm

        with slim.variable_scope.variable_scope("ball", reuse=reuse):
            weights = slim.model_variable(
                "mean_vectors", (feature_dim, num_classes),
                initializer=tf.truncated_normal_initializer(stddev=1e-3),
                regularizer=None)
            scale = slim.model_variable(
                "scale", (num_classes, ), tf.float32,
                tf.constant_initializer(0., tf.float32), regularizer=None)
            if create_summaries:
                tf.summary.histogram("scale", scale)
            # scale = slim.model_variable(
            #     "scale", (), tf.float32,
            #     initializer=tf.constant_initializer(0., tf.float32),
            #     regularizer=slim.l2_regularizer(1e-2))
            # if create_summaries:
            #     tf.scalar_summary("scale", scale)
            scale = tf.nn.softplus(scale)

        # Each mean vector in columns, normalize axis 0.
        weight_norm = tf.sqrt(
            tf.constant(1e-8, tf.float32) +
            tf.reduce_sum(tf.square(weights), [0], keep_dims=True))
        logits = scale * tf.matmul(features, weights / weight_norm)

    else:
        logits = slim.fully_connected(
            features, num_classes, activation_fn=None,
            normalizer_fn=None, weights_regularizer=fc_regularizer,
            scope="softmax", weights_initializer=fc_weight_init,
            biases_initializer=fc_bias_init)

    return features, logits


def _network_factory(num_classes, is_training, weight_decay=1e-8):

    def factory_fn(image, reuse, l2_normalize):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=is_training):
                with slim.arg_scope([slim.conv2d, slim.fully_connected,
                                     slim.batch_norm, slim.layer_norm],
                                    reuse=reuse):
                    features, logits = _create_network(
                        image, num_classes, l2_normalize=l2_normalize,
                        reuse=reuse, create_summaries=is_training,
                        weight_decay=weight_decay)
                    return features, logits

    return factory_fn


def _preprocess(image, is_training=False, enable_more_augmentation=True):
    image = image[:, :, ::-1]  # BGR to RGB
    if is_training:
        image = tf.image.random_flip_left_right(image)
        if enable_more_augmentation:
            image = tf.image.random_brightness(image, max_delta=50)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return image


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, patch_shape[::-1])

    return image


def _create_image_encoder(preprocess_fn, factory_fn, image_shape, batch_size=32,
                         session=None, checkpoint_path=None,
                         loss_mode="cosine"):
    image_var = tf.placeholder(tf.uint8, (None, ) + image_shape)

    preprocessed_image_var = tf.map_fn(
        lambda x: preprocess_fn(x, is_training=False),
        tf.cast(image_var, tf.float32))

    l2_normalize = loss_mode == "cosine"
    feature_var, _ = factory_fn(
        preprocessed_image_var, l2_normalize=l2_normalize, reuse=None)
    feature_dim = feature_var.get_shape().as_list()[-1]

    if session is None:
        session = tf.Session()
    if checkpoint_path is not None:
        slim.get_or_create_global_step()
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
            checkpoint_path, slim.get_variables_to_restore())
        session.run(init_assign_op, feed_dict=init_feed_dict)


    #for var in tf.all_variables():
    for var in tf.model_variables():
        print(var.op.name)

    def encoder(data_x):
        out = np.zeros((len(data_x), feature_dim), np.float32)
        _run_in_batches(
            lambda x: session.run(feature_var, feed_dict=x),
            {image_var: data_x}, out, batch_size)
        return out

    return encoder


def create_image_encoder(model_filename, batch_size=32, loss_mode="cosine",
                         session=None):
    image_shape = 128, 64, 3
    factory_fn = _network_factory(
        num_classes=1501, is_training=False, weight_decay=1e-8)

    return _create_image_encoder(
        _preprocess, factory_fn, image_shape, batch_size, session,
        model_filename, loss_mode)


def create_box_encoder(model_filename, batch_size=32, loss_mode="cosine"):
    image_shape = 128, 64, 3
    image_encoder = create_image_encoder(model_filename, batch_size, loss_mode)

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches)

    return encoder


def generate_detections(encoder, mot_dir, output_dir, detection_dir=None):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(
            #detection_dir, sequence, "det/det.txt")
            detection_dir, sequence, "st/st.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(
                image_filenames[frame_idx], cv2.IMREAD_COLOR)
            features = encoder(bgr_image, rows[:, 2:6].copy())
            detections_out += [np.r_[(row, feature)] for row, feature
                               in zip(rows, features)]

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model",
        default="networks/mars-small128.ckpt-68577",
        help="Path to checkpoint file")
    parser.add_argument(
        "--loss_mode", default="cosine", help="Network loss training mode")
    parser.add_argument(
        "--detection_dir", help="Path to custom detections. Defaults to "
        "standard MOT detections Directory structure should be the default "
        "MOTChallenge structure: [sequence]/det/det.txt", default=None)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default="detections")
    return parser.parse_args()

def get_tensor_by_name(name):
    var = [v for v in tf.global_variables() if v.name == name][0]
    return var

def save_conv_bn(name):
    conv_weights = get_tensor_by_name(name + "/weights")

if __name__ == "__main__":
    args = parse_args()
    f = create_box_encoder(args.model, batch_size=32, loss_mode=args.loss_mode)
    #writer = tf.summary.FileWriter("/tmp/sort/log", tf.get_default_graph())
    #writer.close()

    net_info = OrderedDict()
    props = OrderedDict()
    props['name'] = 'test'
    props['input'] = 'data'
    props['input_dim'] = [1, 3, 128, 64]
    net_info['props'] = props
    net_info['layers'] = layers
    save_prototxt(net_info, "test.prototxt")
