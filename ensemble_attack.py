"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import os
import sys
import time
import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
import tensorflow as tf

sys.path.append('slim')
from slim.nets import inception, resnet_v2, nets_factory

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_dir', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 4, 'How many images process at one time.')

tf.flags.DEFINE_integer(
    'num_iterations', 9, 'How many iterations to do.')

FLAGS = tf.flags.FLAGS


def load_target_class(input_dir):
    """Loads target classes."""
    with tf.gfile.Open(os.path.join(input_dir, 'target_class.csv')) as f:
        return {row[0]: int(row[1]) for row in csv.reader(f) if len(row) >= 2}


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath) as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
      images: array with minibatch of images
      filenames: list of filenames without path
        If number of file names in this list less than number of images in
        the minibatch then only first len(filenames) images will be saved.
      output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def inception_preprocess(images, crop_height, crop_width):
    return (images + 1.0) / 2.0


def vgg_preprocess(images, crop_height, crop_width):
    image_height, image_width = FLAGS.image_height, FLAGS.image_width
    offset_height = int((image_height - crop_height) / 2)
    offset_width = int((image_width - crop_width) / 2)

    means = [123.68, 116.779, 103.939]

    images = tf.image.crop_to_bounding_box((images + 1.0) * 255.0 / 2.0, offset_height, offset_width, crop_height,
                                           crop_width)

    return images - means


def model(model_name, x_input, num_classes, preprocess=False, is_training=False, label_offset=0, scope=None,
          reuse=None):
    network_fn = nets_factory.get_network_fn(
        model_name,
        num_classes=num_classes - label_offset,
        is_training=is_training,
        scope=scope,
        reuse=reuse)

    eval_image_size = network_fn.default_image_size
    print('model[' + model_name + '] eval_image_size:', eval_image_size)

    images = x_input
    if preprocess:
        #         images = preprocess_batch(model_name, x_input, eval_image_size, eval_image_size)
        if model_name.startswith('resnet_v1') or model_name.startswith('vgg'):
            images = vgg_preprocess(x_input, eval_image_size, eval_image_size)
        # if model_name.startswith('inception_v'):
        #     images = inception_preprocess(x_input, eval_image_size, eval_image_size)

    logits, _ = network_fn(images)

    return logits


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print([str(i.name) for i in not_initialized_vars])  # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def main(_):
    main_start_time = time.time()
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    ckpt_dir = FLAGS.checkpoint_dir

    max_epsilon = FLAGS.max_epsilon
    eps = 2.0 * max_epsilon / 255.0
    num_iterations = FLAGS.num_iterations

    confidence = 40
    targeted = True
    const = 8.75e+3

    learning_rate = 2.0
    print('learning_rate:', learning_rate)
    batch_size = FLAGS.batch_size
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    joint_ckpt_path = ckpt_dir + 'joint_e4ai3_eair2_ir2_i3_r250_e3ai3_i4_ai3.ckpt'
    model_checkpoints = {

        'resnet_v1_50': ckpt_dir + 'resnet_v1_50/resnet_v1_50.ckpt',
        'resnet_v1_101': ckpt_dir + 'resnet_v1_101/resnet_v1_101.ckpt',
        'resnet_v1_152': ckpt_dir + 'resnet_v1_152/resnet_v1_152.ckpt',
        'resnet_v2_50': ckpt_dir + 'resnet_v2_50/resnet_v2_50.ckpt',
        'resnet_v2_101': ckpt_dir + 'resnet_v2_101/resnet_v2_101.ckpt',
        'resnet_v2_152': ckpt_dir + 'resnet_v2_152/resnet_v2_152.ckpt',

        'vgg_16': ckpt_dir + 'vgg_16/vgg_16.ckpt',
        'vgg_19': ckpt_dir + 'vgg_19/vgg_19.ckpt',

        'inception_v3': ckpt_dir + 'inception_v3/inception_v3.ckpt',
        'inception_v4': ckpt_dir + 'inception_v4/inception_v4.ckpt',
        'inception_resnet_v2': ckpt_dir + 'inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt',

        # adversarially trained model:
        'adv_inception_v3': ckpt_dir + 'adv_inception_v3/adv_inception_v3.ckpt',

        # ensemble adversarially trained model:
        'ens3_adv_inception_v3': ckpt_dir + 'ens3_adv_inception_v3_2017_08_18/ens3_adv_inception_v3.ckpt',
        'ens4_adv_inception_v3': ckpt_dir + 'ens4_adv_inception_v3_2017_08_18/ens4_adv_inception_v3.ckpt',
        'ens_adv_inception_resnet_v2': ckpt_dir + 'ens_adv_inception_resnet_v2/ens_adv_inception_resnet_v2.ckpt',
    }

    tf.logging.set_verbosity(tf.logging.INFO)

    all_images_taget_class = load_target_class(FLAGS.input_dir)

    graph_start_time = time.time()

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        y_input = tf.placeholder(tf.int64, [batch_size], 'y_input')

        x_noisy = tf.get_variable('x_noisy', batch_shape, tf.float32, tf.constant_initializer())

        x_input_var = tf.get_variable('x_input_var', batch_shape, tf.float32, tf.constant_initializer())
        x_input_var_init = tf.assign(x_input_var, x_input)

        y_input_var = tf.get_variable('y_input_var', [batch_size], tf.int64, tf.constant_initializer(), trainable=False)
        y_input_var_init = tf.assign(y_input_var, y_input)

        x_max = tf.get_variable('x_max', batch_shape, tf.float32, tf.constant_initializer(), trainable=False)
        x_min = tf.get_variable('x_min', batch_shape, tf.float32, tf.constant_initializer(), trainable=False)

        x_max_init = tf.assign(x_max, tf.clip_by_value(x_input + eps, -1.0, 1.0))
        x_min_init = tf.assign(x_min, tf.clip_by_value(x_input - eps, -1.0, 1.0))

        x_scale = (x_max - x_min) * 0.5
        x_noisy_init_op = tf.assign(x_noisy, tf.atanh((x_input - x_min) / x_scale - 1.0))

        models_scopes = {
            'inception_resnet_v2': 'InceptionResnetV2',
            'ens_adv_inception_resnet_v2': 'ens_adv_inception_resnet_v2',
            'ens3_adv_inception_v3': 'ens3_adv_inception_v3',
            'ens4_adv_inception_v3': 'ens4_adv_inception_v3',
            # 'resnet_v2_50': 'resnet_v2_50',
            'adv_inception_v3': 'adv_inception_v3',
            'inception_v3': 'InceptionV3',
            # 'inception_v4': 'InceptionV4',
            # 'resnet_v1_50' : 'resnet_v1_50',
            # 'resnet_v2_152' : 'resnet_v2_152',
            # 'resnet_v1_152' : 'resnet_v1_152',
            # 'vgg_19' : 'vgg_19',
            # 'resnet_v1_101' : 'resnet_v1_101',
            # 'resnet_v2_101' : 'resnet_v2_101',
            # 'vgg_16' : 'vgg_16'
        }

        network_mapping = {
            'ens_adv_inception_resnet_v2': 'inception_resnet_v2',
            'adv_inception_v3': 'inception_v3',
            'ens3_adv_inception_v3': 'inception_v3',
            'ens4_adv_inception_v3': 'inception_v3',
        }

        model_init_start = time.time()

        clipped_x = (tf.tanh(x_noisy) + 1.0) * x_scale + x_min

        y = tf.one_hot(y_input_var, num_classes)

        losses = []
        for model_name in models_scopes.keys():

            scope = models_scopes[model_name]
            label_offset = 1 if model_name.startswith('resnet_v1') or model_name.startswith('vgg') else 0

            model_name = model_name if model_name not in network_mapping else network_mapping[model_name]

            logits = model(model_name, clipped_x, num_classes, True, label_offset=label_offset, scope=scope)

            if label_offset > 0:
                prob = tf.pad(tf.nn.softmax(logits), tf.constant([[0, 0], [1, 0]]))
                logits = tf.pad(logits, tf.constant([[0, 0], [1, 0]]))
            else:
                prob = tf.nn.softmax(logits)

            real = tf.reduce_sum(y * logits, 1)
            other = tf.reduce_max((1.0 - y) * logits - (y * 10000.0), 1)

            # if targeted:
            #     # loss_f = tf.maximum(0.0, other - real + confidence)
            loss_f = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
            # else:
            #     loss_f = tf.maximum(0.0, real - other + confidence)

            l2dist = tf.reduce_sum(tf.squared_difference(clipped_x, x_input_var), [1, 2, 3])

            if scope == 'InceptionV3':
                loss_f *= 7 * 2

            # loss = const * loss_f
            # loss += l2dist
            loss = loss_f

            losses.append(loss)

        loss = tf.add_n(losses)

        print('Model initialization time:', (time.time() - model_init_start))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=[x_noisy])

        x_adv = clipped_x

        print('Total graph init time:', (time.time() - graph_start_time))

        # Run computation
        with tf.Session() as sess:
            restore_start = time.time()

            checkpoint_mapping = {
                'ens_adv_inception_resnet_v2': 'InceptionResnetV2',
                'adv_inception_v3': 'InceptionV3',
                'ens3_adv_inception_v3': 'InceptionV3',
                'ens4_adv_inception_v3': 'InceptionV3',
            }

            var_lists = []
            joint_var_list = []
            for i, model_name in enumerate(models_scopes.keys()):
                scope = models_scopes[model_name]
                mapping_scope = scope if model_name not in checkpoint_mapping else checkpoint_mapping[model_name]

                var_list = {(mapping_scope + v.name[len(scope):][:-2]): v
                            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)}
                joint_var_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))

                var_lists.append(var_list)

            savers = [tf.train.Saver(var_lists[i]) for i, scope in enumerate(models_scopes.values())]
            # joint_saver = tf.train.Saver(joint_var_list)

            for i, model_name in enumerate(models_scopes.keys()):
                savers[i].restore(sess, model_checkpoints[model_name])
            # joint_saver.restore(sess, joint_ckpt_path)

            initialize_uninitialized(sess)
            print('Restored in:', (time.time() - restore_start))

            process_started = time.time()

            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                target_class_for_batch = (
                    [all_images_taget_class[n] for n in filenames]
                    + [0] * (FLAGS.batch_size - len(filenames)))

                _ = sess.run([x_min_init, x_max_init],
                             feed_dict={x_input: images})

                _ = sess.run([x_noisy_init_op, x_input_var_init],
                             feed_dict={x_input: images})

                _ = sess.run(y_input_var_init, feed_dict={y_input: target_class_for_batch})

                for _ in range(num_iterations):
                    _ = sess.run(optimizer)

                adv_images = sess.run(x_adv)
                save_images(adv_images, filenames, FLAGS.output_dir)

                sys.stdout.write('.')
                sys.stdout.flush()

            print()
            print('Processed in:', (time.time() - process_started))
    print('Main processed in:', (time.time() - main_start_time))


if __name__ == '__main__':
    tf.app.run()
