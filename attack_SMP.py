"""Implementation of attack."""
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
from keras.utils import to_categorical
import utils
import os
import numpy as np
from skimage import io, color, transform
from ctypes import cdll, c_double, POINTER, c_int, c_float

slim = tf.contrib.slim
tf.flags.DEFINE_string('model_name', 'vgg_16', 'The Model used to generate adv')

tf.flags.DEFINE_string('attack_method', 'SMP', 'The name of attack method.')

tf.flags.DEFINE_string('layer_name', 'vgg_16/conv3/conv3_3/Relu', 'The layer to be attacked.')

tf.flags.DEFINE_string('input_dir', './dataset/images/', 'Input directory with images.')

tf.flags.DEFINE_string('output_dir', './adv/vgg16/', 'Output directory with images.')

tf.flags.DEFINE_string('superpixels_dir', '224', 'superpixels directory with npy.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('num_iter', 1, 'Number of iterations.')

tf.flags.DEFINE_float('alpha', 1.6, 'Step size.')

tf.flags.DEFINE_integer('batch_size', 1, 'How many images process at one time.')

tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_string('GPU_ID', '0', 'which GPU to use.')

"""parameter for DIM"""
tf.flags.DEFINE_integer('image_size', 224, 'size of each input images.')      

tf.flags.DEFINE_integer('image_resize', 250, 'size of each diverse images.')

tf.flags.DEFINE_float('prob', 0.7, 'Probability of using diverse inputs.')    

"""parameter for TIM"""
tf.flags.DEFINE_integer('Tkern_size', 15, 'Kernel size of TIM.')

"""parameter for PIM"""
tf.flags.DEFINE_float('amplification_factor', 2.5, 'To amplifythe step size.')

tf.flags.DEFINE_float('gamma', 0.5, 'The gamma parameter.')

tf.flags.DEFINE_integer('Pkern_size', 3, 'Kernel size of PIM.')

"""parameter for FIA"""
tf.flags.DEFINE_float('ens', 60, 'Number of random mask input.') 

tf.flags.DEFINE_float('probb', 0.7, 'keep probability = 1 - drop probability.')

"""parameter for SMP"""
tf.flags.DEFINE_list("super_next_dir", ['2000_40','1000_30','500_20','2000_50','1000_40','500_30'], "List of directories for superpixel patches.")

FLAGS = tf.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_ID
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def get_opt_layers(layer_name):
    """obtain the feature map of the target layer"""
    opt_operations = []
    operations = tf.get_default_graph().get_operations()
    for op in operations:
        if layer_name == op.name:
            opt_operations.append(op.outputs[0])
            shape = op.outputs[0][:FLAGS.batch_size].shape
            break
    return opt_operations, shape

"""the loss function for FDA"""

def get_fda_loss(opt_operations):
    loss = 0
    for layer in opt_operations:
        batch_size = FLAGS.batch_size
        tensor = layer[:batch_size]
        mean_tensor = tf.stack([tf.reduce_mean(tensor, -1), ] * tensor.shape[-1], -1)
        wts_good = tensor < mean_tensor
        wts_good = tf.to_float(wts_good)
        wts_bad = tensor >= mean_tensor
        wts_bad = tf.to_float(wts_bad)
        loss += tf.log(tf.nn.l2_loss(wts_good * (layer[batch_size:]) / tf.cast(tf.size(layer), tf.float32)))
        loss -= tf.log(tf.nn.l2_loss(wts_bad * (layer[batch_size:]) / tf.cast(tf.size(layer), tf.float32)))
    loss = loss / len(opt_operations)
    return loss

"""the loss function for NRDM"""

def get_nrdm_loss(opt_operations):
    loss = 0
    for layer in opt_operations:
        ori_tensor = layer[:FLAGS.batch_size]
        adv_tensor = layer[FLAGS.batch_size:]
        loss += tf.norm(ori_tensor - adv_tensor) / tf.cast(tf.size(layer), tf.float32)
    loss = loss / len(opt_operations)
    return loss

"""the loss function for FIA"""

def get_fia_loss(opt_operations, weights):
    loss = 0
    for layer in opt_operations:
        ori_tensor = layer[:FLAGS.batch_size]
        adv_tensor = layer[FLAGS.batch_size:]
        loss += tf.reduce_sum(adv_tensor * weights) / tf.cast(tf.size(layer), tf.float32)
        # loss += tf.reduce_sum((weights*tf.abs(adv_tensor-ori_tensor))) / tf.cast(tf.size(layer), tf.float32)
    loss = loss / len(opt_operations)
    return loss

"""the loss function for SMP"""

def get_smp_loss(opt_operations, weights):
    loss = 0
    for layer in opt_operations:
        ori_tensor = layer[:FLAGS.batch_size]
        adv_tensor = layer[FLAGS.batch_size:]

        loss += tf.reduce_sum(adv_tensor * weights) / tf.cast(tf.size(layer), tf.float32)
        #loss += tf.reduce_sum(tf.abs(adv_tensor - ori_tensor) * weights) / tf.cast(tf.size(layer), tf.float32)
    loss = loss / len(opt_operations)
    return loss

def normalize(grad, opt=2):
    if opt == 0:
        nor_grad = grad
    elif opt == 1:
        abs_sum = np.sum(np.abs(grad), axis=(1, 2, 3), keepdims=True)
        nor_grad = grad / abs_sum
    elif opt == 2:
        square = np.sum(np.square(grad), axis=(1, 2, 3), keepdims=True)
        nor_grad = grad / np.sqrt(square)
    return nor_grad

def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern]).swapaxes(0, 2)
    stack_kern = np.expand_dims(stack_kern, 3)
    return stack_kern, kern_size // 2

def project_noise(x, stack_kern, kern_size):
    x = tf.pad(x, [[0, 0], [kern_size, kern_size], [kern_size, kern_size], [0, 0]], "CONSTANT")
    x = tf.nn.depthwise_conv2d(x, stack_kern, strides=[1, 1, 1, 1], padding='VALID')
    return x

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)
    return stack_kernel


def input_diversity(input_tensor):
    """Input diversity: https://arxiv.org/abs/1803.06978"""
    rnd = tf.random_uniform((), FLAGS.image_size, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_size, FLAGS.image_size],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret

def slic_segment(image, K, M, prob, so_path="./SLICSP.so"):
    if isinstance(image, str):
        rgb = io.imread(image)
    else:
        rgb = image
    rgb_resized = transform.resize(rgb, (224, 224), mode='reflect')
    data = color.rgb2lab(rgb_resized)
    image_height, image_width = data.shape[:2]
    L = data[:,:,0].flatten()
    A = data[:,:,1].flatten()
    B = data[:,:,2].flatten()
    labels = np.zeros(image_width * image_height, dtype=np.float64)
    X_mask = np.ones((1, image_height, image_width, 3), dtype=np.float32)
    lib = cdll.LoadLibrary(so_path)
    L_ptr = L.ctypes.data_as(POINTER(c_double))
    A_ptr = A.ctypes.data_as(POINTER(c_double))
    B_ptr = B.ctypes.data_as(POINTER(c_double))
    labels_ptr = labels.ctypes.data_as(POINTER(c_double))
    X_mask_ptr = X_mask.ctypes.data_as(POINTER(c_float))
    lib.SLICSP.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double),
                          c_int, c_int, c_int, c_double, c_double, POINTER(c_double), POINTER(c_float)]
    lib.SLICSP(L_ptr, A_ptr, B_ptr,
               c_int(image_width), c_int(image_height),
               c_int(K), c_double(M), c_double(prob), labels_ptr, X_mask_ptr)
    return labels.reshape((image_height, image_width)), K, X_mask

def patch_superpixel(img_shape, super_dir, prob, images):
    super_params = super_dir.split("_")
    K = int(super_params[0])
    M = float(super_params[1])

    super_pixels, num_labels, X_mask = slic_segment(images[0], K, M, prob)
    return X_mask

def setup_seed(rnd_seed):
    np.random.seed(rnd_seed)
    random.seed(rnd_seed)
    tf.set_random_seed(rnd_seed)

P_kern, kern_size = project_kern(FLAGS.Pkern_size)
T_kern = gkern(FLAGS.Tkern_size)

def main(_):
    if FLAGS.model_name in ['vgg_16', 'vgg_19', 'resnet_v1_50', 'resnet_v1_152','mobilenet_v1']:
        eps = FLAGS.max_epsilon
        alpha = FLAGS.alpha
    else:
        eps = 2.0 * FLAGS.max_epsilon / 255.0
        alpha = FLAGS.alpha * 2.0 / 255.0

    num_iter = FLAGS.num_iter
    momentum = FLAGS.momentum
    setup_seed(1234)

    image_preprocessing_fn = utils.normalization_fn_map[FLAGS.model_name]
    inv_image_preprocessing_fn = utils.inv_normalization_fn_map[FLAGS.model_name]
    batch_shape = [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3]
    checkpoint_path = utils.checkpoint_paths[FLAGS.model_name]
    layer_name = FLAGS.layer_name
    super_dir=FLAGS.super_next_dir

    with tf.Graph().as_default():
        # Prepare graph
        ori_input = tf.placeholder(tf.float32, shape=batch_shape)
        adv_input = tf.placeholder(tf.float32, shape=batch_shape)
        num_classes = 1000 + utils.offset[FLAGS.model_name]
        label_ph = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * 2, num_classes])
        accumulated_grad_ph = tf.placeholder(dtype=tf.float32, shape=batch_shape)
        amplification_ph = tf.placeholder(dtype=tf.float32, shape=batch_shape)

        network_fn = utils.nets_factory.get_network_fn(FLAGS.model_name, num_classes=num_classes, is_training=False)
        x = tf.concat([ori_input, adv_input], axis=0)

        # whether using DIM or not
        if 'DI' in FLAGS.attack_method:
            logits, end_points = network_fn(input_diversity(x))
        else:
            logits, end_points = network_fn(x)  

        opt_operations, shape= get_opt_layers(layer_name)
        weights_ph = tf.placeholder(dtype=tf.float32, shape=shape)

        # select the loss function
        if 'FDA' in FLAGS.attack_method:
            loss = get_fda_loss(opt_operations)
        elif 'NRDM' in FLAGS.attack_method:
            loss = get_nrdm_loss(opt_operations)
        elif 'FIA' in FLAGS.attack_method:
            weights_tensor = tf.gradients(logits * label_ph, opt_operations[0])[0]
            loss = get_fia_loss(opt_operations, weights_ph)
        else:   
            weights_tensor = tf.gradients(logits * label_ph, opt_operations[0])[0]
            loss = get_smp_loss(opt_operations, weights_ph)

        gradient = tf.gradients(loss, adv_input)[0]

        noise = gradient
        adv_input_update = adv_input
        amplification_update = amplification_ph

        # whether using TIM or not
        if 'TI' in FLAGS.attack_method:
            noise = tf.nn.depthwise_conv2d(noise, T_kern, strides=[1, 1, 1, 1], padding='SAME')

        # the default optimization process with momentum
        noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
        noise = momentum * accumulated_grad_ph + noise

        # whether using PIM or not
        if 'PI' in FLAGS.attack_method:
            # amplification factor
            alpha_beta = alpha * FLAGS.amplification_factor
            gamma = FLAGS.gamma * alpha_beta

            # Project cut noise
            amplification_update += alpha_beta * tf.sign(noise)
            cut_noise = tf.clip_by_value(abs(amplification_update) - eps, 0.0, 10000.0) * tf.sign(amplification_update)
            projection = gamma * tf.sign(project_noise(cut_noise, P_kern, kern_size))

            # Occasionally, when the adversarial examples are crafted for an ensemble of networks with residual block by combined methods,
            # you may neet to comment the following line to get better result.
            amplification_update += projection

            adv_input_update = adv_input_update + alpha_beta * tf.sign(noise) + projection
        else:
            adv_input_update = adv_input_update + alpha * tf.sign(noise)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
            count = 0

            for images, names, labels in utils.load_image(FLAGS.input_dir, FLAGS.image_size, FLAGS.batch_size):
                count += FLAGS.batch_size
                if count % 100 == 0:
                    print("Generating:", count)

                images_tmp = image_preprocessing_fn(np.copy(images))
                if FLAGS.model_name in ['resnet_v1_50', 'resnet_v1_152', 'vgg_16', 'vgg_19']:
                    labels = labels - 1

                # obtain true label
                labels = to_categorical(np.concatenate([labels, labels], axis=-1), num_classes)

                # add some noise to avoid F_{k}(x)-F_{k}(x')=0
                if 'NRDM' in FLAGS.attack_method:
                    images_adv = images + np.random.normal(0, 0.1, size=np.shape(images))
                else:
                    images_adv = images

                images_adv = image_preprocessing_fn(np.copy(images_adv))

                grad_np = np.zeros(shape=batch_shape)
                amplification_np = np.zeros(shape=batch_shape)
                weight_np = np.zeros(shape=shape)

                for i in range(num_iter):
                    if i == 0 and 'SMP' in FLAGS.attack_method:
                        if FLAGS.ens == 0:
                            images_tmp2 = image_preprocessing_fn(np.copy(images))
                            w, feature = sess.run([weights_tensor, opt_operations[0]],
                                                  feed_dict={ori_input: images_tmp2, adv_input: images_tmp2,
                                                             label_ph: labels})
                            weight_np = w[:FLAGS.batch_size]
                        # use ensemble masked image to obtain weights
                        for l in range(int(FLAGS.ens)):
                            if l % len(super_dir) == 0:
                                mask1 = np.random.binomial(1, FLAGS.probb, size=(batch_shape[0],
                                                                                 batch_shape[1],
                                                                                 batch_shape[2],
                                                                                 batch_shape[3]))
                                mask2 = np.random.uniform(0, 1, size=(batch_shape[0],
                                                                      batch_shape[1],
                                                                      batch_shape[2],
                                                                      batch_shape[3]))
                                mask = np.where(mask1 == 1, 1, mask2)
                            else:
                                super_idx = (l % len(super_dir)) - 1  
                                super_idx = super_idx % len(super_dir)  
                                mask = patch_superpixel((batch_shape[0], batch_shape[1], batch_shape[2],
                                                         batch_shape[3]), super_dir[super_idx], FLAGS.probb,images)
                            images_tmp2 = images * mask
                            images_tmp2 = image_preprocessing_fn(np.copy(images_tmp2))

                            w, feature = sess.run([weights_tensor, opt_operations[0]],
                                                  feed_dict={ori_input: images_tmp2, adv_input: images_tmp2,
                                                             label_ph: labels})
                            weight_np = weight_np + w[:FLAGS.batch_size]
                            
                        weight_np = -normalize(weight_np, 2)
                    elif i == 0 and 'FIA' in FLAGS.attack_method:
                        # only use original image to obtain weights
                        if FLAGS.ens == 0:
                            images_tmp2 = image_preprocessing_fn(np.copy(images))
                            w, feature = sess.run([weights_tensor, opt_operations[0]],
                                                  feed_dict={ori_input: images_tmp2, adv_input: images_tmp2,
                                                             label_ph: labels})
                            weight_np = w[:FLAGS.batch_size]
                        # use ensemble masked image to obtain weights
                        for l in range(int(FLAGS.ens)):
                            # generate the random mask
                            mask = np.random.binomial(1, FLAGS.probb, size=(
                            batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3]))
                            images_tmp2 = images * mask
                            images_tmp2 = image_preprocessing_fn(np.copy(images_tmp2))
                            w, feature = sess.run([weights_tensor, opt_operations[0]],
                                                  feed_dict={ori_input: images_tmp2, adv_input: images_tmp2,
                                                             label_ph: labels})
                            weight_np = weight_np + w[:FLAGS.batch_size]
                        # normalize the weights
                        weight_np = -normalize(weight_np, 2)

                    # optimization
                    images_adv, grad_np, amplification_np = sess.run([adv_input_update, noise, amplification_update],
                                                                     feed_dict={ori_input: images_tmp,
                                                                                adv_input: images_adv,
                                                                                weights_ph: weight_np,
                                                                                label_ph: labels,
                                                                                accumulated_grad_ph: grad_np,
                                                                                amplification_ph: amplification_np})
                    images_adv = np.clip(images_adv, images_tmp - eps, images_tmp + eps)
                
                images_adv = inv_image_preprocessing_fn(images_adv)
                
                utils.save_image(images_adv, names, FLAGS.output_dir)

if __name__ == '__main__':
    tf.app.run()
