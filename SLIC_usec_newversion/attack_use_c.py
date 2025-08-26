"""Implementation of attack."""
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
from keras.utils import to_categorical
#import time
import utils
import os
import time
#from scipy import misc
#from scipy import ndimage
#import PIL
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
#import io
import ctypes 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from SLIC_init import slic_segment

slim = tf.contrib.slim
tf.flags.DEFINE_list("super_next_dir", ['2000_40','1000_30','500_20','2000_50','1000_40','500_30'], "List of directories for superpixel patches.")

tf.flags.DEFINE_string('model_name', 'vgg_16', 'The Model used to generate adv')
#tf.flags.DEFINE_string('model_name', 'inception_v3', 'The Model used to generate adv.')

tf.flags.DEFINE_string('attack_method', 'RPA', 'The name of attack method.')
# vgg_16/conv1/conv1_1/Relu  vgg_16/conv3/conv3_3/Relu
tf.flags.DEFINE_string('layer_name', 'vgg_16/conv3/conv3_3/Relu', 'The layer to be attacked.')
# tf.flags.DEFINE_string('layer_name2', 'InceptionV3/InceptionV3/Mixed_5b/concat', 'The layer to be attacked.')

# tf.flags.DEFINE_string('input_dir', './adv/time/timeours2it1/', 'Input directory with images.')
tf.flags.DEFINE_string('input_dir', './dataset/images/', 'Input directory with images.')


tf.flags.DEFINE_string('output_dir', './adv/vgg16/usec_time/ours/', 'Output directory with images.')

tf.flags.DEFINE_string('superpixels_dir', '224', 'superpixels directory with npy.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('num_iter', 1, 'Number of iterations.')

tf.flags.DEFINE_float('alpha', 1.6, 'Step size.')

tf.flags.DEFINE_integer('batch_size', 1, 'How many images process at one time.')

tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_string('GPU_ID', '2', 'which GPU to use.')

"""parameter for DIM"""
tf.flags.DEFINE_integer('image_size', 224, 'size of each input images.')       # RPA 299 330  FIA 224 250

tf.flags.DEFINE_integer('image_resize', 250, 'size of each diverse images.')

tf.flags.DEFINE_float('prob', 0.7, 'Probability of using diverse inputs.')      #FIA 0.7 RPA0.6

"""parameter for TIM"""
tf.flags.DEFINE_integer('Tkern_size', 15, 'Kernel size of TIM.')

"""parameter for PIM"""
tf.flags.DEFINE_float('amplification_factor', 2.5, 'To amplifythe step size.')

tf.flags.DEFINE_float('gamma', 0.5, 'The gamma parameter.')

tf.flags.DEFINE_integer('Pkern_size', 3, 'Kernel size of PIM.')

"""parameter for FIA"""
tf.flags.DEFINE_float('ens', 60, 'Number of random mask input.') #RPA60  FIA30

tf.flags.DEFINE_float('probb', 0.7, 'keep probability = 1 - drop probability.')   #FIA0.9 RPA0.7

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

def get_rpa_loss(opt_operations, weights):
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

def patch_by_strides(img_shape, patch_size, prob):
    X_mask = np.ones(img_shape)
    N0, H0, W0, C0 = X_mask.shape
    ph = H0 // patch_size[0]
    pw = W0 // patch_size[1]
    X = X_mask[:, :ph * patch_size[0], :pw * patch_size[1]]
    N, H, W, C = X.shape
    shape = (N, ph, pw, patch_size[0], patch_size[1], C)
    strides = (X.strides[0], X.strides[1] * patch_size[0], X.strides[2] * patch_size[0], *X.strides[1:])
    mask_patchs = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    mask_len = mask_patchs.shape[1] * mask_patchs.shape[2] * mask_patchs.shape[-1]
    ran_num = int(mask_len * (1 - prob))
    rand_list = np.random.choice(mask_len, ran_num, replace=False)
    for i in range(mask_patchs.shape[1]):
        for j in range(mask_patchs.shape[2]):
            for k in range(mask_patchs.shape[-1]):
                if i * mask_patchs.shape[2] * mask_patchs.shape[-1] + j * mask_patchs.shape[-1] + k in rand_list:
                    mask_patchs[:, i, j, :, :, k] = np.random.uniform(0, 1,
                                                                      (N, mask_patchs.shape[3], mask_patchs.shape[4]))
    img2 = np.concatenate(mask_patchs, axis=0, )
    img2 = np.concatenate(img2, axis=1)
    img2 = np.concatenate(img2, axis=1)
    img2 = img2.reshape((N, H, W, C))
    X_mask[:, :ph * patch_size[0], :pw * patch_size[1]] = img2
    return X_mask


# def patch_superpixel(img_shape, num, super_dir, prob, super_d, images):
#     super_params = super_dir.split("_")
#     n_seeds = int(super_params[0])  # 超像素的数量
#     cm = float(super_params[1])     # 超像素分割的紧凑性参数
    
#     # Get unique superpixels and num_labels
#     super_pixels, num_labels = slicsp_segmentation(images[0], n_seeds, cm)

#     # Create a tensor X_mask filled with ones 
#     X_mask = np.ones(img_shape)
    
#     # Convert to correct format and pass num_labels to C++ function
#     super_pixels = super_pixels.astype(np.float64)
#     X_mask = X_mask.astype(np.float64)
    
#     slicsp_lib.GenerateSuperpixelMask(
#         super_pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         X_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         img_shape[2],  # width
#         img_shape[1],  # height  
#         img_shape[0],  # batch_size
#         prob,          # probability
#         num_labels     # pass the num_labels we already have
#     )
    
#     return np.ascontiguousarray(X_mask)


def patch_superpixel(img_shape, num, super_dir, prob, images):

    
    super_params = super_dir.split("_")
    K = int(super_params[0])
    M = float(super_params[1])

    super_pixels, num_labels = slic_segment(images[0], K, M)

    X_mask = np.ones(img_shape)

    # Randomly select clusters to mask and assign random values to each pixel within the cluster
    mask_clusters = np.random.choice(num_labels, size=int(num_labels * (1-prob)), replace=False)

    for i in range(img_shape[0]):
        for cluster in mask_clusters:
            cluster_indices = np.where(super_pixels == cluster) 
            random_values = np.random.rand(len(cluster_indices[0]))  
            X_mask[i][cluster_indices] = random_values[:, np.newaxis]  
    return X_mask


def setup_seed(rnd_seed):
    np.random.seed(rnd_seed)
    random.seed(rnd_seed)
    tf.set_random_seed(rnd_seed)


P_kern, kern_size = project_kern(FLAGS.Pkern_size)
T_kern = gkern(FLAGS.Tkern_size)


def main(_):
    #start_time = time.time()  # 记录程序开始运行的时间
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
    super_d=FLAGS.superpixels_dir
    super_dir=FLAGS.super_next_dir #
    
    # 获取当前日期，格式为月日
    current_date = datetime.now().strftime('%m%d')

    # 创建子文件夹，如果不存在的话
    subfolder_path = os.path.join("output", current_date)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    # 初始化时间记录数组
    time_records = []
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
            logits, end_points = network_fn(x)  #logits形状[batch_size, num_classes]

        problity = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(logits, axis=1)  #预测 一维张量
        one_hot = tf.one_hot(pred, num_classes)  #将预测结果转化编码 ；num_classes 是分类的类别数目，即 one-hot 编码的长度

        entropy_loss = tf.losses.softmax_cross_entropy(one_hot[:FLAGS.batch_size], logits[FLAGS.batch_size:])
        #真实标签的 one-hot 编码；模型的预测结果；
        # opt_operations,opt_operations2, shape,shape2 = get_opt_layers(layer_name,layer_name2)
        opt_operations, shape= get_opt_layers(layer_name)
        weights_ph = tf.placeholder(dtype=tf.float32, shape=shape)
        # weights_ph2 = tf.placeholder(dtype=tf.float32, shape=shape2)

        #opt_operationsm, shapem = get_opt_layers(layer_name)
        #weights_phm = tf.placeholder(dtype=tf.float32, shape=shape)

        # select the loss function
        if 'FDA' in FLAGS.attack_method:
            loss = get_fda_loss(opt_operations)
        elif 'NRDM' in FLAGS.attack_method:
            loss = get_nrdm_loss(opt_operations)
        elif 'FIA' in FLAGS.attack_method:
            weights_tensor = tf.gradients(logits * label_ph, opt_operations[0])[0]
            loss = get_fia_loss(opt_operations, weights_ph)
        else:   #RPA方法的loss定义
            weights_tensor = tf.gradients(logits * label_ph, opt_operations[0])[0]
            loss = get_rpa_loss(opt_operations, weights_ph)
            # weights_tensor2 = tf.gradients(logits * label_ph, opt_operations2[0])[0]
            # loss2 = get_rpa_loss(opt_operations2, weights_ph2)

        gradient = tf.gradients(loss, adv_input)[0]
        # gradient2 = tf.gradients(loss2, adv_input)[0]

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
                start_time = time.time()  # 开始时间
                count += FLAGS.batch_size
                #print(count)
                if count % 100 == 0:
                    print("Generating:", count)
                # if names[0] != '1.png':
                #     continue
                
                # 检查是否是1-10.png的图片
                # current_img = int(names[0].split('.')[0])  # 获取图片编号
                # if current_img < 1 or current_img > 10:  # 只处理1-10.png
                #     continue

                images_tmp = image_preprocessing_fn(np.copy(images))
                if FLAGS.model_name in ['resnet_v1_50', 'resnet_v1_152', 'vgg_16', 'vgg_19']:
                    labels = labels - 1

                # obtain true label
                labels = to_categorical(np.concatenate([labels, labels], axis=-1), num_classes)
                # labels = sess.run(one_hot, feed_dict={ori_input: images_tmp, adv_input: images_tmp})

                # add some noise to avoid F_{k}(x)-F_{k}(x')=0
                if 'NRDM' in FLAGS.attack_method:
                    images_adv = images + np.random.normal(0, 0.1, size=np.shape(images))
                else:
                    images_adv = images

                images_adv = image_preprocessing_fn(np.copy(images_adv))

                grad_np = np.zeros(shape=batch_shape)
                amplification_np = np.zeros(shape=batch_shape)
                weight_np = np.zeros(shape=shape)
                # weight_np2 = np.zeros(shape=shape2)
                #sess1=tf.Session()

                #imageupdate = input_diversity(images)
                #imageupdate = new_input_diversity(images)
                #images_tmp3 = sess1.run(imageupdate)
                #images_tmp3 = image_preprocessing_fn(np.copy(images_tmp3))


                for i in range(num_iter):
                    all_weights_np = []
                    update_weights_np = []
                    if i == 0 and 'RPA' in FLAGS.attack_method:
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
                                super_idx = (l % len(super_dir)) - 1  # Adjust for 1-based index
                                super_idx = super_idx % len(super_dir)  # Ensure within bounds of super_dir
                                mask = patch_superpixel((batch_shape[0], batch_shape[1], batch_shape[2],
                                                         batch_shape[3]), count, super_dir[super_idx], FLAGS.probb,images)
                            images_tmp2 = images * mask
                            images_tmp2 = image_preprocessing_fn(np.copy(images_tmp2))

                            w, feature = sess.run([weights_tensor, opt_operations[0]],
                                                  feed_dict={ori_input: images_tmp2, adv_input: images_tmp2,
                                                             label_ph: labels})
                            weight_np = weight_np + w[:FLAGS.batch_size]#形状(1, 35, 35, 256)
                            
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
                end_time = time.time()  # 记录每张图片的结束时间
                time_records.append(end_time - start_time)  # 计算处理时间并添加到列表中

                # # 打印图1和图2的处理时间
                # if count == 1:
                #     print(f"Time for Image 1: {end_time - start_time:.4f} seconds")
                # elif count == 2:
                #     print(f"Time for Image 2: {end_time - start_time:.4f} seconds")
                # elif count == 3:
                #     print(f"Time for Image 2: {end_time - start_time:.4f} seconds")
                # elif count == 4:
                #     print(f"Time for Image 2: {end_time - start_time:.4f} seconds")

                # if count >= 4:  # 如果只需要处理前两张图片，可以在这里退出循环
                #     break
    # end_time = time.time()  # 记录程序结束运行的时间
    # elapsed_time = end_time - start_time  # 计算运行时间
    # print(f"程序运行时间：{elapsed_time:.2f} 秒")  # 输出运行时间
    # 计算平均值和方差
    average_time = np.mean(time_records)
    variance_time = np.var(time_records)

    # 将结果写入到文本文件中
    output_file = os.path.join(FLAGS.output_dir, "time.txt")
    with open(output_file, "a") as f:
        f.write("Time records for each image (in seconds):\n")
        for i, t in enumerate(time_records):
            f.write(f"Image {i+1}: {t:.4f} s\n")
        f.write(f"\nAverage time: {average_time:.4f} s\n")
        f.write(f"Variance: {variance_time:.4f}\n")

    print(f"Time records saved to {output_file}")
if __name__ == '__main__':
    tf.app.run()
