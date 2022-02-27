# -*-coding: utf-8 -*-
"""
    @Project: tf-face-recognition
    @File   : device.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-11-17 10:39:28
"""
import random
import os
import numpy as np
import tensorflow as tf


def set_device_memory(gpu_fraction=0.90, eager_execution=True):
    """
    :param gpu_fraction:
    :param disable_eager_execution:  BUG:Attempting to capture an EagerTensor without building a function.
    :return:
    """
    if not eager_execution:
        tf.compat.v1.disable_eager_execution()
    tf_config = tf.compat.v1.ConfigProto()
    # tf_config.log_device_placement = False
    # tf_config.allow_soft_placement = allow_soft_placement
    # tf_config.gpu_options.allow_growth = allow_growth
    # tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction  # 占用85%显存
    gpu_id = tf.config.experimental.list_physical_devices('GPU')
    if gpu_id:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpu_id:
                tf.config.experimental.set_memory_growth(gpu, True)
                print("use gpu id:{}".format(gpu.name))
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpu_id), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def set_env_random_seed(seed=2020):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)  # sets the global seed.


def plot_model(model, input_shape=None, plot=False):
    """
    Usage:
        input_size = [256, 192]
        input_shape = [1, input_size[1], input_size[0], 3]
        model.build(input_shape=tuple(input_shape))
    :param network:
    :param input_shape:
    :return:
    """
    if input_shape:
        model.build(input_shape=tuple(input_shape))
    model.summary()
    if plot:
        model_image = "./model_train.png"
        tf.keras.utils.plot_model(model, to_file=model_image, show_layer_names=True, show_shapes=True)


def summary_model(model, batch_size=1, input_size=[112, 112], plot=True):
    """
    Return number of parameters and flops.
    常见模型的参数量和FLOPs:
    - https://arxiv.org/pdf/1807.11164.pdf
    - https://www.jianshu.com/p/29d74d7a954a
    1 GFLOPs = 10^9 FLOPs
    1 MFLOPs = 10^6 FLOPs
    :param model:
    :param input_shape:[batch size ,H,W,D]
    :return:
    """
    input_shape = tuple([batch_size, input_size[1], input_size[0], 3])
    # inputs = tf.random.uniform(input_shape)
    inputs = tf.random.uniform(input_shape, dtype=tf.float32, minval=0, maxval=200)
    output = model(inputs)
    flops = get_model_flops(model, input_shape)
    # inputs = tf.keras.layers.Input(shape=tuple(input_shape[1:]))
    # outputs = model(inputs)
    # model = tf.keras.Model(inputs, outputs)
    plot_model(model, plot=plot)
    nparams = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    options['output'] = 'none'
    nparams = float(nparams) * 1e-6
    flops = float(flops) * 1e-6  #
    print('total_params {:7.4f} M'.format(nparams))
    print('total_flops  {:7.4f} MFLOPs'.format(flops))
    print("===" * 10)
    print("inputs.shape:{}".format(inputs.shape))
    print("output.shape:{}".format(output.shape))
    return nparams, flops


def get_model_flops(model, input_shape):
    """
    https://github.com/tensorflow/tensorflow/issues/39834
    :param model: TF-Keras Model
    :param input_shape:[batch size ,H,W,D]
    :return:
    """
    model_path = "tmp.h5"
    # tf.keras.models.Model.save(model, model_path, include_optimizer=False)
    # flops = get_model_file_flops(model_path, input_shape)
    flops = model_flops(model, input_shape)
    return flops


def model_flops(model, input_shape):
    """
    https://github.com/tensorflow/tensorflow/issues/39834
    :param h5_file: "path/to/tf_keras.h5"
    :param input_shape:[batch size ,H,W,D]
    :return:
    """
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        with session.as_default():
            # model = tf.keras.models.load_model(h5_file, compile=False, custom_objects={'tf': tf})
            # inputs = tf.keras.layers.Input(shape=input_shape)
            inputs = tf.random.uniform(input_shape, dtype=tf.float32, minval=0, maxval=200)
            outputs = model(inputs)
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops


def get_model_file_flops(h5_file, input_shape):
    """
    https://github.com/tensorflow/tensorflow/issues/39834
    :param h5_file: "path/to/tf_keras.h5"
    :param input_shape:[batch size ,H,W,D]
    :return:
    """
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(h5_file, compile=False, custom_objects={'tf': tf})
            # inputs = tf.keras.layers.Input(shape=input_shape)
            inputs = tf.random.uniform(input_shape, dtype=tf.float32, minval=0, maxval=200)
            outputs = model(inputs)
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops


def MyNet(input_shape=(224, 224, 3)):
    '''
    构建一个CNN网络模型
    :param input_shape: 指定输入维度input_shape=(H,W,D)
    :return:
    '''
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=(2, 2),
                                     padding='same', activation=tf.nn.relu, input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
    return model


if __name__ == "__main__":
    """
    total_params 33.5768 M
    total_flops  1059.3053 MFLOPs
    """
    set_device_memory()
    input_size = [224, 224]
    model = MyNet()
    nparams, flops = summary_model(model, batch_size=1, input_size=input_size)
    # h5_file = "path/to/tf_keras.h5"
    # flops = get_model_file_flops(h5_file, input_shape)
    # print(flops)
