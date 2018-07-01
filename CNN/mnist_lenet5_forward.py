# -*- coding:utf-8 -*-

import tensorflow as tf

# 图片大小
IMAGE_SIZE = 28
# 通道数(由于是灰度图，所以通道数为１，。如果为RGB格式，则通道数应为３)
NUM_CHANNELS = 1
# 第一层卷积核的大小
CONV1_SIZE = 5
# 第一层使用了32个卷积核
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
# 第一层全连接网络有512个神经元
FC_SIZE = 512
# 第二层全连接网络有10个神经元
OUTPUT_NODE = 10


def get_weight(shape, regularizer):
    # tf.truncated_normal()　生成去掉过大偏离点的正态分布随机数
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
        return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


# 求卷积的函数，ｘ是输入的图片，ｗ是卷积核描述,strides代表滑动步长
def conv2d(x, w):
    # tf.nn.conv2d(输入描述 eg.[batch, 5, 5, 3]
    # 第一个５和第二个５代表分辨率，３代表通道数
    #           卷积核描述 eg. [3, 3, 3, 16]
    # 第一个３和第二个３代表行和列，第三个３代表通道数，16代表卷积核的个数
    #           核滑动步长 eg [1, 1, 1, 1]
    # 第一个１和最后一个１固定不变，第二个１和第三个１分别代表行和列移动步长
    #           padding='SAME' 表示以０填充
    # )
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')


# 最大池化函数,x是输入的图片，ksize是池化核大小
def max_pool_2x2(x):
    # tf.nn.max_pool(
    #           输入描述 eg. [batch, 28, 28, 6]
    #           池化核描述 eg. [1, 2, 2, 1]
    # 第一个１和最后一个１固定不变，第二个２和第三个２代表池化核的行和列
    #           池化核移动步长 eg. [1, 2, 2, 1]
    #           padding='SAME'
    # )
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# 前向传播网络结构
def forward(x, train, regularizer):
    # 初始化第一次卷积核
    conv1_w = get_weight([CONV1_SIZE,CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    # 初始化第一次偏置
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    # 进行卷积运算
    conv1 = conv2d(x, conv1_w)
    # 对conv1添加偏置
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    # 进行最大池化操作
    pool1 = max_pool_2x2(relu1)

    # 第二层卷积核的深度等于上层卷积核的个数
    conv2_w = get_weight([CONV2_SIZE,CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    # 得到pool2的维度，存到list中
    pool_shape = pool2.get_shape().as_list()
    # pool_shape[1]特征的长度,pool_shape[2]特征的宽度,pool_shape[3]特征的深度
    # 三个相乘得到所有特征点的个数
    nodels = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # pool_shape[0]是一个batch的值
    #　改变pool2到二维形状
    reshaped = tf.reshape(pool2, [pool_shape[0], nodels])

    fc1_w = get_weight([nodels, FC_SIZE], regularizer)
    fc1_b = get_bias([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    # 如果是训练阶段，则对该层的输出使用50%的dropout,即放弃使用50%的神经元
    if train:
        fc1 = tf.nn.dropout(fc1, 0.5)

    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE],regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y

