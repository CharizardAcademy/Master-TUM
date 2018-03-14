# -*- coding:UTF-8

import random
import numpy as np
import tensorflow as tf
from task1 import *

#sess = tf.InteractiveSession()


def weight_init(shape):
    weights = tf.truncated_normal(shape,stddev=0.1,dtype=tf.float32)
    return tf.Variable(weights)


def bias_init(shape):
    biases = tf.random_normal(shape,dtype=tf.float32)
    return tf.Variable(biases)


def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="VALID")


# pooling的输入x是经过非线性函数之后的
def max_pool(x):
    return tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)


def euclidean_distance(x,y):
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), 2))


# m is margin
def compute_loss(descriptor, margin):

    anchor = descriptor[0::3]
    puller = descriptor[1::3]
    pusher = descriptor[2::3]

    with tf.variable_scope("triplet_loss"):
        diff_pos = tf.reduce_sum(tf.square(tf.subtract(anchor, puller)), 1)
        #print(diff_pos)
        diff_neg = tf.reduce_sum(tf.square(tf.subtract(anchor, pusher)), 1)
        #print(diff_neg)
        Ltriplet = tf.reduce_mean(tf.maximum(0.0, 1 - diff_neg / (diff_pos + margin)))
        tf.Print(Ltriplet,[tf.shape(Ltriplet)])
        Lpair = tf.reduce_mean(diff_pos)
        #print(Lpair)
        Loss = tf.add(Ltriplet,Lpair)
        #Loss = tf.Print(Loss, [tf.shape(descriptor), tf.shape(Loss)])
        #print(Loss)
    return Loss


config = tf.ConfigProto(log_device_placement=True)

learning_rate = 0.001
margin = 0.01
x = tf.placeholder(tf.float32, [None, 63, 63, 3])
x_image = x

# Conv Layer 1 + ReLU + maxpooling
w_conv1 = weight_init([8, 8, 3, 16])  # 用了16个filter
b_conv1 = bias_init([16])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)  # conv1输出张量尺寸:56*56*16
# print(h_conv1)
h_pool1 = max_pool(h_conv1)  # 池化后张量尺寸:28*28*16
# print(h_pool1)

# Conv Layer 2 + ReLU + maxpooling
w_conv2 = weight_init([5, 5, 16, 7])  # 用了7个filter
b_conv2 = bias_init([7])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)  # conv2输出张量尺寸:24*24*7
h_pool2 = max_pool(h_conv2)  # 池化后张量尺寸：12*12*7
# print(h_pool2)

# FC Layer 1,256个神经元
# h_pool2是一个12*12*7的tensor，转换成一个一维向量

# h_pool2_flat展平的向量，维数12*12*7，展平了再扔到全连接层里面
h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 7])
# print(h_pool2_flat)
w_fc1 = weight_init([12 * 12 * 7, 256])
b_fc1 = bias_init([256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# add dropout layer,失活概率0.6
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=0.6)

# FC Layer 2,16个神经元
w_fc2 = weight_init([256, 16])
b_fc2 = bias_init([16])
with tf.name_scope("h_fc2"):
    h_fc2 = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
# print(h_fc2)

loss = compute_loss(h_fc2, margin)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# Session
# 初始化全部参数

with tf.Session(config=config) as sess:
    saver = tf.train.Saver()

    sess.run(tf.initialize_all_variables())
    loss_bank = []
    # 跑100个epoch
    for j in range(100):
        print("Batch generating...")
        batch = batch_generator(50, Ptrain, Pdb, Strain, Sdb)
        print("Batch generating done.")
        # 每个batch跑200个step
        for i in range(100):
            data = batch  # 把数据传进来
            _, myloss = sess.run([optimizer, loss], feed_dict={x: data})
            loss_bank.append(myloss)
            print("The loss at epoch", j, "step", i, "is", myloss)

    print("Traning done.")

    saver.save(sess, "/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/ex3")


"""
# plot the loss curve for training
plt.plot(loss_bank)
plt.show()
"""














