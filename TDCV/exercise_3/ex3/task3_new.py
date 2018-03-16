# -*- coding：UTF-8

import tensorflow as tf
from functions import *
from task1 import *
import os


def weight_init(shape):
    with tf.name_scope("weight"):
        weights = tf.truncated_normal(shape,stddev=0.1,dtype=tf.float32)
    return tf.Variable(weights)


def bias_init(shape):
    with tf.name_scope("bias"):
        biases = tf.random_normal(shape,dtype=tf.float32)
    return tf.Variable(biases)


def conv2d(x,w):
    with tf.name_scope("conv_layer"):
        return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="VALID")


# pooling的输入x是经过非线性函数之后的
def max_pool(x):
    with tf.name_scope("max_pooling"):
        return tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)


def euclidean_distance(vec1, vec2):
    with tf.name_scope("euclidean_distance"):
        return np.sqrt(np.sum(np.square(vec1 - vec2)))


# m is margin
def compute_loss(descriptor, margin):

    anchor = descriptor[0::3]
    puller = descriptor[1::3]
    pusher = descriptor[2::3]

    with tf.variable_scope("triplet_loss"):
        diff_pos = tf.reduce_sum(tf.square(tf.subtract(anchor, puller)), 1)
        diff_neg = tf.reduce_sum(tf.square(tf.subtract(anchor, pusher)), 1)
        Ltriplet = tf.reduce_mean(tf.maximum(0.0, 1 - diff_neg / (diff_pos + margin)))
        tf.Print(Ltriplet,[tf.shape(Ltriplet)])
        Lpair = tf.reduce_mean(diff_pos)
        Loss = tf.add(Ltriplet,Lpair)

    return Loss


def nearest_neighbor(Stest, Sdb):
    angle = []
    for i in range(0, 3530):
        distance_bank = []
        for j in range(0, 1335):
            distance = euclidean_distance(Stest[i,:], Sdb[j,:])
            distance_bank.append(distance)
        distance_min = np.argmin(distance_bank) # 返回db里最相似的template的下标
        if i % 5 == distance_min % 5: # 说明是一个类
            Stest_norm = Stest[i,:].dot(Stest[i,:])
            Sdb_norm = Sdb[distance_min,:].dot(Sdb[distance_min,:])
            cos_theta = Stest[i,:].dot(Sdb[distance_min,:]) / (Stest_norm * Sdb_norm)
            theta = np.arccos(cos_theta)
            theta = theta * 360/2/np.pi
            angle.append(theta)
        else:
            continue

    return angle


def bins_assignment(angle_list):
    bin10 = 0
    bin20 = 0
    bin40 = 0
    bin180 = 0

    for angle in angle_list:
        if angle <= 10.0:
            bin10 = bin10 + 1
            bin20 = bin20 + 1
            bin40 = bin40 + 1
            bin180 = bin180 + 1
        elif angle > 10.0 and angle <= 20.0:
            bin20 = bin20 + 1
            bin40 = bin40 + 1
            bin180 = bin180 + 1
        elif angle > 20.0 and angle <= 40.0:
            bin40 = bin40 + 1
            bin180 = bin180 + 1
        else:
            bin180 = bin180 + 1

    return bin10, bin20, bin40, bin180


config = tf.ConfigProto(log_device_placement=True)

learning_rate = 0.001
margin = 0.01
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 63, 63, 3])
    x_image = x

# Conv Layer 1 + ReLU + maxpooling

with tf.name_scope("w_conv1"):
    w_conv1 = weight_init([8, 8, 3, 16])  # 用了16个filter
with tf.name_scope("b_conv1"):
    b_conv1 = bias_init([16])
with tf.name_scope("h_conv1"):
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)  # conv1输出张量尺寸:56*56*16
# print(h_conv1)
h_pool1 = max_pool(h_conv1)  # 池化后张量尺寸:28*28*16
# print(h_pool1)

# Conv Layer 2 + ReLU + maxpooling
with tf.name_scope("w_conv2"):
    w_conv2 = weight_init([5, 5, 16, 7])  # 用了7个filter
with tf.name_scope("b_conv2"):
    b_conv2 = bias_init([7])
with tf.name_scope("h_conv2"):
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)  # conv2输出张量尺寸:24*24*7
h_pool2 = max_pool(h_conv2)  # 池化后张量尺寸：12*12*7
# print(h_pool2)

# FC Layer 1,256个神经元
# h_pool2是一个12*12*7的tensor，转换成一个一维向量

# h_pool2_flat展平的向量，维数12*12*7，展平了再扔到全连接层里面
with tf.name_scope("h_pool_flat"):
    h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 7])
# print(h_pool2_flat)
with tf.name_scope("w_fc1"):
    w_fc1 = weight_init([12 * 12 * 7, 256])
with tf.name_scope("b_fc1"):
    b_fc1 = bias_init([256])
with tf.name_scope("h_fc1"):
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# FC Layer 2,16个神经元
with tf.name_scope("w_fc2"):
    w_fc2 = weight_init([256, 16])
with tf.name_scope("b_fc2"):
    b_fc2 = bias_init([16])
with tf.name_scope("h_fc2"):
    h_fc2 = tf.matmul(h_fc1, w_fc2) + b_fc2

with tf.name_scope("loss"):
    loss = compute_loss(h_fc2, margin)
    tf.summary.scalar("loss", loss)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# Session
# 初始化全部参数

with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)

    sess.run(tf.initialize_all_variables())
    loss_bank = []
    # 跑10个epoch，一个epoch是所有训练数据全用一次
    Sdb, Pdb = generate_Sdb()
    Stest, Ptest = generate_Stest()

    for i in range(10000):
        # print("Batch generating...")
        batch = batch_generator(32, Ptrain, Pdb, Strain, Sdb)
        # print("Batch generating done.")
        data = batch  # 把数据传进来
        _, myloss, mrg = sess.run([optimizer, loss, merged], feed_dict={x: data})
        if i % 10 == 0:
            loss_bank.append(myloss)
            print("The loss at epoch", i, "is", myloss)
            writer.add_summary(mrg, i)
        if i % 1000 == 0:
            path = "/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/model/" + str(i / 1000)
            if ~os.path.exists(path):
                os.mkdir(path)
            saver.save(sess, "/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/model/" + str(i / 1000))
    print("Training done.")

    """
    database = Sdb
    data_test = Stest

    database = np.concatenate(database)
    data_test = np.concatenate(data_test)

    Sdb_descriptor = sess.run(h_fc2, feed_dict={x: database})
    Stest_descriptor = sess.run(h_fc2, feed_dict={x: data_test})

    angle = nearest_neighbor(Stest_descriptor, Sdb_descriptor)
    #bin10, bin20, bin40, bin180 = bins_assignment(angle)
    print("The total number of matched test image at epoch", i, "is", len(angle))
    """

    #saver.save(sess, "/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/ex3")

