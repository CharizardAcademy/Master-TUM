# -*- coding:UTF-8

import numpy as np
import tensorflow as tf
import cv2
from task1 import *

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))


# 输入都是descriptor
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
        if angle <= 10:
            bin10 = bin10 + 1
            bin20 = bin20 + 1
            bin40 = bin40 + 1
            bin180 = bin180 + 1
        elif angle > 10 & angle <= 20:
            bin20 = bin20 + 1
            bin40 = bin40 + 1
            bin180 = bin180 + 1
        elif angle > 20 & angle <= 40:
            bin40 = bin40 + 1
            bin180 = bin180 + 1
        else:
            bin180 = bin180 + 1

    return bin10, bin20, bin40, bin180


#print_tensors_in_checkpoint_file("/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/ex3", None, False, True)

#reload model
with tf.Session() as session:

    database = Sdb
    data_test = Stest
    loss_bank = []
    new_saver = tf.train.import_meta_graph("/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/temp/ex3.meta")
    new_saver.restore(session, tf.train.latest_checkpoint("/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/temp"))

    database = np.concatenate(database)
    data_test = np.concatenate(data_test)

    graph = tf.get_default_graph()
    h_fc2 = graph.get_tensor_by_name('h_fc2/add:0')
    x = graph.get_tensor_by_name("Placeholder:0")
    Sdb_descriptor = session.run(h_fc2, feed_dict={x: database})
    Stest_descriptor = session.run(h_fc2, feed_dict={x: data_test})

    angle = nearest_neighbor(Stest_descriptor, Sdb_descriptor)
    print(angle)
