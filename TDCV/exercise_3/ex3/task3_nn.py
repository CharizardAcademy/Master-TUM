# -*- coding:UTF-8


from task1 import *
import tensorflow as tf

ape = np.zeros(267)
benchvise = np.ones(267)
cam = 2 * np.ones(267)
cat = 3 * np.ones(267)
duck = 4 * np.ones(267)

label = [ape, benchvise, cam, cat, duck]
label = np.concatenate(label)

"""
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
"""

with tf.Session() as sess:
    database = Sdb
    data_test = Stest
    loss_bank = []
    new_saver = tf.train.import_meta_graph(
        "/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/temp/ex3.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint(
        "/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/temp"))

    database = np.concatenate(database)
    data_test = np.concatenate(data_test)

    graph = tf.get_default_graph()
    h_fc2 = graph.get_tensor_by_name('h_fc2/add:0')
    x = graph.get_tensor_by_name("Placeholder:0")
    Sdb_descriptor = sess.run(h_fc2, feed_dict={x: database})
    Stest_descriptor = sess.run(h_fc2, feed_dict={x: data_test})

    bf = cv2.BFMatcher(cv2.NORM_L2SQR, crossCheck=True)
    matches = bf.knnMatch(Stest_descriptor, Sdb_descriptor, k=1)

    correct_matches = []
    for match in matches:
        if match.trainIdx == match.queryIdx
    print(len(matches))













