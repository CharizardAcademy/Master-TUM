# -*- coding:UTF-8


from task1 import *
import tensorflow as tf
import pandas as pd

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

def count_false_cls(matches, obj_index):
    obj_cls = []
    for i in range((obj_index - 1)*706, obj_index * 706):
        if int(matches[i][0].trainIdx / 267) != int(matches[i][0].queryIdx / 706):
            obj_cls.append(int(matches[i][0].trainIdx / 267))
        else:
            obj_cls.append(int(matches[i][0].queryIdx / 706))
    return obj_cls

#def count_correct_cls(correct_list):

with tf.Session() as sess:
    database = Sdb
    data_test = Stest
    loss_bank = []
    new_saver = tf.train.import_meta_graph(
        "/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/model/8.0.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint(
        "/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/model"))

    database = np.concatenate(database)
    data_test = np.concatenate(data_test)
    database_q = np.concatenate(Pdb)
    data_test_q = np.concatenate(Ptest)

    #print(len(database_q),len(data_test_q))

    graph = tf.get_default_graph()
    h_fc2 = graph.get_tensor_by_name('h_fc2/add:0')
    x = graph.get_tensor_by_name("input/Placeholder:0")
    Sdb_des = sess.run(h_fc2, feed_dict={x: database})
    Stest_des = sess.run(h_fc2, feed_dict={x: data_test})

    #print(Stest_descriptor.size)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(Stest_des, Sdb_des, k=1)

    #print(len(matches))
    correct_matches = [] # 记录正确分类的match
    correct_label = [] # 记录正确分类的match中test属于哪个object

    for match in matches:
        if int(match[0].trainIdx / 267) == int(match[0].queryIdx / 706):
            correct_matches.append(match[0])
            correct_label.append(int(match[0].queryIdx / 706))
    correct_label = pd.DataFrame(correct_label)
    #print(correct_label[0].value_counts())

    # 查看每个类被错误分类成其他类的个数，填写confuse matrix
    cls_ape = count_false_cls(matches, 5)
    cls_ape = pd.DataFrame(cls_ape)
    print(cls_ape[0].value_counts())

    #print(correct_matches)
    angular_diff = []
    # angular difference还是要用quaternion算？
    for correct in correct_matches:
        qmulti = np.abs(np.inner(data_test_q[correct.queryIdx, :], database_q[correct.trainIdx, :]))
        if qmulti > 1:
            qmulti = 1
        theta = 2 * np.arccos(qmulti)
        theta = theta * 180 / np.pi
        angular_diff.append(theta)

    #print(angular_diff)
    bin10, bin20, bin40, bin180 = bins_assignment(angular_diff)
    #print(bin10, bin20, bin40, bin180)

    bins_label = [10,20,40,180]
    bins = [bin10, bin20, bin40, bin180]
    #bins = np.log(bins)
    num_bins = 4
    index = np.arange(num_bins)
    plt.bar(index, bins,alpha=0.9, width=0.35, facecolor='lightskyblue', edgecolor='white', lw=1)
    plt.xticks(index, ('<10', '<20', '<40', '<180'))
    plt.tight_layout()
    plt.show()














