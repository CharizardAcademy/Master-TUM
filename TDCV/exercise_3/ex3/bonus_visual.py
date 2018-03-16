# coding: UTF-8

from task1 import *
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt
from tensorflow.contrib.tensorboard.plugins import projector


with tf.Session() as sess:
    #######生成descriptor并保存#######
    """
    #database = Sdb
    data_test = Stest
    loss_bank = []
    saver = tf.train.import_meta_graph(
        "/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/model/9.0.meta")
    saver.restore(sess, tf.train.latest_checkpoint(
        "/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/model"))

    #database = np.concatenate(database)
    data_test = np.concatenate(data_test)
    #database_q = np.concatenate(Pdb)
    #data_test_q = np.concatenate(Ptest)

    #print(len(database_q),len(data_test_q))

    graph = tf.get_default_graph()
    h_fc2 = graph.get_tensor_by_name('h_fc2/add:0')
    x = graph.get_tensor_by_name("input/Placeholder:0")

    Stest_des = sess.run(h_fc2, feed_dict={x: data_test})

    # 打包descriptor并保存
    np.savetxt("Stest_des.txt",Stest_des)
    pickle.dump(Stest_des, open("Stest_des.pkl", 'wb'))

    PATH = os.getcwd()
    LOG_DIR = PATH + '/embedding-logs'

    # 导入descriptor
    descriptor = np.loadtxt('Stest_des.txt')
    num_des = descriptor.shape[0]
    dim_des = descriptor.shape[1]
    print("descriptor shape:", descriptor.shape)
    print("num of descriptors:", num_des)
    print("dimension of each descriptor:", dim_des)

    # 存成tensor变量
    tensor_des = tf.Variable(descriptor, name='Stest_des')

    # 给descriptor上标签，写到metafile里面
    obj = np.ones(num_des, dtype='int64')
    obj[0:706] = 0
    obj[706:706*2] = 1
    obj[706*2:706*3] = 2
    obj[706*3:706*4] = 3
    obj[706*4:706*5] = 4

    obj_names = ['ape', 'benchvise', 'cam', 'cat', 'duck']

    metadata_file = open(os.path.join(LOG_DIR, 'metadata_obj.tsv'), 'w')
    metadata_file.write('Object\tName\n')
    k = 706  # number of samples in each class
    j = 0

    for i in range(num_des):
        c = obj_names[obj[i]]
        if i % k == 0:
            j = j + 1
        metadata_file.write('{}\t{}\n'.format(j, c))
        # metadata_file.write('%06d\t%s\n' % (j, c))
    metadata_file.close()
    """


    LOG_DIR = "/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/ex3/embedding-logs"

    # 导入descriptor
    descriptor = np.loadtxt('Stest_des.txt')
    num_des = descriptor.shape[0]
    dim_des = descriptor.shape[1]
    print("descriptor shape:", descriptor.shape)
    print("num of descriptors:", num_des)
    print("dimension of each descriptor:", dim_des)

    # 存成tensor变量
    tensor_des = tf.Variable(descriptor, name='Stest_des')

    saver = tf.train.Saver([tensor_des])

    sess.run(tensor_des.initializer)
    saver.save(sess, LOG_DIR + "/")

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = tensor_des.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata_obj.tsv')
    # Comment out if you don't want sprites
    # embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite_4_classes.png')
    # embedding.sprite.single_image_dim.extend([img_data.shape[1], img_data.shape[1]])
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)



