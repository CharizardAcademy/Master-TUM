# -*-coding: UTF-8

import sys
sys.path.insert(0, "/Users/gaoyingqiang/Desktop/大学/Master/Guided-Research/nlp-Yingqiang/data_utils")
import json
import codecs
import math
import numpy as np 
import utils
#from data_utils.utils import load_data, load_embedding, binary_mask_generator, binary_mask_generator, mask_generator
import tensorflow as tf 

# this model focus on aspect- based sentiment analysis. For aspect term (i.e. the attributes of entities show up in sentence) sentiment analysis, predict the position and the polarity of aspect term. For aspect category (i.e. the classification of aspect attributes, which may not show up in sentence) sentiment analysis , predict the category and polarity.

# input (feature engineering): GloVe embeddings, universal sentence embeddings, binary mask
# output: position of aspect, polarity of aspect

import json
import codecs
import math
import numpy as np 
import utils
import tensorflow as tf

################################################################################

# initialize model parameter

batch_size = 128
decay = 0.85
max_epoch = 5
max_max_epoch = 10
vocab_size = 5000 # 数据中不同的词的个数，处理数据的时候可以用训练Glove的那个函数数一下
embedding_size = 50
num_sentiment_label = 3 # postive, negative, neutral
fc_hidden_size = 256
lstm_hidden_size = 256 #lstm中前馈网络隐含层节点数
TRAINING_ITERATIONS = 15000
num_lstm_layer = 1 # LSTM cell的纵向堆叠厚度,图上只有一层
max_grad_norm = 5.0 # 最大梯度，超过此值的梯度将被裁剪
LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.0005
negative_weight = 2.0
positive_weight = 1.0
neutral_weight = 3.0
label_dict = {'positive':1, 'neutral':0, 'negative':2}

data_dir = '/home/gaoyingqiang/Desktop/nlp-Yingqiang/nlp-Yingqiang/convert'
embedding_dir = '/home/gaoyingqiang/Desktop/nlp-Yingqiang/nlp-Yingqiang/glove/glove.6B/glove.6B.50d.txt'

################################################################################

# functions for constructing model

def set_flag():
    flag_domain = 'Restaurant' # Organic, Restaurant etc. use flag_train_or_test=='test' for Organic dataset since Restaurant test data and organic data has the same format
    flag_train = True
    flag_test = False
    flag_uni_sent_embedding = True
    flag_aspect = 'term'# term or category
    return flag_domain, flag_train, flag_test, flag_uni_sent_embedding, flag_aspect

def fcin_weight_init():
    initial_weight = tf.Variable(tf.truncated_normal([embedding_size, fc_hidden_size], stddev=1.0/math.sqrt(embedding_size)))
    return initial_weight

def fcin_bias_init():
    initial_bias = tf.Variable(tf.zeros(fc_hidden_size))
    return initial_bias

def fcout_weight_init():
    initial_weight = tf.Variable(tf.truncated_normal([lstm_hidden_size, num_sentiment_label], stddev=1.0/math.sqrt(2*lstm_hidden_size)))
    return initial_weight

def fcout_bias_init():
    initial_bias = tf.Variable(tf.zeros(num_sentiment_label))
    return initial_bias

# number of lstm cells == max_sent_len
def bi_lstm():
    # one layer of lstm cells
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size, forget_bias=1.0)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size, forget_bias=1.0)
    lstm_fw_multicell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell]*num_lstm_layer)
    lstm_bw_multicell = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell]*num_lstm_layer)
    return lstm_fw_multicell, lstm_bw_multicell

def softmax_classifier(input, label, outputsize=3):
    weights1 = tf.Variable(tf.random_normal([len(input), 2*len(input)]))
    bias1 = tf.Variable(tf.zeros([1, 2*len(input)])+0.1)
    w1x_plus_b1 = tf.matmul(input, weights1) + bias1
    hidden_out = tf.nn.relu(w1x_plus_b1)
    weights2 = tf.Variable(tf.random_normal([2*len(input), 3]))
    bias2 = tf.Variable(tf.zeros(1, 3)+0.1)
    w2x_plus_b2 = tf.matmul(hidden_out, weights2) + bias2
    output = tf.nn.softmax(w2x_plus_b2)
    cross_entropy = -tf.reduce_sum(label*tf.log(output))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    '''
    #训练模型
    for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
 
    #测试并输出准确率
    #tf.argmax 能给出某个tensor对象在某一维上的其数据最大值所在的索引值。由于标签向量是由0,1组成，因此最大值#1所在的索引位置就是类别标签，比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，
    #而 tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表#示匹配)。
    #y的维度可选值是[0, 1]
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
 
    #上行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。例如，#[True, False, True, True] 会变成 [1,0,1,1] ，取平均值（reduce_mean）后得到 0.75.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
 
    #将预测集输入并输出准确率
    print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    '''
    return train_step
################################################################################

# prepare data and labels
flag_domain, flag_train, flag_test, flag_uni_sent_embedding, flag_aspect = set_flag()

if(flag_domain=='Restaurant'):
    # set path for universal sentence embedding
    universal_dir = '/home/gaoyingqiang/Desktop/nlp-Yingqiang/nlp-Yingqiang/universal_sentence_encoder/SemEval16_Restaurant/'

    # compute max. sentence length
    train_max_sent_len = utils.compute_max_sent_length(data_dir + '/SemEval16_Restaurant_Train.json', flag_train_or_test='train')
    test_max_sent_len = utils.compute_max_sent_length(data_dir + '/SemEval16_Restaurant_Test.json', flag_train_or_test='test')

    # read training data
    train_data_sentence, train_data_target, train_data_category, train_data_polarity  = utils.load_data(data_dir + "/SemEval16_Restaurant_Train.json",flag_train_or_test='train')

    # read test data
    test_data_sentence, test_data_target, test_data_polarity  = utils.load_data(data_dir + "/SemEval16_Restaurant_Test.json", flag_train_or_test='test')

    # process training label, mark the target with 1
    train_label = utils.label_generator(train_data_sentence, train_data_target)
    test_label = utils.label_generator(test_data_sentence, test_data_target)

    # compute mask for training data
    train_binary_mask = utils.binary_mask_generator(data_dir + '/SemEval16_Restaurant_Train.json',flag_train_or_test='train')
    train_mask = utils.mask_generator(data_dir,flag_domain='Restaurant',flag_train_or_test='train')

    # compute mask for test data
    test_binary_mask = utils.binary_mask_generator(data_dir + '/SemEval16_Restaurant_Test.json', flag_train_or_test='test')
    test_mask = utils.mask_generator(data_dir, flag_domain='Restaurant',flag_train_or_test='test')

    num_sample_train = len(train_data_sentence)









    

    


# functions polarity prediction

# functions category prediction



 

