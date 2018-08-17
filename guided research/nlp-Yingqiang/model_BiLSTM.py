# -*-coding: UTF-8
# tensorflow version 1.8.0
#---------------------------------------------------------------------------#
#       author: Yingqiang Gao                                               #
#       description: Bi-direction LSTM model for ASBA                       # 
#       input: sentences contain aspects                                    #
#       output: sentiment label for aspects                                 #
#       last update on 22/6/2018                                            #
#---------------------------------------------------------------------------#
import sys
sys.path.insert(0, "/home/gaoyingqiang/Desktop/nlp-Yingqiang/nlp-Yingqiang/model")
import json
import codecs
import math
import numpy as np 
import utils
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.metrics import f1_score


###################### model config ####################
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

# set flags
flag_domain = 'Organic' # Organic, Restaurant etc. use flag_train_or_test=='test' for Organic dataset since Restaurant test data and organic data has the same format
flag_train = True
flag_test = False
flag_uni_sent_embedding = False


data_dir = '/home/gaoyingqiang/Desktop/nlp-Yingqiang/nlp-Yingqiang/convert'
embedding_dir = '/home/gaoyingqiang/Desktop/nlp-Yingqiang/nlp-Yingqiang/glove/glove.6B/glove.6B.50d.txt'

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

elif(flag_domain=='Organic'):
    # set path for universal sentence embedding
    universal_dir = '/home/gaoyingqiang/Desktop/nlp-Yingqiang/nlp-Yingqiang/universal_sentence_encoder/'

    # read data
    data_sentence, data_target, data_polarity  = utils.load_data(data_dir + "/Organic_Train.json", flag_train_or_test='test')

    # split the data
    train_data_sentence = data_sentence[0:round(len(data_sentence)*0.6)]
    train_data_target = data_target[0:round(len(data_sentence)*0.6)]
    train_data_polarity = data_polarity[0:round(len(data_sentence)*0.6)]

    valid_data_sentence = data_sentence[round(len(data_sentence)*0.6):round(len(data_sentence)*0.6)+round(len(data_sentence)*0.2)]
    valid_data_target = data_target[round(len(data_sentence)*0.6):round(len(data_sentence)*0.6)+round(len(data_sentence)*0.2)]
    valid_data_polarity = data_polarity[round(len(data_sentence)*0.6):round(len(data_sentence)*0.6)+round(len(data_sentence)*0.2)]

    test_data_sentence = data_sentence[round(len(data_sentence)*0.6)+round(len(data_sentence)*0.2):len(data_sentence)]
    test_data_target = data_target[round(len(data_sentence)*0.6)+round(len(data_sentence)*0.2):len(data_sentence)]
    test_data_polarity = data_polarity[round(len(data_sentence)*0.6)+round(len(data_sentence)*0.2):len(data_sentence)]

    # processlabel, mark the target with 1
    train_label = utils.label_generator(train_data_sentence, train_data_target)
    valid_label = utils.label_generator(valid_data_sentence, valid_data_target)
    test_label = utils.label_generator(test_data_sentence, test_data_target)

    # compute mask for organic dataset
    organic_binary_mask = utils.binary_mask_generator(data_dir + '/Organic_Train.json',flag_train_or_test='test')
    organic_mask = utils.mask_generator(data_dir,flag_domain='Organic',flag_train_or_test='test')
    
    train_binary_mask = organic_binary_mask[0:round(len(data_sentence)*0.6)]
    valid_binary_mask = organic_binary_mask[round(len(data_sentence)*0.6):round(len(data_sentence)*0.6)+round(len(data_sentence)*0.2)]
    test_binary_mask = organic_binary_mask[round(len(data_sentence)*0.6)+round(len(data_sentence)*0.2):len(data_sentence)]

    train_mask = organic_mask[0:round(len(data_sentence)*0.6)]
    valid_mask = organic_mask[round(len(data_sentence)*0.6):round(len(data_sentence)*0.6)+round(len(data_sentence)*0.2)]
    test_mask = organic_mask[round(len(data_sentence)*0.6)+round(len(data_sentence)*0.2):len(data_sentence)]

    # compute max_sent_length
    max_sent_len = utils.compute_max_sent_length(data_dir + '/Organic_Train.json', flag_train_or_test='test')
    # print(max_sent_len)

    num_sample_train = len(train_data_sentence)
    

if(flag_uni_sent_embedding and flag_domain=='Restaurant'):
    # load universal sentence embedding
    uni_dict, uni_embedding = utils.load_embedding(universal_dir+'Train/uni_sent_embedding_compressed.txt')
    uni_embedding = uni_embedding[1:len(uni_embedding)]
elif(flag_uni_sent_embedding and flag_domain=='Organic'):
    uni_dict, uni_embedding = utils.load_embedding(universal_dir+'Test/uni_sent_embedding_compressed.txt')
    uni_embedding = uni_embedding[1:len(uni_embedding)]

    train_uni_embedding = uni_embedding[0:round(len(data_sentence)*0.6)]
    valid_uni_embedding = uni_embedding[round(len(data_sentence)*0.6):round(len(data_sentence)*0.6)+round(len(data_sentence)*0.2)]
    test_uni_embedding = uni_embedding[round(len(data_sentence)*0.6)+round(len(data_sentence)*0.2):len(data_sentence)]

# load word embeddings
word_dict, word_embedding = utils.load_embedding(embedding_dir)
#print(word_embedding[0:2])



###################################BUILD LSTM ##################################

# this function create a weight variable with appropriate initialization
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

# 一层lstm的cell个数等于最大的句子长度
def bi_lstm():
    # one layer of lstm cells
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size, forget_bias=1.0)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size, forget_bias=1.0)
    # dropout 这里Pylint抽风不用管
    #lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, ouput_keep_prob=keep_prob)
    #lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, ouput_keep_prob=keep_prob)
    # multiple layer of lstm cells
    lstm_fw_multicell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell]*num_lstm_layer)
    lstm_bw_multicell = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell]*num_lstm_layer)
    return lstm_fw_multicell, lstm_bw_multicell


###################################BUILD LSTM ##################################

# Modeling
graph = tf.Graph()
# high version of pylint may report a false positive here
with graph.as_default(), tf.device('/device:GPU:0'):
    
    if(flag_domain=='Restaurant'):
        if(flag_train):
            tf_X = tf.placeholder(tf.float32, shape=[None, train_max_sent_len, embedding_size])
            tf_X_mask = tf.placeholder(tf.float32, shape=[None, train_max_sent_len])
            tf_X_binary_mask = tf.placeholder(tf.float32, shape=[None, train_max_sent_len])
            tf_y = tf.placeholder(tf.int64, shape=[None, train_max_sent_len])
            if(flag_uni_sent_embedding):
                tf_universal_X = tf.placeholder(tf.float32, shape=[None, train_max_sent_len])
            keep_prob = tf.placeholder(tf.float32)
        if(flag_test):
            tf_X = tf.placeholder(tf.float32, shape=[None, test_max_sent_len, embedding_size])
            tf_X_mask = tf.placeholder(tf.float32, shape=[None, test_max_sent_len])
            tf_X_binary_mask = tf.placeholder(tf.float32, shape=[None, test_max_sent_len])
            tf_y = tf.placeholder(tf.int64, shape=[None, test_max_sent_len])
            if(flag_uni_sent_embedding):
                tf_universal_X = tf.placeholder(tf.float32, shape=[None, test_max_sent_len])
            keep_prob = tf.placeholder(tf.float32)
    
    elif(flag_domain=='Organic'):
        tf_X = tf.placeholder(tf.float32, shape=[None, max_sent_len, embedding_size])
        tf_X_mask = tf.placeholder(tf.float32, shape=[None, max_sent_len])
        tf_X_binary_mask = tf.placeholder(tf.float32, shape=[None, max_sent_len])
        tf_y = tf.placeholder(tf.int64, shape=[None, max_sent_len])
        if(flag_uni_sent_embedding):
            tf_universal_X = tf.placeholder(tf.float32, shape=[None, max_sent_len])
        keep_prob = tf.placeholder(tf.float32)
        

    # initialize fc input layer
    fcin_w = fcin_weight_init()
    fcin_b = fcin_bias_init()

    # initialize fc output layer
    fcout_w = fcout_weight_init()
    fcout_b = fcout_bias_init()

    # get labels
    y_labels = tf.one_hot(tf_y, num_sentiment_label, on_value=1.0, off_value=0.0, axis=-1)

    # 用于交换张量的维度，[1，0，2]相当于是把第第0位和第1为交换，交换后shape=[train_max_sent_len, None, embedding_size]
    X = tf.transpose(tf_X, [1, 0, 2])
    # reshaping to [batch_size*sentence_length, embedding_size] 
    X = tf.reshape(X, [-1, embedding_size])
    X = tf.add(tf.matmul(X, fcin_w), fcin_b)
    X = tf.nn.relu(X)
    # 从input layer出来之后分成两份，分别进lstm_fw和lstm_bw, 问题这样的话岂不是有的句子只走lstm_fw有的只走lstm_bw?
    if(flag_domain=='Restaurant'):
        if(flag_train):
            X = tf.split(axis=0, num_or_size_splits=train_max_sent_len, value=X)
        if(flag_test):
            X = tf.split(axis=0, num_or_size_splits=test_max_sent_len, value=X)
    elif(flag_domain=='Organic'):
        X = tf.split(axis=0, num_or_size_splits=max_sent_len, value=X)

    # Bi-LSTM
    lstm_fw_multicell, lstm_bw_multicell = bi_lstm() 
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_multicell, lstm_bw_multicell, X ,dtype='float32')
    # split the output for fw and bw
    output_fw, output_bw = tf.split(outputs, [lstm_hidden_size, lstm_hidden_size], 2)
    # add two output and use as the final feature for classification
    output_feature = tf.reshape(tf.add(output_fw, output_bw), [-1, lstm_hidden_size]) 
    output_feature = tf.nn.dropout(output_feature, keep_prob)
    output_feature = tf.add(tf.matmul(output_feature, fcout_w), fcout_b)
    # 为什么这里要分成train_max_sent_len这么多份，不应该是num_sentement_label这么多份吗？
    if(flag_domain=='Restaurant'):
        if(flag_train):
            output_feature = tf.split(axis=0, num_or_size_splits=train_max_sent_len, value=output_feature)
        if(flag_test):
            output_feature = tf.split(axis=0, num_or_size_splits=test_max_sent_len, value=output_feature)
    elif(flag_domain=='Organic'):
        output_feature = tf.split(axis=0, num_or_size_splits=max_sent_len, value=output_feature)

    # change back dimension to [batch_size, n_step, n_input]
    output_feature = tf.stack(output_feature)
    output_feature = tf.transpose(output_feature, [1, 0, 2])
    if(flag_uni_sent_embedding):
        output_feature = tf.add(output_feature, tf.expand_dims(tf_universal_X,2))/2
   
    output_feature = tf.multiply(output_feature, tf.expand_dims(tf_X_binary_mask, 2))
    
    # output_feature has shape [?, max_sent_length, 3]
    #print(output_feature.shape)
    cross_entropy = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=output_feature, labels=y_labels),  tf_X_mask))
    prediction = tf.argmax(tf.nn.softmax(output_feature), 2)
    correct_prediction = tf.reduce_sum(tf.multiply(tf.cast(tf.equal(prediction, tf_y), tf.float32), tf_X_binary_mask))

    TP = tf.count_nonzero(prediction*tf_y)
    TN = tf.count_nonzero((prediction - 1)*(tf_y - 1))
    FP = tf.count_nonzero(prediction*(tf_y - 1))
    FN = tf.count_nonzero((prediction - 1)*tf_y)
    
    accuracy = (TP+TN)/(TP+FN+TN+FP)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*precision * recall/(precision+recall)

    # fix tf_X_train_mask to 0, 1 vector
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, 1000, 0.65, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

    saver = tf.train.Saver()


# sess.run
with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # generate sentence representation
    x_train = utils.sent_represent_generator(train_data_sentence, word_embedding, word_dict)
    x_valid = utils.sent_represent_generator(valid_data_sentence, word_embedding, word_dict)
    x_test = utils.sent_represent_generator(test_data_sentence, word_embedding, word_dict)
       
    # modify the sentence vectors with fixed length of max_sent_length
    if(flag_domain=='Restaurant'):
        x_train = utils.sent_represent_padding(x_train, max_sent_length=train_max_sent_len, embedding_size=embedding_size)
        #x_valid = utils.sent_represent_padding(x_valid, max_sent_length=train_max_sent_len,  embedding_size=embedding_size)
        x_test = utils.sent_represent_padding(x_test, max_sent_length=test_max_sent_len, embedding_size=embedding_size)
        train_binary_mask = utils.binary_mask_padding(train_binary_mask, train_max_sent_len)
        test_binary_mask = utils.binary_mask_padding(test_binary_mask, test_max_sent_len)
        train_mask = utils.mask_padding(train_mask, train_max_sent_len)
        test_mask = utils.mask_padding(test_mask, test_max_sent_len)
        train_label = utils.label_padding(train_label, train_max_sent_len)
        test_label = utils.label_padding(test_label, test_max_sent_len)

    elif(flag_domain=='Organic'):
        x_train = utils.sent_represent_padding(x_train, max_sent_length=max_sent_len, embedding_size=embedding_size)
        x_valid = utils.sent_represent_padding(x_valid, max_sent_length=max_sent_len,  embedding_size=embedding_size)
        x_test = utils.sent_represent_padding(x_test, max_sent_length=max_sent_len, embedding_size=embedding_size)
        # modify the binary mask with fixed length of max_sent_length
        train_binary_mask = utils.binary_mask_padding(train_binary_mask, max_sent_len)
        valid_binary_mask = utils.binary_mask_padding(valid_binary_mask, max_sent_len)
        test_binary_mask = utils.binary_mask_padding(test_binary_mask, max_sent_len)
    
        # modify the mask with fixed length of max_sent_length
        train_mask = utils.mask_padding(train_mask, max_sent_len)
        valid_mask = utils.mask_padding(valid_mask, max_sent_len)
        test_mask = utils.mask_padding(test_mask, max_sent_len)

        # modify the label with fixed length of max_sent_length
        train_label = utils.label_padding(train_label, max_sent_len)
        valid_label = utils.label_padding(valid_label, max_sent_len)
        test_label = utils.label_padding(test_label, max_sent_len)
    

    if(flag_train):
        loss_list = []
        accuracy_list = []

        for it in range(TRAINING_ITERATIONS):
            if(it * batch_size % num_sample_train + batch_size < num_sample_train):
                index = it * batch_size % num_sample_train
            else:
                index = num_sample_train - batch_size

            if(flag_uni_sent_embedding):
                _, correct_prediction_train, cost_train = sess.run([optimizer, correct_prediction, cross_entropy], feed_dict={tf_X: np.asarray(x_train[index:index+batch_size]), tf_X_mask: np.asarray(train_mask[index:index+batch_size]), tf_X_binary_mask: np.asarray(train_binary_mask[index:index+batch_size]), tf_universal_X: np.asarray(uni_embedding[index:index+batch_size]),tf_y:np.asarray(train_label[index:index+batch_size]), keep_prob: 1.0 
                })
            else:
                 _, correct_prediction_train, cost_train = sess.run([optimizer, correct_prediction, cross_entropy], feed_dict={tf_X: np.asarray(x_train[index:index+batch_size]), tf_X_mask: np.asarray(train_mask[index:index+batch_size]), tf_X_binary_mask: np.asarray(train_binary_mask[index:index+batch_size]),tf_y:np.asarray(train_label[index:index+batch_size]), keep_prob: 1.0 })
                 
                 #train_accuracy = sess.run(accuracy, feed_dict={tf_X: np.asarray(x_test), tf_X_binary_mask: np.asarray(test_binary_mask), tf_X_mask: np.asarray(test_mask), tf_y: np.asarray(test_label), keep_prob:1.0})

            #print('correct prediction: ', correct_prediction_train)
            #print('training accuracy => %.3f, cost value => %.5f for step %d, learning_rate => %.5f' % (float(correct_prediction_train)/np.sum(np.asarray(train_binary_mask[index:index+batch_size])), cost_train, it, learning_rate.eval()))
            
            print('training accuracy => %.3f, cost value => %.5f for step %d, learning_rate => %.5f' % (correct_prediction_train/np.sum(np.asarray(train_binary_mask[index:index+batch_size])), cost_train, it, learning_rate.eval()))
            

            loss_list.append(cost_train)
            accuracy_list.append(float(correct_prediction_train)/np.sum(np.asarray(train_binary_mask[index:index+batch_size])))

            '''
            if(it%50==0):
                _, correct_prediction_valid, cost_valid = sess.run([optimizer, correct_prediction, cross_entropy], feed_dict={tf_X: np.asarray(x_valid), tf_X_mask: np.asarray(valid_mask), tf_X_binary_mask: np.asarray(valid_binary_mask),tf_y:np.asarray(valid_label), keep_prob: 1.0})

                print('validation accuracy => %.3f, cost value => %.5f' % (float(correct_prediction_valid)/np.sum(np.asarray(valid_binary_mask),cost_valid)))
            '''

        saver.save(sess, '../ckpt/glove50_embedding_15000.ckpt')

        plt.plot(accuracy_list)
        axes = plt.gca()
        axes.set_ylim([0,1.2])
        plt.title('batch train accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('step')
        plt.savefig('accuracy.png')
        
        plt.close()

        plt.plot(loss_list)
        plt.title('batch train loss')
        plt.ylabel('loss')
        plt.xlabel('step')
        plt.savefig('loss.png')
        plt.close()

    if(flag_test):
        saver.restore(sess, '../ckpt/glove50_embedding_15000.ckpt')

        if(flag_uni_sent_embedding):
            correct_prediction_test, cost_test = sess.run([correct_prediction, cross_entropy],feed_dict={tf_X: np.asarray(x_test), tf_X_binary_mask: np.asarray(test_binary_mask), tf_X_mask: np.asarray(test_mask),tf_universal_X: np.asarray(uni_embedding) ,tf_y: np.asarray(test_label), keep_prob:1.0})

            f1_score = sess.run(f1, feed_dict={tf_X: np.asarray(x_test), tf_X_binary_mask: np.asarray(test_binary_mask), tf_X_mask: np.asarray(test_mask), tf_universal_X: np.asarray(uni_embedding), tf_y: np.asarray(test_label), keep_prob:1.0})

        else:
            correct_prediction_test, cost_test = sess.run([correct_prediction, cross_entropy],feed_dict={tf_X: np.asarray(x_test), tf_X_binary_mask: np.asarray(test_binary_mask), tf_X_mask: np.asarray(test_mask),tf_y: np.asarray(test_label), keep_prob:1.0})

            f1_score = sess.run(f1, feed_dict={tf_X: np.asarray(x_test), tf_X_binary_mask: np.asarray(test_binary_mask), tf_X_mask: np.asarray(test_mask), tf_y: np.asarray(test_label), keep_prob:1.0})


        print('test accuracy => %.3f' %(float(correct_prediction_test)/np.sum(test_binary_mask)))

        print('F1 score => %.3f' % (f1_score))
       











