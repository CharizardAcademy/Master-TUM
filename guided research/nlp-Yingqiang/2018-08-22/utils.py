# -*-coding:UTF-8
from keras.preprocessing.text import text_to_word_sequence
from nltk.tokenize import word_tokenize
import json
import pickle
import xml.etree.ElementTree as ET
import operator
import json
from openpyxl import load_workbook
from collections import defaultdict
import numpy as np 
import re
import codecs
import os
import tensorflow as tf


def sentence_tokenizer(filepath, flag_train_or_test):
    with open(filepath) as f:
        data = json.load(f)
    data_tokenized = []
    all_punctuation = "!@#$%^&*()_-+=,./ <>"
    if(flag_train_or_test == 'train'):
        for i in range(0,len(data)):
            data_dic = {}
            # remove punctuations in the sentence
            for token in data[i]["sentence"]:
                if token in all_punctuation:
                    data[i]["sentence"] = data[i]["sentence"].replace(token, ' ')
            data_dic["sentence"] = word_tokenize(data[i]["sentence"])
            # remove punctuations in the target
            for token in data[i]["target"]:
                if token in all_punctuation:
                    data[i]["target"] = data[i]["target"].replace(token, ' ')
            data_dic["target"] = word_tokenize(data[i]["target"]) 
            data_dic["category"] = data[i]["category"]
            data_dic["polarity"] = data[i]["polarity"]
            data_tokenized.append(data_dic)
        
        return data_tokenized

    elif(flag_train_or_test =='test'):
        for i in range(0,len(data)):
            
            data_dic = {}
            # remove punctuations in the sentence
            try:
                for token in data[i]["sentence"]:
                    if token in all_punctuation:
                        data[i]["sentence"] = data[i]["sentence"].replace(token, ' ')
                data_dic["sentence"] = word_tokenize(data[i]["sentence"])
                # remove punctuations in the target
                try:
                    for token in data[i]["target"]:
                        if token in all_punctuation:
                            data[i]["target"] = data[i]["target"].replace(token, ' ')
                    data_dic["target"] = word_tokenize(data[i]["target"]) 
                except KeyError:
                    for token in data[i]["category"]:
                        if token in all_punctuation:
                            data[i]["category"] = data[i]["category"].replace(token, ' ')
                    data_dic["category"] = word_tokenize(data[i]["category"])

                data_dic["polarity"] = data[i]["polarity"]
                data_tokenized.append(data_dic)

            except TypeError:
                continue
                
        return data_tokenized

# this function generates a binary mask that indicates the location of aspect within the sentence, 1 indicates the aspect and 0 for all other words
# input is the json file
def binary_mask_generator(filepath,flag_train_or_test):
    # tokenize the sentence 
    data_tokenized = sentence_tokenizer(filepath,flag_train_or_test)
    # find the length of the longest sentence
    longest_sentence_len = max([len(data_tokenized[i]['sentence']) for i in range(0, len(data_tokenized))])
    #max_index = [len(data_tokenized[i]['sentence']) for i in range(0, len(data_tokenized))]
    #print(max_index.index(longest_sentence_len))

    # every row of aspect_tag is vector indicates the aspect location within the sentence
    aspect_tag = np.zeros([len(data_tokenized), longest_sentence_len])
    
    for i in range(0, len(data_tokenized)):
        aspect = data_tokenized[i]['target']
        for j in range(0, len(data_tokenized[i]['target'])):
            if(aspect[j] != 'NULL'):
                try:
                    aspect_index = data_tokenized[i]['sentence'].index(aspect[j])
                    aspect_tag[i, aspect_index] = 1
                except ValueError:
                    continue 
            else:
                continue
        if(np.sum(aspect_tag[i,:])==0):
             aspect_tag[i,:] = 0.5

    return aspect_tag


# this function generates a mask, 1 for positive words 2 for negative word and 3 for neutral word
def mask_generator(filepath, flag_domain, flag_train_or_test):
    if(flag_domain == 'Restaurant'):
        if(flag_train_or_test == 'train'):
            # tokenize the sentence 
            data_tokenized = sentence_tokenizer(filepath + '/SemEval16_Restaurant_Train.json',flag_train_or_test)
        elif(flag_train_or_test == 'test'):
            # tokenize the sentence 
            data_tokenized = sentence_tokenizer(filepath + '/SemEval16_Restaurant_Test.json',flag_train_or_test)
    elif(flag_domain == 'Organic'):
        data_tokenized = sentence_tokenizer(filepath + '/Organic_Train.json',flag_train_or_test='test')
    
    # find the length of the longest sentence
    longest_sentence_len = max([len(data_tokenized[i]['sentence']) for i in range(0, len(data_tokenized))])
    # load dictionary for positive, negative, incresmental, decremental, reverse words
    positive_words = []
    negative_words = []
    incremental_words = []
    decremental_words = []
    reverse_words = []
    with open(filepath + '/positive_words.txt',encoding = "ISO-8859-1") as f:
        for line in f.readlines():
            positive_words.append(line[:-1]) # don't copy '\n'
    f.close()

    with open(filepath + '/negative_words.txt',encoding = "ISO-8859-1") as f:
        for line in f.readlines():
            negative_words.append(line[:-1])
    f.close()

    with open(filepath + '/incremental_words.txt',encoding = "ISO-8859-1") as f:
        for line in f.readlines():
            incremental_words.append(line[:-1])
    f.close()

    with open(filepath + '/decremental_words.txt',encoding = "ISO-8859-1") as f:
        for line in f.readlines():
            decremental_words.append(line[:-1])
    f.close()

    with open(filepath + '/reverse_words.txt',encoding = "ISO-8859-1") as f:
        for line in f.readlines():
            reverse_words.append(line[:-1])
    f.close()

    mask = np.zeros([len(data_tokenized), longest_sentence_len])
    for i in range(0, len(data_tokenized)):
        sentence = data_tokenized[i]['sentence']
        for word in sentence:
            if(word in positive_words):
                mask[i, sentence.index(word)] = 1
            elif (word in negative_words):
                mask[i, sentence.index(word)] = 2
            else:
                mask[i, sentence.index(word)] = 0

    return mask

    
def load_stop_words(filepath):
    stop_words = list()
    fsw = codecs.open('../data/stop_words.txt','r','utf-8')
    for line in fsw:
        stop_words.append(line.strip())
    fsw.close()
    return stop_words


def load_sentiment_dictionary():
    positive_words = list()
    negative_words = list()
    reverse_words = list()
    increment_words = list()
    decrement_words = list()
    sent_words_dict = dict()

    fneg = open('.../data/negative_words.txt', 'r')
    fpos = open('../data/positive_words.txt', 'r')
    frev = open('../data/reverse_words.txt', 'r')
    finc = open('../data/incrental_words.txt', 'r')
    fdec = open('../data/decremental_words.txt', 'r')

    for line in fpos:
        if not line.strip() in sent_words_dict:
            sent_words_dict[line.strip()] = 0
            positive_words.append(line.strip())

    for line in fneg:
        if not line.strip() in sent_words_dict:
            sent_words_dict[line.strip()] = 1
            negative_words.append(line.strip())

    for line in frev:
        if not line.strp() in sent_words_dict:
            sent_words_dict[line.strip()] = 2
            reverse_words.append(line.strip())

    for line in finc:
        if not line.strip() in sent_words_dict:
            sent_words_dict[line.strip()] = 3
            increment_words.append(line.strip())

    for line in fdec:
        if not line.strip() in sent_words_dict:
            sent_words_dict[line.strip()] = 4
            decrement_words.append(line.strip())

    fneg.close()
    fpos.close()
    frev.close()
    fdec.close()
    finc.close()

    return positive_words, negative_words, reverse_words, increment_words, decrement_words, sent_words_dict


def train_fasttext_embedding(filename, output, vec_dim):
    vec_dim = str(vec_dim)
    os.system('cd ../fastText && ./fasttext cbow -input ../data/' + filename + ' -output ../data/' + output + ' -dim ' + vec_dim + ' -minCount 0 -epoch 2000')


def train_glove_embedding(filepath, filename, output, vec_dic):
    os.system('cd ' + filepath)
    write_input = ''
    write_output_vec = ''
    write_output_vocab = ''
    write_dim = ''
    with open(filepath+'demo_backup.sh', 'r+') as f:
        for line in f.readlines():
            if(line.find('CORPUS')==0):
                line = 'CORPUS=' + filename + '.txt' +'\n'
            write_input += line
    with open(filepath+'demo_backup.sh', 'w') as f:
        f.write(write_input)
    f.close()
    with open(filepath+'demo_backup.sh', 'r+') as f:
        for line in f.readlines():
            if(line.find('SAVE_FILE')==0):
                line = 'SAVE_FILE=' + output + '\n'
            write_output_vec += line
    with open(filepath+'demo_backup.sh', 'w') as f:
        f.write(write_output_vec)
    f.close()
    with open(filepath+'demo_backup.sh', 'r+') as f:
        for line in f.readlines():
            if(line.find('VOCAB_FILE')==0):
                line = 'VOCAB_FILE=' + output + '_vocab.txt' + '\n'
            write_output_vocab += line
    with open(filepath+'demo_backup.sh', 'w') as f:
        f.write(write_output_vocab)
    f.close()
    with open(filepath+'demo_backup.sh', 'r+') as f:
        for line in f.readlines():
            if(line.find('VECTOR_SIZE')==0):
                line = 'VECTOR_SIZE=' + str(vec_dic) + '\n'
            write_dim += line
    with open(filepath+'demo_backup.sh', 'w') as f:
        f.write(write_dim)
    f.close()

    os.system('cd ../glove/')
    open('../glove/' + output + '.txt', 'a').close()
    open('../glove/' + output + '_vocab.txt', 'a').close()
    os.system('cd ../glove/'  + ' && ./demo_backup.sh')
    

def load_embedding(filename):
    word_dic = dict()
    embedding = list()
    fvec = codecs.open(filename, 'r', 'utf-8')
    idx = 0
    for line in fvec:
        if(len(line)<50):
            continue
        else:
            component = line.strip().split(' ') # extract every real number in the embedding vector
            
            # the first element of component is the word, except for </s>
            try:
                word_dic[component[0].lower()] = idx 
            except KeyError:
                word_dic[component[0]] = idx
            word_vec = list()
            for i in range(1, len(component)):
                word_vec.append(float(component[i]))
            embedding.append(word_vec)
        idx += 1
    fvec.close()
    word_dic['<padding>'] = idx 
    embedding.append([0.]*len(embedding[0])) # this is zero embedding vector for '<padding>'
    return word_dic, embedding

# this function load tokenized sentences
def load_data(filepath, flag_train_or_test, flag_aspect):
    if(flag_train_or_test == 'train' and flag_aspect == 'train'):
        data = sentence_tokenizer(filepath, flag_train_or_test)
        data_sentence = []
        data_target = []
        data_category = []
        data_polarity = []
        for i in range(0,len(data)):
            data_sentence.append(data[i]['sentence'])
            data_target.append(data[i]['target'])
            data_category.append(data[i]['category'])
            data_polarity.append(data[i]['polarity'])    
        return data_sentence, data_target, data_category, data_polarity
    
    elif(flag_train_or_test == 'test'):
        data = sentence_tokenizer(filepath, flag_train_or_test)
        data_sentence = []
        data_target = []
        data_category = []
        data_polarity = []
        for i in range(0,len(data)):
            data_sentence.append(data[i]['sentence'])
            if(flag_aspect=='term'):
                data_target.append(data[i]['target'])
            elif(flag_aspect=='category'):
                data_category.append(data[i]['category'])
            data_polarity.append(data[i]['polarity']) 
    
        return data_sentence, data_target, data_polarity


def load_untokenized_data(filepath, flag_train_or_test):
    if(flag_train_or_test == 'train'):
        with open(filepath) as f:
            data = json.load(f)
        data_sentence = []
        data_target = []
        data_category = []
        data_polarity = []
        for i in range(0,len(data)):
            data_sentence.append(data[i]['sentence'])
            data_target.append(data[i]['target'])
            data_category.append(data[i]['category'])
            data_polarity.append(data[i]['polarity'])    
        return data_sentence, data_target, data_category, data_polarity
    
    elif(flag_train_or_test == 'test'):
        with open(filepath) as f:
            data = json.load(f)
        data_sentence = []
        data_target = []
        data_polarity = []
        for i in range(0,len(data)):
            data_sentence.append(data[i]['sentence'])
            data_target.append(data[i]['target'])
            data_polarity.append(data[i]['polarity']) 
    
        return data_sentence, data_target, data_polarity

# mark the position of target with 1
def label_generator(data_sentence, data_target):
    label = []
    for i in range(0, len(data_sentence)):
        label_temp = []
        for word in data_sentence[i]:
            if(word == data_target[i][0]):
                label_temp.append(1)
            else:
                label_temp.append(0)
        if(np.sum(label_temp)==0):
            label_temp = [0.5]*len(label_temp)
        label.append(label_temp)
    return label

def sent_represent_generator(data_sentence, embedding_vector, embedding_dict):
    sent_represent = []
    for i in range(0, len(data_sentence)):
        sentence = []
        for word_id in data_sentence[i]:
            try:
                sentence.append(embedding_vector[embedding_dict[word_id.lower()]])
            except KeyError:
                sentence.append(embedding_vector[-1])
        #sentence = np.array(sentence)
        sent_represent.append(sentence)
    #print(np.array(x_train[0]).shape)
    return sent_represent

def sent_represent_padding(data, max_sent_length, embedding_size):
    for i in range(0, len(data)):
        if(len(data[i])<max_sent_length):
            while(len(data[i])!= max_sent_length):
                data[i].append([0]*embedding_size)
    return data

def binary_mask_padding(binary_mask, max_sent_length):
    for i in range(0, len(binary_mask)):
        if(len(binary_mask[i])<max_sent_length):
            while(len(binary_mask[i])!= max_sent_length):
                binary_mask[i].append(0)
    return binary_mask

def mask_padding(mask, max_sent_length):
    for i in range(0, len(mask)):
        if(len(mask[i])<max_sent_length):
            while(len(mask[i])!= max_sent_length):
                mask[i].append(0)
    return mask
    
def label_padding(label, max_sent_length):
    for i in range(0, len(label)):
        if(len(label[i])<max_sent_length):
            while(len(label[i])!= max_sent_length):
                label[i].append(0)
    return label

def compute_max_sent_length(filepath,flag_train_or_test):
    data_tokenized = sentence_tokenizer(filepath, flag_train_or_test)
    # find the length of the longest sentence
    longest_sentence_len = max([len(data_tokenized[i]['sentence']) for i in range(0, len(data_tokenized))])
    return longest_sentence_len

# convert this to one_hot: tf.one_hot(label, 3, on_value=1.0, off_value=0.0)
# negatvie - 0 neutral - 1 positive - 2
def polarity_label_generator(data_polarity):
    label = []
    for i in range(0, len(data_polarity)):
        if(data_polarity[i]=='negative'):
            label.append([0])
        elif(data_polarity[i]=='neutral'):
            label.append([1])
        else:
            label.append([2])
            
    return label

# convert this to one_hot: tf.one_hot(label, len(label_ref), on_value=1.0, off_value=0.0, axis=-1)
def category_label_generator(data_category):
    label_ref = []
    label = []
    for i in range(0, len(data_category)):
        if(data_category[i] not in label_ref):
            label_ref.append(data_category[i])

    for i in range(0,len(data_category)):
        label.append([label_ref.index(data_category[i])])

    return label, len(label_ref)


if __name__ == '__main__':
    data_dir = '/home/gaoyingqiang/Desktop/nlp-Yingqiang/nlp-Yingqiang/convert'
    #embedding_dir = '/home/gaoyingqiang/Desktop/nlp-Yingqiang/nlp-Yingqiang/glove/glove.6B/glove.6B.50d.txt'
   
    train_data_sentence, train_data_target, train_data_category, train_data_polarity  = load_data(data_dir + "/SemEval16_Restaurant_Train.json",flag_train_or_test='train', flag_aspect='train')
    #label = polarity_label_generator(train_data_polarity)
    
    label,_ = category_label_generator(train_data_category)
    print(label)
    #x_train = sent_represent_generator(train_data_sentence, word_embedding, word_dict)