# -*- coding:UTF-8

# This script converts the data -- both SemEval dataset and our dataset, to the json format.
# The jason data structure is 
# [{"sentence": "but the waitstaff was so horrible to us.", "aspect": "service", "sentiment": "negative"},{...},{...},...]
# if the sentence has multiple aspects, then they should be stored separately
# which is a list of dictionaries. 

import pickle
import xml.etree.ElementTree as ET
import operator
import json
from openpyxl import load_workbook


# this function converts the SemEval16 xml dataset to json
# comment the target attribute for Laptop dataset
def xml2json_category(jsonname, filepath, flag_train_or_test):
    if(flag_train_or_test == 'train'):
        tree = ET.parse(filepath)
        root = tree.getroot()
        data = []
        data_sentence = []
        data_category = []
        data_polarity = []
        data_target = []
        for review in root.findall('Review/sentences'):
            for sentence in review:
                for text in sentence.findall('text'):
                    for opinion in sentence.findall('Opinions/Opinion'):
                        #print(opinion.attrib)
                        category = opinion.attrib['category'].split('#')
                        data_category.append(category[0].lower())
                        polarity = opinion.attrib['polarity']
                        data_polarity.append(polarity)
                        #target = opinion.attrib['target']
                        #data_target.append(target)
                        data_sentence.append(text.text.strip())

        for i in range(0,len(data_sentence)):
            data_dic = {}
            data_dic["sentence"] = data_sentence[i]
            #data_dic["target"] = data_target[i]
            data_dic["category"] = data_category[i]
            data_dic["polarity"] = data_polarity[i]
            data.append(data_dic)

        with open(jsonname+'.json', 'w') as outfile:
            json.dump(data, outfile)

    # for test data, if term exists, then use term as apect, otherwise use category as aspect
    if(flag_train_or_test == 'test'):

        tree = ET.parse(filepath)
        root = tree.getroot()

        data = []
        data_sentence = []
        data_polarity = []
        data_target = []
        data_category = []

        for sentence in root.findall('sentence'):
            if(sentence.find('aspectTerms') != None):
                for aspect_term in sentence.findall('aspectTerms/aspectTerm'):
                    term = aspect_term.attrib['term']
                    data_target.append(term)
                    polarity = aspect_term.attrib['polarity']
                    data_polarity.append(polarity)
                    for text in sentence.findall('text'):
                        data_sentence.append(text.text.strip())
            elif(sentence.find('aspectTerms') == None): 
                for aspect_term in sentence.findall('aspectTerms/aspectTerm'): 
                    data_target.append('None')
                    polarity = aspect_term.attrib['polarity']
                    data_polarity.append(polarity)
                    for text in sentence.findall('text'):
                        data_sentence.append(text.text.strip())

        for i in range(0,len(data_sentence)):
            data_dic = {}
            data_dic["sentence"] = data_sentence[i]
            data_dic["target"] = data_target[i]
            data_dic["polarity"] = data_polarity[i]
            data.append(data_dic)

        with open(jsonname+'_term.json', 'w') as outfile:
            json.dump(data, outfile)

        data = []
        data_sentence = []
        data_polarity = []
        data_target = []
        data_category = []

        for sentence in root.findall('sentence'):
            for aspect_category in sentence.findall('aspectCategories/aspectCategory'):
                if('/' in aspect_category.attrib['category']):
                    category = aspect_category.attrib['category'].split('/')
                    data_category.append(category[0].lower())   
                    polarity = aspect_category.attrib['polarity']
                    data_polarity.append(polarity)
                    for text in sentence.findall('text'):
                        data_sentence.append(text.text.strip())
                    data_category.append(category[1].lower())   
                    polarity = aspect_category.attrib['polarity']
                    data_polarity.append(polarity)
                    for text in sentence.findall('text'):
                        data_sentence.append(text.text.strip())
                else:
                    category = aspect_category.attrib['category']
                    data_category.append(category.lower())
                    polarity = aspect_term.attrib['polarity']
                    data_polarity.append(polarity)
                    for text in sentence.findall('text'):
                        data_sentence.append(text.text.strip())

        for i in range(0,len(data_sentence)):
            data_dic = {}
            data_dic["sentence"] = data_sentence[i]
            data_dic["category"] = data_category[i]
            data_dic["polarity"] = data_polarity[i]
            data.append(data_dic)

        with open(jsonname+'_category.json', 'w') as outfile:
            json.dump(data, outfile)


# this function converts the organic dataset to json
def excel2json_category(jsonname, filepath):
    data = []
    data_attributes = []
    data_entities = []
    data_sentences = []
    data_sentiments = []

    wb = load_workbook(filepath)
    sheet = wb.get_sheet_by_name("comments2")
    sentences = sheet["G"]
    entities = sheet["E"]
    attributes = sheet["F"]
    sentiments = sheet["D"]
    for i in range(0,len(sentences)):
        if(sentences[i].value == None or sentences[i].value == 'Sentence'):
            continue
        else:
            data_sentences.append(sentences[i].value)
            data_entities.append(entities[i].value)
            data_attributes.append(attributes[i].value)
            data_sentiments.append(sentiments[i].value)
    
    pattern = {'p':'positive', 'n':'negative', None:'neutral', 0:'neutral'}
    data_sentiments = [pattern[x] if x in pattern else x for x in data_sentiments]

    pattern = {'g':'general','p':'products','f':'farmers','c':'companies','cg':'general','cp':'products', 'cf':'farming', 'cc':'companies', None:'None'}
    data_entities = [pattern[x] if x in pattern else x for x in data_entities]

    pattern = {'g':'general', 'p':'price', 'll':'label', 'q':'quality', 's':'safety', 'l':'local', 'av':'availability', 't':'taste', 'h':'health', 'c':'chemicals', 'e':'environment', 'a':'animal', 'pp':'productivity', None:'None'}
    data_attributes = [pattern[x] if x in pattern else x for x in data_attributes]

    for i in range(0,len(data_sentences)):
        data_dic = {}
        data_dic["sentence"] = data_sentences[i]
        data_dic["target"] = data_attributes[i]
        data_dic["polarity"] = data_sentiments[i]
        data.append(data_dic)

    print(len(data))

    with open(jsonname+'.json', 'w') as outfile:
        json.dump(data, outfile) 
    

if __name__ == '__main__':
   xml2json_category('SemEval16_Laptop_Train','/home/gaoyingqiang/Desktop/nlp-Yingqiang/nlp-Yingqiang/data/ABSA16_Laptops_Train_SB1_v2.xml',flag_train_or_test='train')
   #excel2json_category("Organic_Train","/home/gaoyingqiang/Desktop/nlp-Yingqiang/nlp-Yingqiang/data/aspect-labeled-comments2_JB.xlsx")


    
    
    
   
    
   

