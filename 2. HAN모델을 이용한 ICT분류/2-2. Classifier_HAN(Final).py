#!/usr/bin/env python
# coding: utf-8

# # 기존 HAN 모델을 사용할 경우 실행
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
import pandas as pd
import numpy as np
import pickle
from eunjeon import Mecab
import os
import collections
import re
from tqdm import tqdm
import nltk

# 다른 형태소 분석기를 사용하게 될 경우 수정 필요
mecab = Mecab(dicpath = 'C:\\mecab\\mecab-ko-dic') ## dic 반영 형태소 분석기  

from collections import defaultdict
from collections import Counter 
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Activation 
from keras.layers import Conv1D, MaxPooling1D, merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.callbacks import EarlyStopping
from keras.models import Model 

from keras import optimizers
from keras import backend as K
from tensorflow.keras.layers import Layer,InputSpec
from keras import initializers as initializers, regularizers, constraints

import json

from sklearn.metrics import precision_recall_curve
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from keras.models import load_model
from datetime import datetime

import timeit

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
version_name=str(datetime.today().strftime("%Y%m%d")) # 버전 정보(오늘의 날짜) 담는 변수


# # Part 2. Classification with trained HAN
# - 학습된 분류모델, HAN을 가지고 분류하고 싶은 NTIS 혹은 NEWS의 ICT 분류 및 결과 저장
os.chdir("C:\\Users\\newcomer02\\NTIS_Project")
han_ver = input("\n불러올 HAN모델 폴더명을 입력하세요 (Default : 엔터키): ")
# data 폴더에 존재
ntis_ver = input("\n불러올 데이터 파일명을 입력하세요 (Default : 엔터키): ")
# Trained_HAN_kisau_20221021_0.8773
# GOV,NKIS,NTIS,REPORT

# input : model 폴더명, 없으면 오류 발생 (원하는 버전 모델 있으면 지정 필요)
# 학습 모델 및 threshold 로드 (threshold 폴더 이름에 자동 저장)

if han_ver =='':
    model = load_model('./model/Default/HAN_0.4240')
    thr = float(0.4240)
    print('\nThreshold of the model : {}'.format(thr)) 
    
else:
    model = load_model('./model/'+han_ver)
    thr = float(han_ver[-6:])
    print('\nThreshold of the model : {}'.format(thr)) 
    
if thr >= 0.5 :
    thr = 0.5


# ## 2-1. NTIS 
# ### (1) 학습 모델 및 목적 데이터 로드
ntis["year"] =ntis["year"].replace(2022,2021)
len(ntis)
ntis["year"]
# ntis 데이터 로드
# input : pkl 파일, 없으면 오류 발생 (data 폴더)

if ntis_ver =='GOV':
    with open('./data/Default/' + ntis_ver + "/" + ntis_ver + "_2022.pkl", 'rb') as f:
        ntis = pickle.load(f)
else:
    with open('./data/Default/'+ ntis_ver + "/" + ntis_ver + "_2021.pkl",'rb') as f:
        ntis = pickle.load(f) 


# ### (2) 데이터 전처리
def nouns(text):
    answer = list()
    target = ["NNG","NNP","NNB","NR","NP","SL"]
    for i in mecab.pos(text):
        if i[1] in target:
            answer.append(i[0])
        else: 
            continue
    return answer
# nouns("ict iot ai atml automl 안녕 직생하다 모르겠다 장의사")
def preprocessing(text_list):
    global text
    for text in text_list:
        text = re.sub(r'\W+', ' ', text) 
        text = re.sub('o','', text)
        text = re.sub('○','', text)
        text = re.sub('ㅇ','', text)
        text = nouns(text)
    return text

def preprocessing_1(text_list):
    preprocessed_text = []
    for text in text_list:
        text = re.sub(r'\W+', ' ', text) 
        text = re.sub('o','', text)
        text = re.sub('○','', text)
        text = re.sub('ㅇ','', text)
        text = nouns(text)
        preprocessed_text.append(text)
    return preprocessed_text

from nltk.tokenize import LineTokenizer

line_tokenizer = LineTokenizer()

# 실행 시 다소 긴 소요 시간 발생 유의

ntis = ntis.reset_index(drop=True)
if ntis_ver =="GOV":
    ntis['content'] = ntis['contents'].apply(lambda text: [preprocessing(line_tokenizer.tokenize(i)) for i in text.split("\t")])
else :
    ntis['content'] = ntis['contents'].apply(lambda text: line_tokenizer.tokenize(text))
    ntis['content']= ntis.content.apply(preprocessing_1) 
    
# HAN 모델 학습 전 파라미터값 설정
# [문서, 문장, 단어]를 계층적으로 표현하기 위해 담을 Data Shape 구성

DIC = pd.read_csv('./data/Default/user-custom.csv')
#ICT_DIC = DIC[DIC['category']=='정보·통신']['term']
#list_ICT_DIC = list(ICT_DIC)
#list_ICT_DIC

# HAN 모델 학습 전 파라미터값 설정
# [문서, 문장, 단어]를 계층적으로 표현하기 위해 담을 Data Shape 구성

max_sent_length= 150 # 문장 내 최대 단어 수 
max_sentences = 30   # 문서 내 최대 문장 수 
max_words_dic = 25000
embedding_dim = 50 
adam = tf.optimizers.Adam(learning_rate=0.001) 

# 단어 단위 토큰화 작업
# input : json 형태의 파일, 해당 파일 없으면 오류 발생 (data/HAN 폴더)
tokenizer = Tokenizer(num_words=max_words_dic, lower=False, oov_token="term")
tokenizer.word_index = dict.fromkeys(DIC['term'])
new_dict = {}
k =1 
for v in tokenizer.word_index.keys():
    new_dict[v] = k
    k += 1

tokenizer.word_index = new_dict 

# 데이터 구성 작업 수행 : [문서, 문장, 단어]

data = np.zeros((len(ntis), max_sentences, max_sent_length), dtype='int32') 
documents = list(ntis.content)

for i, sentences in tqdm(enumerate(documents)):
    for j, sentence in enumerate(sentences): 
        if j < max_sentences:
            wordTokens = text_to_word_sequence(str(sentence))
            k = 0 
            for _, word in enumerate(wordTokens):
                try:
                    word = word.strip('\'')
                    if k < max_sent_length :
                        if word not in tokenizer.word_index:
                            continue
                        else :
                            data[i,j,k] = tokenizer.word_index[word]
                            k+=1
                    
                except SyntaxError:
                    print(word)
                    if k < max_sent_length :
                        if word not in tokenizer.word_index:
                            continue
                        else :
                            data[i,j,k] = tokenizer.word_index[word]
                            k+=1
  

# ### (3)  ICT 분류
# #### model input으로 NTIS 데이터를 넣어 ICT 분류 여부를 확인함 
# 데이터 크기에 따라 소요 시간이 커짐

s = timeit.default_timer()

Ict_prob = model.predict(data)
ntis['ICT_prob'] = Ict_prob[:,1]
ntis = ntis.sort_values(by='ICT_prob',ascending=False)
ntis['filtered_ICT'] = ntis.ICT_prob.apply(lambda x: 1 if x > thr else 0)
ntis
    
e = timeit.default_timer()
print(f"{e-s} secs taken")

# ICT / Non ICT 분류 결과 확인 (filtered_ICT 컬럼)

predicted_ntis = ntis.copy()
predicted_ntis
predicted_ntis["filtered_ICT"].value_counts()
predicted_ntis.to_excel(".\\data\\Default\\"+ ntis_ver + "/HAN_Filtered_"+ntis_ver+  ".xlsx")

# ### (4) 분류 결과 저장 및 갱신
# - Classification 결과 ICT 여부 컬럼을 함께 저장 (news의 경우 ICT 뉴스만 남기고 저장) 

# output type : data 폴더에 pkl 형식으로 변환 (사용자명, 오늘 날짜)

predicted_ntis.to_pickle('./data/Default/' + ntis_ver + "/HAN_Filtered_" + ntis_ver +'.pkl')
pred_data = pd.read_pickle("./data/Default/" + ntis_ver + "/HAN_Filtered_"+ntis_ver+".pkl")