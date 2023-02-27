#!/usr/bin/env python
# coding: utf-8

import pickle, gzip
import os
from eunjeon import Mecab
mecab = Mecab(dicpath = 'C:\\mecab\\mecab-ko-dic') ## dic 반영 형태소 분석기 # 다른 형태소 분석기를 사용하게 될 경우 수정 필요
import pandas as pd
from tqdm import tqdm, tqdm_notebook
import re
import numpy as np
from collections import Counter
from gensim.test.utils import common_texts
from datetime import datetime
version_name=str(datetime.today().strftime("%Y%m%d")) # 버전 정보 변수

# ## data 및 Dictionary 불러오기
# #### 아무것도 입력하지 않고 엔터만 누를 시 디폴트 파일 Load

os.chdir("C:\\Users\\newcomer02\\NTIS_Project")
folder_name = input("분석에 사용할 폴더 이름을 입력하세요:")
dict_ver = input("\nDICTIONARY 데이터 이름을 입력하세요(Default: 엔터키) : ")
print("\n%s , ,%s 데이터를 사용합니다."%(folder_name,dict_ver))

# NKIS, NTIS, REPORT, GOV
# user-custom
# input : pkl 형태의 파일, 없으면 오류 발생 (data 폴더)
ntis = pd.read_pickle('./data/Default/'+ folder_name+ "/HAN_Filtered_" + folder_name + ".pkl")
ntis=ntis[ntis["filtered_ICT"]==1]    

from gensim.corpora.dictionary import Dictionary
# input : pkl 형태의 파일, 없으면 오류 발생 (data 폴더)
dictionary = pd.read_pickle('./data/Default/'+dict_ver+'.pkl')

dictionary["NUM"] =range(0,len(dictionary))
dictionary=dictionary[["NUM","term"]]
dictionary["NUM"]=dictionary["NUM"].apply(str)
dictionary = Dictionary(dictionary.values.tolist())

## NTIS 전처리 및 Dictionary 적용

data = ntis['contents']

def nouns(text):
    answer = list()
    target = ["NNG","NNP","NNB","NR","NP","SL"]
    for i in mecab.pos(text):
        if i[1] in target:
            answer.append(i) # i[0]
        else: 
            continue
    return answer
nouns("ict iot ai atml automl 안녕 직생하다 모르겠다 장의사")

p = re.compile(r'\W+')
pos_data = []

for i in tqdm(data):
    j = p.sub(' ',i)
    pos_data.append(nouns(j))

corpus=[]
for i in tqdm(range(len(data))):
    corpus.append(dictionary.doc2idx([word[0] for word in pos_data[i]]))

for i in tqdm(range(len(corpus))):
    corpus[i] = [word for word in corpus[i] if word != -1]

for i in tqdm(range(len(corpus))):
    corpus[i] = [dictionary[j] for j in corpus[i]]


# 길이가 너무 짧은 문서 제거
# 뉴스기사 부고란 등이 해당됨(50글자 미만)

new=[]
#too_short=[]
a=','
for i in tqdm(range(len(corpus))):
    new.append(a.join(corpus[i]).replace(',',' '))

#for i in range(len(new)):
#    if len(new[i])<50:
#        too_short.append(i)
        
#new=[i for i in new if len(i)>=50]
new

# ntis=ntis[ntis['filtered_ICT']==1]
# ntis.to_excel("ICT_NTIS.xlsx")

# 길이가 너무 짧아 제거된 문서 개수 확인

#before = len(ntis)
#deleted = [ntis['contents'][i] for i in too_short]
#ntis = ntis.drop(too_short)
#ntis = ntis.reset_index(drop=True)
#after= len(ntis)
#print("%s documents deleted\n"%(before-after))
#print(deleted)


# ## Part 1. LDA input 생성
# ### LDA를 하기 위한 input 형태로 변경해주는 작업

# try:
#     os.mkdir('./data/LDA/NTIS/'+user_name+'_'+version_name)
# except FileExistsError:
#     print("폴더가 이미 존재합니다.")

# output type : data/LDA 폴더에 npy 파일 저장

#np.save('./data/Default/NTIS/LDA/ict_index',[list(ntis['filtered_ICT']).index(1),len(ntis)])
#np.save('./data/LDA/KISAU_20210621/ict_index',[list(ntis['filtered_ICT']).index(0),len(ntis)])
# zz = np.load('./data/Default/NTIS/LDA/ict_index.npy')

# input : data/LDA 폴더에 있는 txt 파일 Load, 없으면 오류 발생 

f = open("./data/Default/" + folder_name + "/LDA_PRED_910_"+ folder_name +".txt", 'w')
for i in tqdm(range(len(new))):
    f.writelines(new[i]+'\n')
f.close()


# ## Part 2. DTM input 생성
# ### DTM을 하기 위한 input 형태로 변경해주는 작업

# ### (1) NTIS 문서 전체를 DTM에 학습
# NTIS 전체를 DTM에 학습시키려면 해당 셀을 실행

year=list(ntis['year'])
year = [int(i) for i in year]
for i in tqdm(range(len(new))):
    new[i] = str(year[i]-2016)+' '+new[i]

# try:
#     os.mkdir('./data/DTM/NTIS/'+user_name+'_'+version_name)
# except FileExistsError:
#     print("폴더가 이미 존재합니다.")

# # NTIS 전체 
# # input : data/DTM 폴더에 사용자별 txt 파일 Load, 없으면 오류 발생

# f = open("./data/DTM/NTIS/"+user_name+'_'+version_name+"/DTM_NTIS_ALL.txt", 'w')
# for i in tqdm(range(len(new))):
#     f.writelines(new[i]+'\n')
# f.close()

# ### (2) ICT_NTIS문서만을 DTM에 학습

new=new[len(corpus)-len(ntis):len(corpus)]

# input : data/DTM 폴더에 사용자별 txt 파일 Load, 없으면 오류 발생

f = open("./data/Default/NTIS/DTM/"+"DTM"+"_"+ntis_ver+".txt", 'w')
for i in tqdm(range(len(new))):
    f.writelines(new[i]+'\n')
f.close()