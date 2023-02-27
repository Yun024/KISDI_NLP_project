#!/usr/bin/env python
# coding: utf-8

# # 단어사전 변경을 원하는 경우 실행
# Package Load
import re
import pickle
import numpy as np
import copy
import pandas as pd
from tqdm import tqdm
from eunjeon import Mecab
from datetime import datetime
from timeit import default_timer
from gensim import corpora
from gensim.corpora import Dictionary
version_name=str(datetime.today().strftime("%Y%m%d")) # 버전 정보(오늘의 날짜) 담는 변수


# ### 사용자명, 불러올 DICTIONARY 데이터명 입력 (사용자별 폴더 생성)
user_name = input("사용자명을 설정해주세요 : ")

data_ver = input("\n불러오실 DICTIONARY 데이터명을 입력하세요 (Default : 엔터키): ")
# 기존 Dictionary Load
# input : pkl 형태의 파일, 없으면 오류 발생
if data_ver =='':
    with open('./data/Default/DICTIONARY.pkl', 'rb') as f:
        dictionary = pickle.load(f)
else:
    with open('./data/Default/'+data_ver+'.pkl', 'rb') as f:
        dictionary = pickle.load(f) 
dictionary = Dictionary([dictionary[0].tolist()])
origin_dictionary =  copy.deepcopy(dictionary)

# # Part 1. 단어사전 추가/제거
# ## Option 1 : 단어사전 추가
# ### 1-1 : 엑셀 파일로 단어 일괄 추가
# - data폴더의 add.csv 파일
# 아래 3번 단계로 가서 수정된 단어 저장 해야만 dictionary에 저장됨
# input : data폴더의 add.csv 파일, 없으면 오류 발생
data=pd.read_csv("./data/Update/add.csv",  names=['word'], header=None)
data=data.dropna()
new_words=list(data.iloc[1:,0])
dictionary.add_documents([new_words])
new_dictionary = dictionary
print("\n")
print("단어사전에 단어가", len(new_words), "개 추가되었습니다.")
print("추가된 단어 : ", new_words)
print("\n")
print("적용 전 단어 사전 개수 : ", len(origin_dictionary))
print("적용 후 단어 사전 개수 : ", len(new_dictionary))

#pd.DataFrame(new_dictionary.values()).to_csv('./data/Default/USING_DIC(696393).csv', encoding='CP949')
#pd.DataFrame(new_dictionary.values()).to_pickle('./data/Default/USING_DIC(696393).pkl')

# ### 1-2 : 직접 입력해서 단어 하나씩 추가
# 아래 3번 단계로 가서 수정된 단어 저장 해야만 dictionary에 저장됨
print("단어사전에 추가하고 싶은 새로운 단어를 입력하시오.")
print("입력을 그만두고 싶으면 'yes' 를 입력하시오. (y or Y도 가능!)")
break_list = ['yes', 'YES', 'Yes', 'y', 'Y'] # 입력 그만둘 때 입력하는 명령어, 직접 변경 가능
new_words = []
while True :
    word = input("추가하고 싶은 단어를 입력하시오. : ") 
    if word == '' :
        print("단어가 입력되지 않았습니다. 단어를 다시 입력하세요.")
    else :
        if word in break_list :
            break
        new_words.append(word)
dictionary.add_documents([new_words])
new_dictionary = dictionary
print("\n")
print("단어사전에 단어가", len(new_words), "개 추가되었습니다.")
print("추가된 단어 : ", new_words)
print("\n")
print("적용 전 단어 사전 개수 : ", len(origin_dictionary))
print("적용 후 단어 사전 개수 : ", len(new_dictionary))

# ## Option 2 : 단어사전 제거
# ### 2-1 : 엑셀파일로 단어 일괄 제거
# - data폴더의 delete.csv 파일
# 아래 3번 단계로 가서 수정된 단어 저장 해야만 dictionary에 저장됨
# input : data폴더의 delete.csv 파일, 없으면 오류 발생
data=pd.read_csv("./data/Update/delete.csv",  names=['word'], header=None)
del_list=list(data.iloc[:,0])for cnt in range(len(del_list)):  
    dictionary.filter_tokens(bad_ids=[dictionary.token2id[del_list[cnt]]])
new_dictionary = dictionary
print("\n")
print("단어사전에 단어가", len(del_list), "개 제거되었습니다.")
print("제거된 단어 : ", del_list)

# ### 2-2 : 직접 입력받은 단어 하나씩 제거
# 아래 3번 단계로 가서 수정된 단어 저장 해야만 dictionary에 저장됨
print("단어사전에서 제거하고 싶은 단어를 입력하시오.")
print("입력을 그만두고 싶으면 'yes' 를 입력하시오. (y or Y도 가능!)")
break_list = ['yes', 'YES', 'Yes', 'y', 'Y'] # 입력 그만둘 때 입력하는 명령어, 직접 변경 가능
del_list = []
while True :
    del_word = input("제거하고 싶은 단어를 입력하시오. : ") 

    if del_word in break_list :
        break

    else :
        if del_word == '' :
            print("단어가 입력되지 않았습니다. 단어를 다시 입력하세요.")
        elif del_word not in dictionary.token2id :
            print("단어사전에 존재하지 않는 단어입니다. 단어를 다시 입력하세요.")
        else :
            del_list.append(del_word)
before_dictionary = copy.deepcopy(dictionary)
for cnt in range(len(del_list)):  
    dictionary.filter_tokens(bad_ids=[dictionary.token2id[del_list[cnt]]])
new_dictionary = dictionary
print("\n")
print("단어사전에 단어가", len(del_list), "개 제거되었습니다.")
print("제거된 단어 : ", del_list)
print("\n")
print("적용 전 단어 사전 개수 : ", len(before_dictionary))
print("적용 후 단어 사전 개수 : ", len(new_dictionary))

# ### Option1 or Option2 수행 후 아래부터 실행
# # Part 2. 수정된 단어사전 저장
# ## 아래 코드를 실행해야 사용자사전에 최종 반영됨
# 1,2번 과정 작업후 실행해야만 dictionary 수정 반영됨
version_name=str(datetime.today().strftime("%Y%m%d")) 
# input : pkl 파일, 없으면 오류 발생\\\\\\\\\\\
with open('./data/DICTIONARY_'+user_name+'_'+str(version_name)+'.pkl', 'wb') as f:
    pickle.dump(new_dictionary, f)
    
# ## 단어사전 적용과정
# ### mecab 대신 다른 형태소 분석기로 바뀌면 해당 분석기에 맞는 코드로 변경 필요
data_use=pd.read_csv("C:/mecab/user-dic/custom.csv",index_col=0)
new_words=data_use["term"].copy()
new_words
# input : csv 파일, 해당 파일 없으면 오류 발생 (mecab 설치 필요)
data=pd.read_csv("C:/mecab/user-dic/nnp.csv",index_col=0)

# 복합명사 mecab 추가 모듈
index_num = len(data)
for idn in tqdm(range(len(new_words))):
    data.loc[index_num+idn]={'<OOV>' : new_words[idn], 'NNP' : 'NNP', '*' : '*', 'F' : 'T', '<OOV>' : new_words[idn], '*.1' : '*', '*.2' : '*', '*.3' : '*', '*.4' : '*', '*.5' : '*'}
for idd in range(len(del_list)):
    del_id=data[data['과학연구']==del_list[idd]].index
    data=data.drop(index=del_id)

data=data.drop_duplicates(['<OOV>'])
data=data.reset_index()
data=data.drop(columns='index')
data

# output type : csv 형식으로 저장, 해당 파일 없으면 오류 발생 (mecab 설치 필요)
#data.to_csv("C:/mecab/user-dic/custom.csv", index=False, encoding='CP949')
data.to_csv("C:/mecab/user-dic/custom.csv", index=False)

# # 우선순위 설정
# ## window powershell 관리자 모드로 실행 (아래 순서대로 복붙가능)
# ### 1. cd C:\mecab
# ### 2. tools\add-userdic-win.ps1
# ### 3. 에러가 발생할 경우 Set-ExecutionPolicy Unrestricted 입력 후 y 입력
# ### 4. 다시올라가서 2. 내용 입력
# ## 아래 코드 실행전에 반드시 window powershell 적용!!! 아닐 경우 power shell 오류발생

# ## 오류가 발생될 시 kernel 리셋 이후 power shell 실행 이후 다음 코드 실행

from eunjeon import Mecab
## 형태소 분석기 수정 후 결과 (확인작업)
## customizing 한 mecab 파일로 작업(뒷작업은 이것으로 사용)
user_mecab=Mecab(dicpath='C:/mecab/mecab-ko-dic')
#user_mecab.morphs("객체지향프로그래밍") ## 테스트 해보고 싶은 단어 입력
#user_mecab.nouns("플랫폼")