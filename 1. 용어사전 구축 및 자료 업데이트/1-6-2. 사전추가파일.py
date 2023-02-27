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
# - data폴더의 delete.csv 파일

# # Part 2. 수정된 단어사전 저장
# ## 아래 코드를 실행해야 사용자사전에 최종 반영됨

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
# 
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