#!/usr/bin/env python
# coding: utf-8
# # 1. 라이브러리 및 작업 디렉토리 설정
import pandas as pd
import numpy as np
import os
import pickle
os.getcwd()
os.chdir("C:\\Users\\newcomer02\\NTIS_project\\data\\Default\\NTIS\\새로운 사전 만들기")
# # 2. 데이터 불러오기
# ## NIA_DIC 불러오기
df_nia = pd.read_csv("NIADic(1130).csv")
df_nia = df_nia[((df_nia["tag"]=="nbu")|(df_nia["tag"]=="mmd")|(df_nia["tag"]=="pvg")|(df_nia["tag"]=="ncn"))]
df_nia

# ## TTA용어사전 데이터 불러오고 필요한 열 추출 및 이름 변경 
df_tta = pd.read_csv("1_TTA용어사전_KISDI.csv")
df_tta.fillna("",inplace=True)
df_tta = df_tta[["국문표제어","동의어"]]
df_tta.columns=["term","동의어"]
df_tta

# ## NTIS 데이터 불러오기
data_ver = "NTIS_2021"
with open('C:\\Users\\newcomer02\\NTIS_project\\data\\Default\\NTIS\\'+data_ver+'.pkl', 'rb') as f:
        original_ntis = pickle.load(f)

print(original_ntis.columns)
original_ntis.head()

# # 3. 데이터 전처리
# ## NTIS 키워드 데이터 전처리
keyword =original_ntis["ko_key"].str.split(",")
keyword
## 38까지 가야 NA가 없음
## 하나의 행렬에 콤마를 기준으로 나누었을때 몇개까지 데이터가 들어가 있는지 확인하는 반복문 
i = 0
while (len(keyword.str[i].dropna())) > 0 :
    i = i +1 
else :
    i = i -1 
    print(i)

## 반복문을 통해 키워드의 모든 데이터를 데이터프레임형태로 추출
df_ntis = pd.DataFrame()
for j in range(1,i) :
    kt = pd.DataFrame(keyword.str[j].dropna(axis=0))
    df_ntis = pd.concat([df_ntis,kt])
df_ntis.columns = ["term"]
df_ntis

# ## 데이터 병합 후 전처리
## 데이터 병합 
df_dic=pd.concat([df_nia,df_tta,df_ntis])

## 특수문자 제거 
df_dic["term"]=df_dic["term"].str.replace(pat=r'[^\w\s]',repl=r'',regex=True)

## 공백 제거
df_dic["term"]=df_dic["term"].str.strip()

## 중복 제거
df_dic=df_dic.drop_duplicates(["term"],keep="first")
df_dic["term"].value_counts()

## NaN 변환 후 필요없는 행 제거
df_dic.fillna("",inplace=True)
df_dic = df_dic.drop(df_dic[df_dic["term"]==""].index)
df_dic = df_dic.drop(df_dic[df_dic["term"]=="0"].index)

## term열 기준 정렬 
df_dic = df_dic.sort_values(df_dic.columns[0])

## index초기화
df_dic.reset_index(inplace=True)
df_dic.pop("index")

## 문자열길이확인하는 열 추가
dic_len=len(df_dic["term"])
df_dic["Rank"]=""
for i in range(0,dic_len):
    df_dic["Rank"][i] = len(df_dic["term"][i])
df_dic=df_dic.astype({'Rank':'int'})

# # 4. 데이터 쓰기
# df_dic.to_csv("New_Dic.csv")
df_use = df_dic[["term","Rank"]]
# df_use.to_csv("Use_Dic.csv")
df_use
# #데이터 저장 pkl형태 
# with open("Use_Dic.pkl","wb") as file:
#      pickle.dump(df_use,file)