#!/usr/bin/env python
# coding: utf-8
# # 데이터 수집 및 1차 정제
# ##     1. 새롭게 추가할 데이터는 data/Update 폴더에 배치 
# ###         (1) 패키지 설치
import re
import pickle
import numpy as np
import copy
import pandas as pd
from tqdm import tqdm
from eunjeon import Mecab
mecab = Mecab(dicpath = 'C:\\mecab\\mecab-ko-dic') ## dic 반영 형태소 분석기 # 형태소 분석기 바뀔때 여기 바꿔야함
from datetime import datetime
from timeit import default_timer
from gensim import corpora
from gensim.corpora import Dictionary
version_name=str(datetime.today().strftime("%Y%m%d"))
import os
os.chdir("C:\\Users\\newcomer02\\NTIS_Project")

# ###         (2) 사용자명, 불러올 데이터명 입력 (NTIS, NKSI, KISDI_REPORT,GOV) 
data_ver = input("\n불러오실 데이터명을 입력하세요 (Default : 엔터키): ")
# NTIS, NKIS, REPORT, GOV
# 기존 data load 
# input : pkl 파일, 없으면 오류

if data_ver =='':
    with open('./data/Default/NTIS.pkl', 'rb') as f:
        original_data = pickle.load(f)
elif data_ver =="GOV":
    with open("./data/Default/"+data_ver+"/"+data_ver+"_2022.pkl","rb") as f:
        original_data = pickle.load(f)
else:
    with open('./data/Default/'+data_ver+"/"+data_ver+'_2021.pkl', 'rb') as f:
        original_data = pickle.load(f) 

# ###         (3) 신규 데이터 불러오기 및 필요한 변수(열) 추출
add_data=input("추가할 엑셀파일명을 입력하세요 : ")
data_type = input("\n추가할 파일 형식을 입력하세요 (csv / xlsx 중 선택) : ")
print("\n%s.%s 파일을 기존 데이터에 추가합니다."%(add_data,data_type))

# 새로 추가할 연도 data
if data_type =='xlsx':
    new_data=pd.read_excel('./data/Update/'+data_ver+"/"+add_data+'.xlsx')  # .xlsx / .csv  둘 다 가능하도록 추가 
else:
    try:
        new_data=pd.read_csv('./data/Update/'+data_ver+"/"+add_data+'.csv')
    except:
        new_data=pd.read_csv('./data/Update/'+data_ver+"/"+add_data+'.csv',encoding='cp949')

# 신규 데이터프레임 생성, output : dataframe
if data_ver == "NTIS":
    new_df=pd.DataFrame(list(new_data['과학기술표준분류1-대']),columns=['L_type_1'])
    new_df['M_type_1']=new_data['과학기술표준분류1-중']
    new_df['L_type_2']=new_data['과학기술표준분류2-대']
    new_df['M_type_2']=new_data['과학기술표준분류2-중']
    new_df['L_type_3']=new_data['과학기술표준분류3-대']
    new_df['M_type_3']=new_data['과학기술표준분류3-중']
    new_df['M_L_type']=new_data['중점과학기술분류-대']
    new_df['M_M_type']=new_data['중점과학기술분류-중']
    new_df['AP_1']=new_data['적용분야1']
    new_df['AP_2']=new_data['적용분야2']
    new_df['AP_3']=new_data['적용분야3']
    new_df['6T_L_type']=new_data['6T관련기술-대']
    new_df['6T_M_type']=new_data['6T관련기술-중']
    new_df['ES_type']=new_data['경제사회목적']
    new_df['RES_num']=new_data['주관과제고유번호']
    new_df['RES_name']=new_data['과제명']
    new_df['obj']=new_data['요약문_연구목표']
    new_df['con']=new_data['요약문_연구내용']
    new_df['exp']=new_data['요약문_기대효과']
    new_df['ko_key']=new_data['요약문_한글키워드']
    new_df['en_key']=new_data['요약문_영문키워드']
    new_df['year']=new_data['제출년도']
    new_df['contents']=new_data['contents']
elif data_ver =="NKIS":
    new_df=pd.DataFrame(list(new_data["보고서명"]),columns=['title'])
    new_df['en_title']=new_data['영문보고서명']
    new_df['M_author']=new_data['연구책임자']
    new_df['organization']=new_data['소속기관']
    new_df['J_author']=new_data['공동책임자']
    new_df['in_researcher']=new_data['내부연구참가자']
    new_df['ex_researcher']=new_data['외부연구참가자']
    new_df['private']=new_data['공개여부']
    new_df['year']=new_data['출판년도']
    new_df['type_rep']=new_data['보고서유형']
    new_df['type_research']=new_data['연구유형']
    new_df['type']=new_data['표준분류']
    new_df['type_L']=new_data['대분류']
    new_df['type_M']=new_data['소분류']
    new_df['type_data']=new_data['자료유형']
    new_df['contents']=new_data['국문초록']
    new_df['url']=new_data['Url주소']    
elif data_ver == "REPORT":
    new_df=pd.DataFrame(list(new_data['서지제어번호']),columns=['NUM'])
    new_df['NAME_MAG']=new_data['잡지명']
    new_df['NAME_ART']=new_data['기사명']
    new_df['type']=new_data['분류']
    new_df['M_author']=new_data['대표저자']
    new_df['J_author']=new_data['공동저자']
    new_df['keyword']=new_data['키워드']
    new_df['year']=[str(i)[:4] for i in new_data['발행일자']]
    new_df['contents']=new_data['contents']
elif data_ver == "GOV":
    new_df = pd.DataFrame(list(new_data["과제"]),columns=["title"])
    new_df["task_objectives"] = new_data["과제목표"]
    new_df["main_content"] = new_data["주요내용"]
    new_df["expected_effect"] = new_data["기대효과"]
    new_df["contents"] = (new_data["과제"]  +" "+ new_data["과제목표"] + " "+ new_data["주요내용"] + " " + new_data["기대효과"])

# ###         (4) contents에서 전문이 영어인 문장 제거
# 중복제거, 빈 값 제거
new_df = new_df.dropna(axis=0,subset=["contents"])
new_df = new_df.drop_duplicates(['contents'], keep='first', ignore_index=True)
docs = new_df['contents']
sents = []
words = []
for i in tqdm(docs):
    sents.append(i.split('.'))
for i in tqdm(sents):
    word_sent = []
    for j in i:
        word_sent.append(j.split())
    words.append(word_sent)
count = 0
kor_doc = []
for i,l in tqdm(enumerate(sents)):
    kor_sent = []
    for j in l:
        if len(re.compile('[가-힣]+').findall(j)) != 0:
            kor_sent.append(j)
            count += 1
    kor_doc.append(kor_sent)
print(count)
sent_len = 0 # 제거 전 
kor_doc_len = 0 # 제거 후 
for i in sents:
    sent_len += len(i)
for i in kor_doc:
    kor_doc_len += len(i)
kor_fil = []
for i in kor_doc:
    kor_fil.append('.'.join(i))
new_df['contents'] = kor_fil
#new_df.to_pickle("./data/Default/NTIS_2018"+".pkl")

# ###         (5) 기존 데이터와 신규 데이터 병합 및 저장
updated_data=pd.concat([original_data,new_df])
updated_data['year']=[int(i)for i in updated_data['year']]
updated_data = updated_data.sort_values(by=['year'],axis=0,ignore_index=True)

# #### ※ 별도 저장 이름 및 경로 조정 必
# output : pkl 형식
updated_data.to_pickle("./data/Default/"+data_ver+"/"+data_ver+"_2022.pkl"

len(original_data)

len(updated_data)