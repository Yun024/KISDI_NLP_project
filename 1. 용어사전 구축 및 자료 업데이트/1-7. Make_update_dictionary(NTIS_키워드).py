#!/usr/bin/env python
# coding: utf-8
# # NTIS 자료 KEYWORD를 단어사전에 추가
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


# ### 사용자명, 불러올 NTIS데이터명 입력 (사용자별 폴더 생성)
add_ntis=input("추가할 엑셀파일명을 입력하세요 : ")
data_type = input("\n추가할 파일 형식을 입력하세요 (csv / xlsx 중 선택) : ")
print("\n%s.%s 파일을 기존 NTIS에 추가합니다."%(add_ntis,data_type))

# 새로 추가할 연도 NTIS
if data_type =='xlsx':
    new_ntis_data=pd.read_excel('./data/Update/NTIS/'+add_ntis+'.xlsx')  # .xlsx / .csv  둘 다 가능하도록 추가 
else:
    try:
        new_ntis_data=pd.read_csv('./data/Update/NTIS/'+add_ntis+'.csv')
    except:
        new_ntis_data=pd.read_csv('./data/Update/NTIS/'+add_ntis+'.csv',encoding='cp949')
    
# 신규 데이터프레임 생성, output : dataframe
new_ntis=pd.DataFrame(list(new_ntis_data['과학기술표준분류1-대']),columns=['L_type_1'])
new_ntis['M_type_1']=new_ntis_data['과학기술표준분류1-중']
new_ntis['L_type_2']=new_ntis_data['과학기술표준분류2-대']
new_ntis['M_type_2']=new_ntis_data['과학기드술표준분류2-중']
new_ntis['L_type_3']=new_ntis_data['과학기술표준분류3-대']
new_ntis['M_type_3']=new_ntis_data['과학기술표준분류3-중']
new_ntis['RES_num']=new_ntis_data['주관과제고유번호']
new_ntis['RES_name']=new_ntis_data['과제명']
new_ntis['obj']=new_ntis_data['요약문_연구목표']
new_ntis['con']=new_ntis_data['요약문_연구내용']
new_ntis['exp']=new_ntis_data['요약문_기대효과']
new_ntis['ko_key']=new_ntis_data['요약문_한글키워드']
new_ntis['en_key']=new_ntis_data['요약문_영문키워드']
new_ntis['year']=new_ntis_data['제출년도']
new_ntis['contents']=new_ntis_data['contents']
con = (new_ntis.L_type_1=='정보/통신') | (new_ntis.L_type_2=='정보/통신') | (new_ntis.L_type_3=='정보/통신')
ict_ntis = new_ntis[con]
keyword = ict_ntis.ko_key.str.split(',|\r\n')
keyword.str[0]

temp = list(keyword.str[0])
temp.extend(list(keyword.str[1]))
temp.extend(list(keyword.str[2]))
temp.extend(list(keyword.str[3]))
temp.extend(list(keyword.str[4]))
#temp = temp.strip()

l_keyword = pd.DataFrame(set(temp))
l_keyword = l_keyword.drop([0,1])
l_keyword[0] = l_keyword[0].str.strip()

#l_keyword.to_csv('./data/Update/add.csv', encoding="CP949")
l_keyword.to_csv('./data/Update/add.csv')

#l_keyword_2016 = l_keyword

#temp = list(l_keyword_2020[0])
#temp.extend(list(l_keyword_2019[0]))
#temp.extend(list(l_keyword_2018[0]))
#temp.extend(list(l_keyword_2017[0]))
#temp.extend(list(l_keyword_2016[0]))

#l_keyword = pd.DataFrame(set(temp))
#l_keyword = l_keyword.drop([0,1])
#l_keyword[0] = l_keyword[0].str.strip()