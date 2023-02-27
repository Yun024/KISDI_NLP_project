#!/usr/bin/env python
# coding: utf-8
import os 
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import numpy as np
from konlpy.tag import Okt
from tqdm import tqdm


# # 경로 설정 후 데이터 불러오기
os.listdir()
os.chdir("C:\\Users\\newcomer02\\NTIS_Project\\data\\Default\\NTIS\\LDA")
book_mask = np.array(Image.open("book.jpg"))  #워드클라우드를 입힐 이미지 마스크

data_name = input("분석 할 데이터의 출처 사이트 이름을 적으시오:(ntis,nkis,report)")
if data_name == "ntis":
    data = pd.read_table("LDA_HanModel_kisau_(NTIS_2021)20221021_0.5.txt",encoding="cp949",header=None)
elif data_name =="nkis":
    data = pd.read_table("LDA_HanModel_(NKIS_2021)_20221021_0.5.txt",encoding="cp949",header=None)
elif data_name =="report":
    data = pd.read_table("LDA_HanModel_(KISDI_REPORT(17-21))_20221021_0.5.txt",encoding="cp949",header=None)

#LDA_HanModel_kisau_(NTIS_2021)20221021_0.5.txt
#LDA_HanModel_(KISDI_REPORT(17-21))_20221021_0.5.txt
#LDA_HanModel_(NKIS_2021)_20221021_0.5.txt

# # 워드클라우드 진행하기위한 데이터 구조 변경
### 빈 데이터프레임 생성
result = pd.DataFrame(index=range(len(data[0])),columns=["cf","df"])

### 띄어쓰기를 기준으로 문자열 분리  => min_cf역할 
for i in tqdm(range(len(data[0]))):
    result.iloc[i]["cf"] = data.iloc[i][0].split(" ")
    
### NaN을 리스트로 변환 
for i in tqdm(range(len(result["df"]))):
    result.iloc[i]["df"] = []
    
### 각 행에서 중복 제거하는 반복문 => min_df역할
for i in tqdm(range(len(result["cf"]))):
    for j in result.iloc[i]["cf"]:
        if j not in result.iloc[i]["df"]:
                result.iloc[i]["df"].append(j)
        else: continue
        
#wordlist = [n for n in tqdm(wordlist) if len(n)>1 ] # 길이가 한개인 데이터 전처리 

cf_wordlist = []
df_wordlist = []
for i in tqdm(result["cf"]):
    cf_wordlist = cf_wordlist + i 
for i in tqdm(result["df"]):
    df_wordlist = df_wordlist + i 
cf_wordcount = Counter(cf_wordlist)
df_wordcount = Counter(df_wordlist)
#cf_wordcount.most_common()
#df_wordcount.most_common()

zz = pd.concat([pd.DataFrame(cf_wordcount.most_common()),pd.DataFrame(df_wordcount.most_common())],axis=1)
zz.columns= (["term_1","cf","term_2","df"])
zz.to_csv(data_name+"_"+str(len(data))+"_wordcount.csv",encoding="cp949")
len(data)


# #### 지우고 싶은 단어 설정 
#불용어 설정 stopword를 통해 제거 가능 
wordcount.pop("연구")
wordcount.pop("분석")
wordcount.pop("서비스")
wordcount.most_common()


# # 워드클라우드 진행
wc = WordCloud(font_path="malgun",width=1000,height=1000,scale=2.0,max_font_size=100,
               background_color="white",mask=book_mask)
gen = wc.generate_from_frequencies(wordcount)
plt.figure()
plt.imshow(gen)
plt.axis("off")

wc.to_file(filename=data_name+"_wordcloud.png")