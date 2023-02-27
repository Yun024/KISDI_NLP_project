#!/usr/bin/env python
# coding: utf-8

# Package Load
import os 
import pickle
import pandas as pd
from tqdm import tqdm, tqdm_notebook
import numpy as np
from timeit import default_timer
import tomotopy as tp
import tomotopy.coherence as tpc
from datetime import datetime
from collections import Counter
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from functools import reduce
import re
from sklearn.datasets import fetch_20newsgroups
import nltk
version_name=str(datetime.today().strftime("%Y%m%d")) # 버전 정보

# pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org --upgrade nbformat

user_name = input("사용자명을 설정해주세요(영어로만 설정해주세요) : ")


# ## Option1 : 기존에 학습된 모델을 불러오기
os.chdir("C:\\Users\\newcomer02\\NTIS_Project")

# input : model 폴더명, 없으면 오류 발생 (원하는 버전 모델 있으면 지정 필요, 없으면 Enter -> Default)
# model 폴더 안에 bin파일, 없으면 오류 발생
                                     
dtm_ver=input("불러올 DTM모델 명을 입력하세요 : ")
if dtm_ver=='':
    print("\nDefault DTM모델을 불러옵니다.")
    mdl=tp.DTModel.load("./model/DTM_model_41.bin")
else:
    print("\n기존에 학습된 %s모델을 불러옵니다."%(dtm_ver))
    mdl=tp.DTModel.load("./model/"+dtm_ver+".bin")

data_ver=input("사용할 데이터 폴더명을 입력하세요(Default: 엔터키): ")
if data_ver=='':
    DATA_DIR = './data/Default/NTIS/DTM/'
else:
    DATA_DIR = './data/DTM/'+data_ver+'/'

dtm_txt=input("사용할 데이터 명을 입력하세요(Default: 엔터키): ")
#DTM_HanModel_kisau_(NTIS_2021)20221021_0.5

# input : data/LDA 폴더에 txt 형태 코퍼스 파일, 없으면 오류 발생

corpus=[]
for n, line in tqdm(enumerate(open(DATA_DIR+dtm_txt+".txt", encoding='CP949'))):
    doc=line.strip().split()
    corpus.append(doc)
print("코퍼스 파일 로드가 완료되었습니다.")

# Corpus Load Function

def data_feeder(input_file):
    for line in tqdm(open(input_file, encoding='CP949')):
        fd = line.strip().split(maxsplit=1) 
        timepoint = int(fd[0])
        if len(fd) == 1 : 
            continue 

        yield fd[1], None, {'timepoint':timepoint}

# train_type = input("Train NTIS Type : ")
# if train_type in ['ALL','all','전체','0']:
#         train_type='DTM_NTIS_ALL'
#         print("\nNTIS 전체 Data로 학습을 진행합니다.")
# elif train_type in ['ICT','ict','1']:
#         train_type='DTM_NTIS_ICT'
#         print("\nNTIS ICT Data로만 학습을 진행합니다.")
# else:
#     print("* Error * 데이터 타입을 다시 입력해주세요 ")

from string import ascii_lowercase
from string import ascii_uppercase
alphabet_list = list(ascii_lowercase) + list(ascii_uppercase) + list("ㆍ") 
len(alphabet_list)

# Corpus Load
# remove_set ={'0','1','2','3','4','5','6','7','8','9','10','할',"위한"}
porter_stemmer = nltk.PorterStemmer().stem
corpus = tp.utils.Corpus(
    tokenizer=tp.utils.SimpleTokenizer(porter_stemmer),
    stopwords= alphabet_list
)
#corpus.process(data_feeder(DATA_DIR+train_type+'.txt'))
corpus.process(data_feeder(DATA_DIR+dtm_txt+'.txt'))

# 토픽수, 연도, 학습수 사용자 입력 

num_topics=int(input("토픽 갯수 : "))
start_year = int(input("NTIS 시작 연도 : "))
num_time=int(input("NTIS 최종 연도 : ")) - start_year + 1

# ## Option2 : 새로운 모델을 학습하기
# ### 코퍼스 파일 로드

data_ver=input("사용할 데이터 폴더명을 입력하세요(Default: 엔터키): ")
if data_ver=='':
    DATA_DIR = './data/Default/NTIS/DTM/'
else:
    DATA_DIR = './data/DTM/'+data_ver+'/'

dtm_txt=input("사용할 데이터 명을 입력하세요(Default: 엔터키): ")
#DTM_HanModel_kisau_(NTIS_2021)20221021_0.5

# input : data/LDA 폴더에 txt 형태 코퍼스 파일, 없으면 오류 발생

corpus=[]
for n, line in tqdm(enumerate(open(DATA_DIR+dtm_txt+".txt", encoding='CP949'))):
    doc=line.strip().split()
    corpus.append(doc)
print("코퍼스 파일 로드가 완료되었습니다.")

# Corpus Load Function

def data_feeder(input_file):
    for line in tqdm(open(input_file, encoding='CP949')):
        fd = line.strip().split(maxsplit=1) 
        timepoint = int(fd[0])
        if len(fd) == 1 : 
            continue 

        yield fd[1], None, {'timepoint':timepoint}

# train_type = input("Train NTIS Type : ")
# if train_type in ['ALL','all','전체','0']:
#         train_type='DTM_NTIS_ALL'
#         print("\nNTIS 전체 Data로 학습을 진행합니다.")
# elif train_type in ['ICT','ict','1']:
#         train_type='DTM_NTIS_ICT'
#         print("\nNTIS ICT Data로만 학습을 진행합니다.")
# else:
#     print("* Error * 데이터 타입을 다시 입력해주세요 ")

# 토픽수, 연도, 학습수 사용자 입력 

num_topics=int(input("토픽 갯수 : "))
start_year = int(input("NTIS 시작 연도 : "))
num_time=int(input("NTIS 최종 연도 : ")) - start_year + 1
iteration=int(input("학습 횟수(10,000회 권장) : "))

print("\n %d ~ %d 년도 NTIS %s 데이터로, 토픽 %d개 DTM을 %d회 학습시킵니다."%(start_year, num_time+start_year-1,train_type[-3:],num_topics,iteration))

#import csv

# input : pkl 형태의 파일, 없으면 오류 발생 (data 폴더)
#unusing_dic = pd.read_csv(filepath_or_buffer='./data/Default/UNUSING_DIC.csv', encoding="cp949", sep=",")
#temp_f =[]
#remove_set =unusing_dic["term"].tolist()

from string import ascii_lowercase
from string import ascii_uppercase
alphabet_list = list(ascii_lowercase) + list(ascii_uppercase) + list("ㆍ") 
len(alphabet_list)

# Corpus Load
# remove_set ={'0','1','2','3','4','5','6','7','8','9','10','할',"위한"}
porter_stemmer = nltk.PorterStemmer().stem
corpus = tp.utils.Corpus(
    tokenizer=tp.utils.SimpleTokenizer(porter_stemmer),
    stopwords= alphabet_list
)
#corpus.process(data_feeder(DATA_DIR+train_type+'.txt'))
corpus.process(data_feeder(DATA_DIR+dtm_txt+'.txt'))

# input : data/DTM 폴더에 사용자별 txt 파일 Load, 없으면 오류 발생

#f = open("./data/DTM/NTIS/DTM_NTIS_ICT_DEL.txt", 'w')
#for i in range(corpus) :
#    temp = str(corpus.__getitem__(i))
#    temp = temp.strip("<tomotopy.Document with words=\"" "\">")
#    temp = str(corpus[i].timepoint) + " " + temp + "\n"
#    f.writelines(temp)
#f.close()

# Train Setting

mdl = tp.DTModel(min_df=1001,rm_top=64, k=num_topics, t=num_time, tw=tp.TermWeight.PMI, corpus=corpus)
mdl.train(0)
print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
    len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
))
print('Removed Top words: ', *mdl.removed_top_words)

# Train DTM

s = default_timer()

mdl.train(iteration)
mdl.summary()
    
e = default_timer()
print(f"{e-s} secs taken")

# 학습된 모델 저장
# output type : mode/DTM 폴더 bin 파일 (사용자별, 날짜별)

mdl.save('./model/DTM_'+str(user_name)+'_'+version_name+'_'+str(num_topics)+'.bin')

# ## Option1 or Option2 수행 후 아래부터 실행

## 토픽 및 시점별 단어분포 확인
for k in tqdm(range(mdl.k)):    
    for m in range(num_time) :
        print(str(start_year+m)+"년 == Topic #{} ==".format(k+1))
        for word, prob in mdl.get_topic_words(k,timepoint=m, top_n=20):
            print(word, prob, sep='\t')
        print()


# ## 학습된 NTIS 데이터의 연도별 토픽 확률 및 평균치 계산

topics = []
topics_d = []
for i in range(mdl.k):
    for k in range(5):
        topic = str("topic") + str(i+1)
        topic_d = str("topic") + str(i+1)
        topics.append(topic)
        topics_d.append(topic_d)

# 연도별 raw number 출력

yr = list(mdl.num_docs_by_timepoint)
yr_tp = [0]
c = 0
for i in yr:
    c+=i
    yr_tp.append(c)
print(yr_tp)

## 수정 필요
# 실행 시 다소 긴 소요 시간 발생 유의

def get_topic(time):
    temp_0=[]
    temp_1=[]
    temp_2=[]
    temp_3=[]
    temp_4=[]
    for i in range(yr_tp[5]):
        if mdl.docs[i].timepoint==0 :
            temp_0.append(sorted(mdl.docs[i].get_topics(top_n=mdl.k), key=lambda x:x[0]))
        elif mdl.docs[i].timepoint==1 :
            temp_1.append(sorted(mdl.docs[i].get_topics(top_n=mdl.k), key=lambda x:x[0]))
        elif mdl.docs[i].timepoint==2 :
            temp_2.append(sorted(mdl.docs[i].get_topics(top_n=mdl.k), key=lambda x:x[0]))
        elif mdl.docs[i].timepoint==3 :
            temp_3.append(sorted(mdl.docs[i].get_topics(top_n=mdl.k), key=lambda x:x[0]))
        elif mdl.docs[i].timepoint==4 :
            temp_4.append(sorted(mdl.docs[i].get_topics(top_n=mdl.k), key=lambda x:x[0]))
    
    temp = []
    temp.append(temp_0)
    temp.append(temp_1)
    temp.append(temp_2)
    temp.append(temp_3)
    temp.append(temp_4)
    print(len(temp_0))
    print(len(temp_1))
    print(len(temp_2))
    print(len(temp_3))
    print(len(temp_4))
    return temp

yr_topic = get_topic(5)

#yr_topic=[]
#for i in range(mdl.num_timepoints):
#    yr_topic.append(get_topic(i))

yr_topic_dist = []
for i in yr_topic:
    topic_dist = list(np.zeros(mdl.k))
    for j in i:
        for k in range(mdl.k):
            topic_dist[k] += j[k][1]
    yr_topic_dist.append(topic_dist)
    
yr_topic_norm = []
for i in yr_topic_dist:
    yr_norm = []
    for j in i:
        yr_norm.append(j/sum(i))
    yr_topic_norm.append(yr_norm)

topic_dist_norm = []
for j in range(mdl.k):
    topic_yr = []
    for i in range(mdl.num_timepoints):
        topic_yr.append(yr_topic_norm[i][j])
    topic_dist_norm.append(topic_yr)

labels=[]
labels_d=[]
for i in range(0,len(topics),5):
    labels.append(topics[i])
    labels_d.append(topics_d[i])

# NTIS/NEWS의 연도별 토픽확률 데이터프레임 생성

ntis_dict = {}
for i in range(mdl.num_timepoints):
    ntis_dict[i]=yr_topic_norm[i]

ntis_topic_dist=pd.DataFrame(ntis_dict)

# NTIS/NEWS의 평균 토픽확률 도출

s1=[]
for i in range(mdl.k):
    s1.append(sum(ntis_topic_dist.iloc[i]))
    
s1=[i/mdl.num_timepoints for i in s1]

ntis_topic_dist['avg']=s1
ntis_topic_dist['topic']=labels_d

# NTIS/NEWS의 상대적 토픽확률 증감 도출

ntis_re=[]
for i in range(mdl.k):
    ntis_re.append((ntis_topic_dist.iloc[i][mdl.num_timepoints-1]-ntis_topic_dist.iloc[i][0])/ntis_topic_dist.iloc[i][0])

ntis_topic_dist['rel']=ntis_re

ntis_sort_avg = ntis_topic_dist.sort_values("avg",ascending=False)
#ntis_topic_dist
ntis_sort_avg["cumsum"] = ntis_sort_avg["avg"].cumsum()
ntis_sort_avg.reset_index(drop=True)


# ### 분석 연도 내 평균 토픽분포

# output type : Average ICT NTIS/NEWS 연도별 html 파일 (html/DTM 폴더)

fig = make_subplots(
        rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1
    )
fig.add_bar(x=labels_d, y=ntis_topic_dist['avg'], name ="NTIS", row=1, col=1)
fig.update_layout(title=go.layout.Title(text="Average DTM Topic Distribution",
                                        font=go.layout.title.Font(size=20)),
                 yaxis_title="Prob")

fig.update_xaxes(visible=True, showticklabels=True)
fig.update_yaxes(visible=True, showticklabels=True)
fig.show()
fig.write_html('./html/DTM/Average_DTM'+str(num_topics)+'.html')
# fig.write_html('./html/DTM/Average_DTM'+str(23)+'.html')


# ## NTIS 토픽 트렌드 분석

color=[ 'aquamarine', 'black', 'blue',
            'blueviolet', 'brown', 'burlywood', 'cadetblue',
            'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
            'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
            'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
            'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
            'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
            'darkslateblue', 'darkslategray', 'darkslategrey',
            'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
            'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
            'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
            'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green',
            'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo',
            'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',
            'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
            'lightgoldenrodyellow', 'lightgray', 'lightgrey',
            'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen',
            'lightskyblue', 'lightslategray', 'lightslategrey',
            'lightsteelblue', 'lightyellow', 'lime', 'limegreen',
   #         linen, magenta, maroon, mediumaquamarine,
   #         mediumblue, mediumorchid, mediumpurple,
   ##         mediumseagreen, mediumslateblue, mediumspringgreen,
   #         mediumturquoise, mediumvioletred, midnightblue,
   #         mintcream, mistyrose, moccasin, navajowhite, navy,
   #         oldlace, olive, olivedrab, orange, orangered,
   #         orchid, palegoldenrod, palegreen, paleturquoise,
   #         palevioletred, papayawhip, peachpuff, peru, pink,
   #         plum, powderblue, purple, red, rosybrown,
   #         royalblue, rebeccapurple, saddlebrown, salmon,
   #         sandybrown, seagreen, seashell, sienna, silver,
   #         skyblue, slateblue, slategray, slategrey, snow,
    #        springgreen, steelblue, tan, teal, thistle, tomato,
    #        turquoise, violet, wheat, white, whitesmoke,
            'yellow', 'yellowgreen']


# output type : NTIS 상위 DTM 토픽 비중 html 파일 (html/DTM 폴더)

def plot_topn_topics_ntis(topn):

    total_ntis = []

    for i in range(topn):
        trace = go.Scatter(y = ntis_topic_dist.sort_values(by=['avg'],axis=0,ascending=False).iloc[i][:mdl.num_timepoints], mode = 'lines+markers',
                           hovertext='NTIS '+ ntis_topic_dist.sort_values(by=['avg'],axis=0,ascending=False)['topic'].iloc[i][5:], 
                           hoverinfo='text',
                           name = 'NTIS '+ ntis_topic_dist.sort_values(by=['avg'],axis=0,ascending=False)['topic'].iloc[i],
                           marker=dict(symbol='cross', color=color[i]),
                           line = dict(color=color[i]))
        total_ntis.append(trace)

    
    layout = go.Layout(title='NTIS DTM 상위 '+str(topn)+'개 토픽별 트렌드',legend=dict(x=0,y=-1.7),margin=dict(l=20, r=20, t=60, b=300),paper_bgcolor="White",
                       autosize=True,title_font_size=30,font=dict(size=15),hoverlabel=dict(
        
        font_size=16,
        font_family="Rockwell"
    ),
                    xaxis=dict({"tickvals":list(range(mdl.num_timepoints)),
                            "ticktext":[str(i) for i in list(range(start_year,start_year+mdl.num_timepoints))],
                           "title":"Year"}),
                    yaxis=dict({"title":"Prob"}),
                    height=2000,width=600)
    
    gen_ntis = go.Figure(data=total_ntis, layout=layout)
    pyo.iplot(gen_ntis)
    gen_ntis.write_html('./html/DTM/NTIS/상위'+str(topn)+'.html')

# 앞단에서 입력했던 토픽 개수 초과 시 에러 발생 유의

plot_topn_topics_ntis(int(input("상위 토픽 갯수 : ")))

## 대표 문서 확인
for i in tqdm(range(mdl.k)) : 
    for m in range(num_time) :
        globals()['doc_to_{}'.format(str(i)+"_"+str(m))] = 0
        globals()['doc_time_{}'.format(str(i)+"_"+str(m))] = 0
        globals()['doc_prob_{}'.format(str(i)+"_"+str(m))] = 0
        globals()['doc_index_{}'.format(str(i)+"_"+str(m))] = 0

for k in tqdm(range(len(corpus))):
    temp_doc_to = mdl.docs[k].get_topics()[0][0]
    temp_doc_time =mdl.docs[k].timepoint
    temp_doc_prob = mdl.docs[k].get_topics()[0][1]
    if globals()['doc_prob_{}'.format(str(temp_doc_to)+"_"+str(temp_doc_time))] < temp_doc_prob :
        globals()['doc_to_{}'.format(str(temp_doc_to)+"_"+str(temp_doc_time))] = temp_doc_to 
        globals()['doc_time_{}'.format(str(temp_doc_to)+"_"+str(temp_doc_time))] = temp_doc_time
        globals()['doc_prob_{}'.format(str(temp_doc_to)+"_"+str(temp_doc_time))] = temp_doc_prob
        globals()['doc_index_{}'.format(str(temp_doc_to)+"_"+str(temp_doc_time))] = k

for i in tqdm(range(mdl.k)) : 
    for m in range(num_time) :
        print("\n topic_" + str(i+1) + "_" + str(m+2016))
        print("index : " + str(globals()['doc_index_{}'.format(str(i)+"_"+str(m))]))
        print("prob : " + str(globals()['doc_prob_{}'.format(str(i)+"_"+str(m))]))
        print(mdl.docs[globals()['doc_index_{}'.format(str(i)+"_"+str(m))]])

mdl.docs[43088]


# ## pdavis 시각화
# pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org pyLDAvis
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

############################### pdavis 시각회 

for timepoint in range(mdl.num_timepoints):
    topic_term_dists = np.stack([mdl.get_topic_word_dist(k, timepoint=timepoint) for k in range(mdl.k)])
    doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs if doc.timepoint == timepoint])
    doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
    doc_lengths = np.array([len(doc.words) for doc in mdl.docs if doc.timepoint == timepoint])
    vocab = list(mdl.used_vocabs)
    term_frequency = mdl.used_vocab_freq

    prepared_data = pyLDAvis.prepare(
        topic_term_dists, 
        doc_topic_dists, 
        doc_lengths, 
        vocab, 
        term_frequency,
        start_index=0,
        sort_topics=False
    )
#    pyLDAvis.save_html(prepared_data, 'dtmvis_{}.html'.format(timepoint))

# ## 단어의 트렌드 확인

def getdata(top_n):
    topic= list(range(mdl.k))
    year=list(range(mdl.num_timepoints))
    for to_nu in tqdm(topic):
        for ye in year:
            wo=[]
            di=[]
            name_year=list(range(start_year,start_year+mdl.num_timepoints))
            word=mdl.get_topic_words(to_nu,ye,top_n)
            for i in word:
                wo.append(i[0])
                di.append(i[1])
            data_FR=pd.DataFrame(data={'word':wo,name_year[ye]:di})
            if ye==0:
                to_da=data_FR
            else:
                to_da=pd.merge(to_da,data_FR,on='word',how='outer')
        
        if to_nu==0:
            final_da=to_da
        else:
            final_da=pd.concat([final_da, to_da])
            final_da=final_da.fillna(0)
    return final_da

df = getdata(len(mdl.vocabs))

topic_index=[]
for i in range(mdl.k):
    for j in range(len(mdl.used_vocab_freq)):
        topic_index.append(i)
df.insert(1,'topic',topic_index)

var_label=[]
for i in range(mdl.k):
    for j in range(len(mdl.used_vocab_freq)):
        var_label.append(labels[i])
df.insert(1,'labels',var_label)


# ### 단어 그래프 (시각화)

import plotly
import plotly.graph_objs as go

def topic_word_plot(data,topic,top_N):
    data = df[df['topic']==topic-1]
    data = data.sort_values(by=[2020],axis=0,ascending=False)
    data = data[:top_N]
    
    total_word = []

    for i in range(top_N):
        trace = go.Scatter(y = data.iloc[i][3:], mode = 'lines+markers', hovertext=data.iloc[i,0], hoverinfo='text+y', name = data.iloc[i,0])
        total_word.append(trace)
        
    layout = go.Layout(title='',legend=dict(x=0,y=-1.7),margin=dict(l=20, r=20, t=60, b=300),paper_bgcolor="White",
                       autosize=True,title_font_size=30,font=dict(size=15),hoverlabel=dict(
        
        font_size=16,
        font_family="Rockwell"
    ),
                    xaxis=dict({"tickvals":list(range(mdl.num_timepoints)),
                            "ticktext":[str(i) for i in list(range(start_year,start_year+mdl.num_timepoints))],
                           "title":"Year"}),
                    yaxis=dict({"title":"Prob"}),
                    height=1000)
    gen_word = go.Figure(data=total_word, layout=layout)
    pyo.iplot(gen_word)

topic_word_plot(data=df,topic=1,top_N=5)

data = df[df['topic']==1]
data = data.sort_values(by=[2020],axis=0,ascending=False)
data[:10].iloc[0][3:]

# 확인하고자 하는 topic 번호 및 상위 몇 개의 단어를 보고싶은지 설정하여 시각화

def topic_word_plot2(data,topic,top_N):  

    data = data
    
    name=list(data.labels.unique())
    word_da=data[data['topic']==topic].iloc[0:,3:]
    word=data[data['topic']==topic].iloc[0:,0:1]
    
    ye=list(word_da.columns[0:-2])
    ye.append(str(int(word_da.columns[-3])+1))
    
    
    for i in range(len(word_da.columns)):
        word_da.iloc[:,i:i+1]=word_da.iloc[:,i:i+1].rank(ascending=False) 
#        word_da.iloc[:,i:i+1]=word_da.iloc[:,i:i+1].rank(ascending=True) 

    ran=list(word_da.iloc[:,-3:-2].sum(axis=1))
    word_da['word']=word
    word_da['rank']=ran

    word_da['rank']=word_da['rank'].rank(ascending=True) 
    word_da.sort_values(by=['rank'],axis=0,inplace=True)
#    word_da['rank']=word_da['rank'].rank(ascending=False) 
#    word_da['rank']=word_da['rank'].rank(ascending=False) 
    
    
    word_da=word_da.reset_index().iloc[:,1:-1]
    tesd=word_da.set_index('word').T
    
   
    col=list(tesd.columns)
    color=['brown','red','darkviolet','deeppink','forestgreen', 'fuchsia','indigo','lawngreen', 'lightslategray','yellow','silver','skyblue','tomato', 'turquoise','yellowgreen','black','chocolate','darkgoldenrod']
    
    plot=[]
    
    for N in range(top_N):
        #기존의 데이터
        plot.append(go.Scatter(y=list(tesd[col[N]]), x=ye,name=col[N],line=dict(color=color[N], width=4),marker = dict(color=color[N]),mode='lines+markers'))
        #18년 예측 데이터
        info=[]
        for z in range(len(word_da.columns)-1):
            info.append(tesd[col[N]].iloc[z])
        del info[-3]
                  
             
    layout = go.Layout(title=name[topic-1],autosize=True,
                       xaxis=dict(
                            tickvals=ye,
                           title="연도"),
                       yaxis=dict(autorange='reversed',title = "Topic내 단어 순위",linewidth=2))

    fig=go.Figure(data=plot,layout=layout)
       
    plotly.offline.iplot({
        "data": plot,
        "layout": go.Layout(autosize=True,title=name[topic-1],legend=dict(font=dict(size=20)),xaxis = dict(title = "연도",linewidth=0.5,
                            tickvals=ye
),yaxis=dict(autorange='reversed',title = "Topic내 단어 순위",linewidth=2))})
       
    plotly.offline.plot({
        "data": plot,
        "layout": go.Layout(autosize=True,title=name[topic-1],xaxis = dict(title = "연도",linewidth=0.5,
                            tickvals=ye
),yaxis=dict(autorange='reversed',title = "Topic내 단어 순위",linewidth=2))}, auto_open=False,filename="./html/DTM/topic_"+str(topic)+".html")

topic_word_plot2(data=data,topic=1,top_N=5)


# ## 문서별 토픽 결과 excel로 추출해서 확인
# 실행 시 다소 긴 소요 시간 발생 유의

top_of_doc=[]
for i in tqdm(range(len(corpus))):
    top_of_doc.append(mdl.docs[i].get_topics(top_n=3))

top_doc_df = pd.DataFrame(columns = ["top_1","top_1_prob","top_2","top_2_prob","top_3","top_3_prob"])

for i in tqdm(range(len(corpus))):
    top_doc_df= top_doc_df.append(pd.Series(np.concatenate(top_of_doc[i]).tolist(),index=top_doc_df.columns),ignore_index=True)

with open('./data/HAN/NTIS/Filtered_NTIS(16-20).pkl', 'rb') as f:
    ntis = pickle.load(f) 
    
ntis=ntis.sort_values(by=['filtered_ICT'],axis=0, ascending=True,ignore_index=True)
ntis=ntis[ntis['filtered_ICT']==1]
ntis=ntis.reset_index()

ntis_top_doc = pd.concat([ntis, top_doc_df],axis=1)

ntis_top_doc.to_excel('./data/DTM/NTIS/NTIS_TOPIC_DOC_temp.xlsx')


# # VAR/LSTM data 생성

# ## Train Set 내에서의 단어 트렌드 분석

def getdata(top_n):
    topic= list(range(mdl.k))
    year=list(range(mdl.num_timepoints))
    for to_nu in tqdm(topic):
        for ye in year:
            wo=[]
            di=[]
            name_year=list(range(start_year,start_year+mdl.num_timepoints))
            word=mdl.get_topic_words(to_nu,ye,top_n)
            for i in word:
                wo.append(i[0])
                di.append(i[1])
            data_FR=pd.DataFrame(data={'word':wo,name_year[ye]:di})
            if ye==0:
                to_da=data_FR
            else:
                to_da=pd.merge(to_da,data_FR,on='word',how='outer')
        
        if to_nu==0:
            final_da=to_da
        else:
            final_da=pd.concat([final_da, to_da])
            final_da=final_da.fillna(0)
    return final_da

df = getdata(len(mdl.vocabs))

topic_index=[]
for i in range(mdl.k):
    for j in range(len(mdl.used_vocab_freq)):
        topic_index.append(i)
df.insert(1,'topic',topic_index)

var_label=[]
for i in range(mdl.k):
    for j in range(len(mdl.used_vocab_freq)):
        var_label.append(labels[i])
df.insert(1,'labels',var_label)

df.sort_values(by=[2020],axis=0,ascending=False)

try:
    os.mkdir('./prediction/NTIS/')
except FileExistsError:
    print("폴더가 이미 존재합니다.")

# output type : 예측 모델링을 위한 pkl 파일 (data/PREDICTION 폴더)

del ntis_topic_dist['avg']
del ntis_topic_dist['rel']
ntis_topic_dist.to_pickle("./prediction/NTIS/NTIS_topic_dist_"+str("23")+".pkl")
df.to_pickle("./prediction/NTIS/WORDS_"+str("23")+".pkl")

ntis_topic_dist.to_excel("./prediction/NTIS/NTIS_topic_dist_"+str("23")+".xlsx")
#df.to_excel("./prediction/NTIS/"+"WORDS_"+str("23")+".xlsx")

# # Topic Labeling

#extractor = tp.label.PMIExtractor(min_cf=1000, min_df=500, max_len=50, max_cand=10000)
extractor = tp.label.PMIExtractor(min_df=50, max_len=2)
cands = extractor.extract(mdl)

labeler = tp.label.FoRelevance(mdl, cands, min_df=50, smoothing=1e-2, mu=0.25)
for k in tqdm(range(mdl.k)):
    print("== Topic #{} ==".format(k))
    print("Labels:", ', '.join(label for label, score in labeler.get_topic_labels(k, top_n=5)))
    for word, prob in labeler.get_topic_labels(k, top_n=10):
        print(word, prob, sep='\t')
    print()

labeler = tp.label.FoRelevance(mdl, cands, min_df=50, smoothing=1e-2, mu=0.25)
for k in tqdm(range(mdl.k)):
    print("== Topic #{} ==".format(k))
    print("Labels:", ', '.join(label for label, score in labeler.get_topic_labels(k, top_n=20)))
    for word, prob in mdl.get_topic_words(k,timepoint=0, top_n=20):
        print(word, prob, sep='\t')
    print()

labeler = tp.label.FoRelevance(mdl, cands, min_df=50, smoothing=1e-2, mu=0.25)
for k in tqdm(range(mdl.k)):
    print("== Topic #{} ==".format(k))
    print("Labels:", ', '.join(label for label, score in labeler.get_topic_labels(k, top_n=20)))
    for word, prob in mdl.get_topic_words(k,timepoint=4, top_n=20):
        print(word, prob, sep='\t')
    print()

lab=[]
for k in range(mdl.k):
    lab.append([label for label, score in labeler.get_topic_labels(k, top_n=5)])

topics = []
topics_d = []
for i in range(mdl.k):
    for k in range(5):
        topic = str("topic") + str(i+1) + str(" : ") + str(lab[i])[1:-1]
        topic_d = str("topic") + str(i+1)
        topics.append(topic)
        topics_d.append(topic_d)

ntis_topic_dist

# output type : ICT NTIS/NEWS 연도별 html 파일 (html/DTM 폴더)

def ntis_time(k):
    trace_ntis = []
    
    for i in range(k):
        trace = go.Scatter(x = np.array(range(start_year,2016+num_time)), y = ntis_topic_dist.iloc[i,:-1], mode = 'lines+markers', name = 'topic_'+str(i+1))
        trace_ntis.append(trace)
    
    layout = go.Layout(title='NTIS DTM 토픽별 트렌드',legend=dict(x=0,y=-1.7),margin=dict(l=20, r=20, t=60, b=300),paper_bgcolor="White",
                       autosize=True,title_font_size=30,font=dict(size=15),hoverlabel=dict(
                                   
        font_size=16,
        font_family="Rockwell"
    ),
                    xaxis=dict({"tickvals":list(range(mdl.num_timepoints)),
                            "ticktext":[str(i) for i in list(range(2017,2017+mdl.num_timepoints))],
                           "title":"연도"}),
                    yaxis=dict({"title":"토픽 비중"}))
    
    gen_ntis = go.Figure(data=trace_ntis, layout=layout)
    pyo.iplot(gen_ntis)
    
    #gen_ntis.write_html(DIR+'/'+"ntis_trace_"+str(num_topics)+".html")
    gen_ntis.write_html("./html/DTM/" + "ntis_trace_" + str(num_topics) + ".html")

ntis_time(23)

ntis_topic_dist['rel']=ntis_re

# output type : 토픽 비중 증감 날짜 버전별 html 파일 (html/DTM 폴더)

fig4 = make_subplots(
    rows=2, cols=1
)

fig4.append_trace(go.Bar(x=ntis_topic_dist.sort_values(by=['rel'],axis=0,ascending=False)['topic'], y=ntis_topic_dist.sort_values(by=['rel'],axis=0,ascending=False)['rel'], name ="NTIS"), row=1, col=1)
fig4.update_layout(title=go.layout.Title(text="2014 대비 "+str(2013+mdl.num_timepoints)+"년도 토픽 비중 증감 분석",
                                    font=go.layout.title.Font(size=20)))

#fig4.update_xaxes(visible=False, showticklabels=False)

fig4.show()
#fig4.write_html(DIR+'/NTIS_NEWS_CHANGE_'+str(num_topics)+'.html')

df[df.word=="인공지능"]


# ## 하나의 단어에 대한 분석
# output type : 토픽 내 단어 비중 DTM html 파일 (DTM/html 폴더)

def single_word_topic(df,word):
    topn=5
    tt=df['word']==word
    df = df[tt].sort_values(by=[2015+mdl.num_timepoints],axis=0,ascending=False)
    topics = list(df['topic'][:topn])
    prob = list(df[2015+mdl.num_timepoints][:topn])
    result=[]
    for i in range(topn):
        result.append((topics[i],prob[i]))

    total_ntis = []
    trace = go.Scatter(y = df.iloc[0][3:3+mdl.num_timepoints], mode = 'lines+markers', hovertext='NTIS '+ labels[topics[0]][5:], hoverinfo='text',name = str(round(prob[0],6))+' / NTIS '+ labels[topics[0]][5:], marker=dict(symbol='cross', size=10))
    total_ntis.append(trace)
            
    for i in range(1,topn):
        if prob[i] >=0.0001:
            trace = go.Scatter(y = df.iloc[i][3:3+mdl.num_timepoints], mode = 'lines+markers', hovertext='NTIS '+ labels[topics[i]][5:], hoverinfo='text',name = str(round(prob[i],6))+' / NTIS '+ labels[topics[i]][5:], marker=dict(symbol='cross', size=10))
            total_ntis.append(trace)
        elif prob[i] >=0.00001 and word in labels[i]:
            trace = go.Scatter(y = df.iloc[i][3:3+mdl.num_timepoints], mode = 'lines+markers', hovertext='NTIS '+ labels[topics[i]][5:], hoverinfo='text',name = str(round(prob[i],6))+' / NTIS '+ labels[topics[i]][5:], marker=dict(symbol='cross', size=10))
            total_ntis.append(trace)

    layout = go.Layout(title=word.upper(),legend=dict(x=0,y=-0.4),margin=dict(l=20, r=20, t=60, b=100),paper_bgcolor="White",
                       autosize=True,title_font_size=30,font=dict(size=15),hoverlabel=dict(
        
        font_size=16,
        font_family="Rockwell"
    ),
                    xaxis=dict({"tickvals":list(range(mdl.num_timepoints)),
                            "ticktext":[str(i) for i in list(range(2016,2016+mdl.num_timepoints))],
                           "title":"연도"}),
                    yaxis=dict({"title":"토픽 비중"}))
    
    gen_ntis = go.Figure(data=total_ntis, layout=layout)
    pyo.iplot(gen_ntis)
#    gen_ntis.write_html(DIR+'/'+word.upper()+'.html')
    return result

single_word_topic(df,input("분석 단어 : "))

# 분석하고 싶은 단어 리스트 직접 입력/수정 가능

target = [ '플랫폼',
 '미디어',
 '클라우드']

for i in target:
    single_word_topic(df,i)


# ## 동의어가 있을 시, 두 단어를 하나로 취급해 분석

def multiple_word_topic(df,word1,word2):
    topn=5
    tt1=df['word']==word1
    tt2=df['word']==word2
    a1 = df[tt1][2013+mdl.num_timepoints]
    a2 = df[tt2][2013+mdl.num_timepoints]
    sum_prob = [x+y for x,y in zip(a1,a2)]
    
    topics = sorted(range(len(sum_prob)), key=lambda i: sum_prob[i], reverse=True)[:topn]
    prob = sorted(sum_prob,reverse=True)[:topn]
    result=[]
    for i in range(topn):
        result.append((topics[i],prob[i]))
        
    total_ntis = []
    trace = go.Scatter(y = ntis_topic_dist.iloc[topics[0]][:mdl.num_timepoints], mode = 'lines+markers', hovertext='NTIS '+ labels[topics[0]][5:], hoverinfo='text',name = str(round(prob[0],6))+' / NTIS '+ labels[topics[0]][5:], marker=dict(symbol='cross', size=10))
    trace2 = go.Scatter(y = news_topic_dist.iloc[topics[0]][:mdl.num_timepoints], mode = 'lines+markers',hovertext='NEWS '+ labels[topics[0]][5:], hoverinfo='text',name = str(round(prob[0],6))+' / NEWS '+ labels[topics[0]][5:],marker=dict(symbol='square', size=10))
    total_ntis.append(trace)
    total_ntis.append(trace2)
    
    for i in range(1,topn):
        if prob[i] >=0.001:
            trace = go.Scatter(y = ntis_topic_dist.iloc[topics[i]][:mdl.num_timepoints], mode = 'lines+markers', hovertext='NTIS '+ labels[topics[i]][5:], hoverinfo='text',name = str(round(prob[i],6))+' / NTIS '+ labels[topics[i]][5:], marker=dict(symbol='cross', size=10))
            trace2 = go.Scatter(y = news_topic_dist.iloc[topics[i]][:mdl.num_timepoints], mode = 'lines+markers',hovertext='NEWS '+ labels[topics[i]][5:], hoverinfo='text',name = str(round(prob[i],6))+' / NEWS '+ labels[topics[i]][5:],marker=dict(symbol='square', size=10))
            total_ntis.append(trace)
            total_ntis.append(trace2)
        elif prob[i] >=0.0001 and word1 in labels[topics[i]]:
            trace = go.Scatter(y = ntis_topic_dist.iloc[topics[i]][:mdl.num_timepoints], mode = 'lines+markers', hovertext='NTIS '+ labels[topics[i]][5:], hoverinfo='text',name = str(round(prob[i],6))+' / NTIS '+ labels[topics[i]][5:], marker=dict(symbol='cross', size=10))
            trace2 = go.Scatter(y = news_topic_dist.iloc[topics[i]][:mdl.num_timepoints], mode = 'lines+markers',hovertext='NEWS '+ labels[topics[i]][5:], hoverinfo='text',name = str(round(prob[i],6))+' / NEWS '+ labels[topics[i]][5:],marker=dict(symbol='square', size=10))
            total_ntis.append(trace)
            total_ntis.append(trace2)
        elif prob[i] >=0.0001 and word2 in labels[topics[i]]:
            trace = go.Scatter(y = ntis_topic_dist.iloc[topics[i]][:mdl.num_timepoints], mode = 'lines+markers', hovertext='NTIS '+ labels[topics[i]][5:], hoverinfo='text',name = str(round(prob[i],6))+' / NTIS '+ labels[topics[i]][5:], marker=dict(symbol='cross', size=10))
            trace2 = go.Scatter(y = news_topic_dist.iloc[topics[i]][:mdl.num_timepoints], mode = 'lines+markers',hovertext='NEWS '+ labels[topics[i]][5:], hoverinfo='text',name = str(round(prob[i],6))+' / NEWS '+ labels[topics[i]][5:],marker=dict(symbol='square', size=10))
            total_ntis.append(trace)
            total_ntis.append(trace2)
        
    layout = go.Layout(title=word1.upper() +'/'+word2.upper(),legend=dict(x=0,y=-0.4),margin=dict(l=20, r=20, t=60, b=100),paper_bgcolor="White",
                       autosize=True,title_font_size=30,font=dict(size=15),hoverlabel=dict(
        
        font_size=16,
        font_family="Rockwell"
    ),
                    xaxis=dict({"tickvals":list(range(mdl.num_timepoints)),
                            "ticktext":[str(i) for i in list(range(2014,2014+mdl.num_timepoints))],
                           "title":"연도"}),
                    yaxis=dict({"title":"토픽 비중"}))
    
    gen_ntis = go.Figure(data=total_ntis, layout=layout)
    pyo.iplot(gen_ntis)
    gen_ntis.write_html(DIR+'/'+word1.upper()+'_'+word2.upper()+'.html')
    return result

# 동의어 처리 (2개일 때)
multiple_word_topic(df, input("분석 단어 : "),input("동의어 : "))

# 동의어 처리 (3개일 때) 
# 혹은, 2개 or 3개가 아니라, n개가 가능하다면, 사용자 입력받아서 처리