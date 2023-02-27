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


user_name = input("사용자명을 설정해주세요(영어로만 설정해주세요) : ")


# ## Option1 : 기존에 학습된 모델을 불러오기
# input : model 폴더명, 없으면 오류 발생 (원하는 버전 모델 있으면 지정 필요, 없으면 Enter -> Default)
# model 폴더 안에 bin파일, 없으면 오류 발생
                                     
dtm_ver=input("불러올 DTM모델 명을 입력하세요 : ")
if dtm_ver=='':
    print("\nDefault DTM모델을 불러옵니다.")
    mdl=tp.DTModel.load("./model/Default/DTM.bin")
else:
    print("\n기존에 학습된 %s모델을 불러옵니다."%(dtm_ver))
    mdl=tp.DTModel.load("./model/"+dtm_ver+".bin")

data_ver=input("사용할 데이터 폴더명을 입력하세요(Default: 엔터키): ")
if data_ver=='':
    DATA_DIR = './data/DTM/Default/'
else:
    DATA_DIR = './data/DTM/'+data_ver+'/'

# Corpus Load Function

def data_feeder(input_file):
    for line in tqdm(open(input_file, encoding='CP949')):
        fd = line.strip().split(maxsplit=1)
        timepoint = int(fd[0])
        yield fd[1], None, {'timepoint':timepoint}

train_type = input("Train NTIS Type : ")
if train_type in ['ALL','all','전체','0']:
        train_type='DTM_NTIS_ALL'
        print("\nNTIS 전체 Data로 학습을 진행합니다.")
elif train_type in ['ICT','ict','1']:
        train_type='DTM_NTIS_ICT'
        print("\nNTIS ICT Data로만 학습을 진행합니다.")
else:
    print("* Error * 데이터 타입을 다시 입력해주세요 ")

# 토픽수, 연도, 학습수 사용자 입력 

num_topics=int(input("토픽 갯수 : "))
start_year = int(input("REPORT 시작 연도 : "))
num_time=int(input("REPORT 최종 연도 : ")) - start_year + 1

#import csv
# input : pkl 형태의 파일, 없으면 오류 발생 (data 폴더)
#unusing_dic = pd.read_csv(filepath_or_buffer='./data/Default/UNUSING_DIC.csv', encoding="cp949", sep=",")
#temp_f =[]
#remove_set =unusing_dic["term"].tolist()

# input : data/DTM 폴더에 사용자별 txt 파일 Load, 없으면 오류 발생

#f = open("./data/DTM/NTIS/DTM_NTIS_ICT_DEL.txt", 'w')
#for i in range(corpus) :
#    temp = str(corpus.__getitem__(i))
#    temp = temp.strip("<tomotopy.Document with words=\"" "\">")
#    temp = str(corpus[i].timepoint) + " " + temp + "\n"
#    f.writelines(temp)
#f.close()

# Corpus Load
remove_set ={'0','1','2','3','4','5','6','7','8','9','10','할',"위한"}
porter_stemmer = nltk.PorterStemmer().stem
corpus = tp.utils.Corpus(
    tokenizer=tp.utils.SimpleTokenizer(porter_stemmer),
    stopwords= remove_set
)
#corpus.process(data_feeder(DATA_DIR+train_type+'.txt'))
corpus.process(data_feeder(DATA_DIR+'DTM_REPORT_ICT.txt'))

# ## Option1 or Option2 수행 후 아래부터 실행

temp_timepoint = []
for i in range(len(corpus)) :
    if corpus[i].timepoint == 4 :
        temp_timepoint.append(5)
    elif corpus[i].timepoint == 3 :
        temp_timepoint.append(4)
    elif corpus[i].timepoint == 2 :
        temp_timepoint.append(3)
    elif corpus[i].timepoint == 1 :
        temp_timepoint.append(2)
    elif corpus[i].timepoint == 0 :
        temp_timepoint.append(1)    
    else :
        temp_timepoint.append(corpus[i].timepoint)

doc_insts = []
for i in range(len(corpus)) :
    if temp_timepoint[i] < 5 : 
        doc_insts.append(mdl.make_doc(str(corpus[i]).strip("<tomotopy.Document with words=\"""\">").split(" "),temp_timepoint[i]))
topic_dist, ll = mdl.infer([doc_insts][0])


# ### 2021년 분포(2020년으로 가정)
doc_insts_2 = []
for i in range(len(corpus)) :
    if temp_timepoint[i] == 5 : 
        doc_insts_2.append(mdl.make_doc(str(corpus[i]).strip("<tomotopy.Document with words=\"""\">").split(" "),4))
topic_dist_2, ll_2 = mdl.infer([doc_insts_2][0])


# ## 평균치계산
for i in range(1,5):
    globals()['temp_{}'.format(i)] = list(np.zeros(mdl.k))
    globals()['count_{}'.format(i)] = 0

for i in range(len(doc_insts)) :
    for k in range(mdl.k):
        globals()['temp_{}'.format(doc_insts[i].timepoint)][k] += topic_dist[i][k]
    globals()['count_{}'.format(doc_insts[i].timepoint)] += 1
        
yr_topic_dist = []
for i in range(1,5):
    yr_topic_dist.append(globals()['temp_{}'.format(i)])
    print(globals()['count_{}'.format(i)])

temp_5 = list(np.zeros(mdl.k))
count_5 = 0
for i in range(len(doc_insts_2)) :
    for k in range(mdl.k):
        temp_5[k] += topic_dist_2[i][k]
    count_5 += 1
    
yr_topic_dist.append(temp_5)
print(count_5)

#for i in range(mdl.num_timepoints):
#yr_topic.append(get_topic(i))
    
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

topics = []
topics_d = []
for i in range(mdl.k):
    for k in range(5):
        topic = str("topic") + str(i+1)
        topic_d = str("topic") + str(i+1)
        topics.append(topic)
        topics_d.append(topic_d)

labels=[]
labels_d=[]
for i in range(0,len(topics),5):
    labels.append(topics[i])
    labels_d.append(topics_d[i])

# NTIS/NEWS의 연도별 토픽확률 데이터프레임 생성

rep_dict = {}
for i in range(mdl.num_timepoints):
    rep_dict[i]=yr_topic_norm[i]
rep_topic_dist=pd.DataFrame(rep_dict)

# NTIS/NEWS의 평균 토픽확률 도출

s1=[]
for i in range(mdl.k):
    s1.append(sum(rep_topic_dist.iloc[i]))
    
s1=[i/mdl.num_timepoints for i in s1]

rep_topic_dist['avg']=s1
rep_topic_dist['topic']=labels_d

# output type : Average ICT NTIS/NEWS 연도별 html 파일 (html/DTM 폴더)

fig = make_subplots(
        rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1
    )
fig.add_bar(x=labels_d, y=rep_topic_dist['avg'], name ="REPORT", row=1, col=1)
fig.update_layout(title=go.layout.Title(text="Average DTM Topic Distribution",
                                        font=go.layout.title.Font(size=20)),
                 yaxis_title="Prob")

fig.update_xaxes(visible=True, showticklabels=True)
fig.update_yaxes(visible=True, showticklabels=True)
fig.show()
#fig.write_html('./html/DTM/Average_DTM'+str(num_topics)+'.html')
fig.write_html('./html/DTM/REPORT/Average_DTM'+str(23)+'.html')


# ## 연도별 변화 확인
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

def plot_topn_topics_rep(topn):

    total_ntis = []

    for i in range(topn):
        trace = go.Scatter(y = rep_topic_dist.sort_values(by=['avg'],axis=0,ascending=False).iloc[i][:mdl.num_timepoints], mode = 'lines+markers',hovertext='REP '+ rep_topic_dist.sort_values(by=['avg'],axis=0,ascending=False)['topic'].iloc[i][5:], hoverinfo='text', name = 'REP '+ rep_topic_dist.sort_values(by=['avg'],axis=0,ascending=False)['topic'].iloc[i], 
                           marker=dict(symbol='cross', color=color[i]),
                           line = dict(color=color[i]))
        total_ntis.append(trace)

    
    layout = go.Layout(title='REP DTM 상위 '+str(topn)+'개 토픽별 트렌드',legend=dict(x=0,y=-1.7),margin=dict(l=20, r=20, t=60, b=300),paper_bgcolor="White",
                       autosize=True,title_font_size=30,font=dict(size=15),hoverlabel=dict(
        
        font_size=16,
        font_family="Rockwell"
    ),
                    xaxis=dict({"tickvals":list(range(0,mdl.num_timepoints)),
                            "ticktext":[str(i) for i in list(range(start_year,start_year+mdl.num_timepoints))],
                           "title":"Year"}),
                    yaxis=dict({"title":"Prob"}),
                    height=2000,width=600)
    
    gen_ntis = go.Figure(data=total_ntis, layout=layout)
    pyo.iplot(gen_ntis)
    gen_ntis.write_html('./html/DTM/REPORT/상위'+str(topn)+'.html')

# 앞단에서 입력했던 토픽 개수 초과 시 에러 발생 유의

plot_topn_topics_rep(int(input("상위 토픽 갯수 : ")))


# ## 대표문서 확인
## 대표 문서 확인
for i in tqdm(range(mdl.k)) : 
    for m in range(num_time) :
        globals()['doc_to_{}'.format(str(i)+"_"+str(m))] = 0
        globals()['doc_time_{}'.format(str(i)+"_"+str(m))] = 0
        globals()['doc_prob_{}'.format(str(i)+"_"+str(m))] = 0
        globals()['doc_index_{}'.format(str(i)+"_"+str(m))] = 0

for k in tqdm(range(len(doc_insts))):
    temp_doc_to = doc_insts[k].get_topics()[0][0]
    temp_doc_time = doc_insts[k].timepoint
    temp_doc_prob = doc_insts[k].get_topics()[0][1]
    if globals()['doc_prob_{}'.format(str(temp_doc_to)+"_"+str(temp_doc_time))] < temp_doc_prob :
        globals()['doc_to_{}'.format(str(temp_doc_to)+"_"+str(temp_doc_time))] = temp_doc_to 
        globals()['doc_time_{}'.format(str(temp_doc_to)+"_"+str(temp_doc_time))] = temp_doc_time
        globals()['doc_prob_{}'.format(str(temp_doc_to)+"_"+str(temp_doc_time))] = temp_doc_prob
        globals()['doc_index_{}'.format(str(temp_doc_to)+"_"+str(temp_doc_time))] = k

## 대표 문서 확인
for i in tqdm(range(mdl.k)) : 
    globals()['doc_to_2_{}'.format(str(i)+"_"+str(5))] = 0
    globals()['doc_time_2_{}'.format(str(i)+"_"+str(5))] = 5
    globals()['doc_prob_2_{}'.format(str(i)+"_"+str(5))] = 0
    globals()['doc_index_2_{}'.format(str(i)+"_"+str(5))] = 0

for k in tqdm(range(len(doc_insts_2))):
    temp_doc_to_2 = doc_insts_2[k].get_topics()[0][0]
    temp_doc_prob_2 = doc_insts_2[k].get_topics()[0][1]
    if globals()['doc_prob_2_{}'.format(str(temp_doc_to_2)+"_"+str(5))] < temp_doc_prob_2 :
        globals()['doc_to_2_{}'.format(str(temp_doc_to_2)+"_"+str(5))] = temp_doc_to_2 
        globals()['doc_prob_2_{}'.format(str(temp_doc_to_2)+"_"+str(5))] = temp_doc_prob_2
        globals()['doc_index_2_{}'.format(str(temp_doc_to_2)+"_"+str(5))] = k

for i in tqdm(range(mdl.k)) : 
    for m in range(num_time-1) :
        print("\n topic_" + str(i+1) + "_" + str(m+2017))
        print("index : " + str(globals()['doc_index_{}'.format(str(i)+"_"+str(m))]))
        print("prob : " + str(globals()['doc_prob_{}'.format(str(i)+"_"+str(m))]))
        print(doc_insts[globals()['doc_index_{}'.format(str(i)+"_"+str(m))]])
    print("\n topic_" + str(i+1) + "_" + str(2021))
    print("index : " + str(globals()['doc_index_2_{}'.format(str(i)+"_"+str(5))]))
    print("prob : " + str(globals()['doc_prob_2_{}'.format(str(i)+"_"+str(5))]))
    print(doc_insts_2[globals()['doc_index_2_{}'.format(str(i)+"_"+str(5))]])

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

dir(doc_insts[0])
doc_insts[0]
doc_insts[0].get_words()
mdl.get_topic_words(0,0,5)
mdl.get_topic_word_dist(0,0)


# # VAR/LSTM data 생성
try:
    os.mkdir('./prediction/REPORT/')
except FileExistsError:
    print("폴더가 이미 존재합니다.")

# output type : 예측 모델링을 위한 pkl 파일 (data/PREDICTION 폴더)

#del rep_topic_dist['avg']
#del rep_topic_dist['rel']
rep_topic_dist.to_pickle("./prediction/REPORT/REPORT_topic_dist_"+str("23")+".pkl")
#df.to_pickle("./prediction/REPORT/WORDS_"+str("23")+".pkl")
rep_topic_dist.to_excel("./prediction/REPORT/REPORT_topic_dist_"+str("23")+".xlsx")
#df.to_excel("./prediction/NTIS/"+"WORDS_"+str("23")+".xlsx")

# ## vis 시각화
import pyLDAvis
import pyLDAvis.gensim as gensimvis

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
    pyLDAvis.save_html(prepared_data, 'dtmvis_{}.html'.format(timepoint))