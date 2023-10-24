#!/usr/bin/env python
# coding: utf-8

#https://bab2min.github.io/tomotopy/v0.12.3/kr/#tomotopy.LDAModel
# Package Load
import pickle
import pyLDAvis
import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
import tomotopy as tp
import tomotopy.coherence as tpc
from gensim.corpora import Dictionary
from datetime import datetime
from tqdm import tqdm
from datetime import datetime
from timeit import default_timer
import plotly.express as px
import plotly.offline
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from collections import Counter
import networkx as nx
import operator
import scipy
import matplotlib.font_manager as fm
import matplotlib as mpl
import itertools

upper_topic_num = int(input("LDA 모델 최적 토픽 수를 검증하기 위한 최대 토픽 수를 입력하세요. (※ 10이상의 자연수, 권장값 = 250개)"))
iteration = int(input("학습 Iteration 수를 입력하세요. (※ 100이상의 자연수, 권장값 = 1000회)"))

DATA_DIR = ".\\data\\LDA\\NTIS\\corpus_topic\\"

for topic_ver in tqdm(range(len(os.listdir(DATA_DIR)))):
    
    # 코퍼스 다운로드
    corpus = []
    for n, line in enumerate(open(DATA_DIR + "LDA_topic_"+str(topic_ver+1)+".txt", encoding='CP949')):
        doc=line.strip().split()
        corpus.append(doc)
    
    # min_cf 자동 입력
    sct = sum(corpus,[])
    df_sct = pd.DataFrame(Counter(sct).most_common(),columns=["word","count"])
    mcf = df_sct.iloc[:int(len(df_sct)*  0.2),:]["count"].iloc[-1]-1
    
    # LDA모델 최적화
   
    score_c_list=[]
   
    for opt_tp in range(5, upper_topic_num+5, 5): 
        model=tp.LDAModel(k=opt_tp, min_cf=mcf, rm_top=64 ,tw=tp.TermWeight.PMI, seed=42)
        for cnt in range(len(corpus)):
            model.add_doc(corpus[cnt])
        model.train(iter=iteration, workers=0)
        coherence_c=tpc.Coherence(model, coherence='c_v')
        score_c=coherence_c.get_score()
        score_c_list.append(score_c)
    
    max_index = score_c_list.index(max(score_c_list)) +1 
    
    score_c_list_2=[]

    for opt_tp in range(max_index*5-5, max_index*5+5, 1): 
        model=tp.LDAModel(k=opt_tp, min_cf=mcf, rm_top=64 ,tw=tp.TermWeight.PMI, seed=42)
        for cnt in range(len(corpus)):
            model.add_doc(corpus[cnt])
        model.train(iter=iteration, workers=0)
        coherence_c_2=tpc.Coherence(model, coherence='c_v')
        score_c_2=coherence_c_2.get_score()
        score_c_list_2.append(score_c_2)
        
    score_c_list_2.index(max(score_c_list_2))
    
    
    # 토픽개수 결정
    topic_num = range(max_index*5-5, max_index*5+5, 1)[score_c_list_2.index(max(score_c_list_2))]


    # LDA 모델 학습 

    s = default_timer()

    mdl=tp.LDAModel(min_cf=mcf, rm_top=64, k=topic_num, alpha=0.1, eta=0.01, tw=tp.TermWeight.PMI, seed=42)
    for cnt in range(len(corpus)):
        mdl.add_doc(corpus[cnt])
    mdl.train(iter=iteration, workers=0)

    e = default_timer()
    mdl.save('./model/middle_model/LDA_MODEL_topic_'+str(topic_ver+1)+'.bin')
    
    # 각 토픽 별 시트 저장 
    final = pd.DataFrame()
    for j in range(mdl.k):
        word = ["Topic_"+ str(j+1)]
        topic_word = pd.DataFrame(np.array(mdl.get_topic_words(j,top_n=15))[:,0].tolist(),columns=word)
        final = pd.concat([final,topic_word],axis=1)
    
    if topic_ver==0:
        final.to_excel(excel_writer="./data\\LDA\\NTIS\\topic_middle_word.xlsx",
                      sheet_name="topic1",index=False, encoding="utf-8")
    else:
        file_name = "./data\\LDA\\NTIS\\topic_middle_word.xlsx"
        writer = pd.ExcelWriter(file_name,mode="a",engine="openpyxl",if_sheet_exists="overlay")
        final.to_excel(writer,sheet_name="topic"+str(topic_ver+1),startcol=0,startrow=0,index=False)
        writer.save()
        writer.close()
        
        
    path = "C:\\Users\\KISDI\\LDA\\html\\LDA\\NTIS\\Topic_Network_NTIS_middle\\"+"Topic_"+str(topic_ver+1)
    os.mkdir(path)
    
    # 단어 네트워크 
    wd_network = pd.DataFrame(index=range(len(corpus)),columns=["corpus","mdl.docs","Topic","Topic_prob"])

    for u in tqdm(range(len(corpus))):
        wd_network.iloc[u][0] =  ' '.join(map(str, corpus[u]))

    for j in tqdm(range(len(mdl.docs))):
        wd_network.iloc[j][1] = mdl.docs[j]

    for z in tqdm(range(len(mdl.docs))):
        wd_network.iloc[z][2] = mdl.docs[z].get_topics(top_n=mdl.k)[0][0] + 1

    for c in tqdm(range(len(mdl.docs))):
        wd_network.iloc[c][3] = mdl.docs[c].get_topics(top_n=mdl.k)[0][1]
        
    topic_len = int(topic_num)
    
    for topic_num in tqdm(range(int(topic_len))):
        using_word= []
        try:
            for i in range(500):
                using_word.append(mdl.get_topic_words(topic_num,top_n=500)[i][0]) 

            df = pd.DataFrame()
            if len(wd_network[wd_network["Topic"]==topic_num+1].sort_values("Topic_prob",ascending=False)) < 100:
                a = len(wd_network[wd_network["Topic"]==topic_num+1].sort_values("Topic_prob",ascending=False))
            else :
                a = 100 
            for i in range(a):
                temp = wd_network[wd_network["Topic"]==topic_num+1].sort_values("Topic_prob",ascending=False).iloc[i][0]
                temp2 = []
                for j in (temp.split(" ")):
                    if j in using_word:
                        temp2.append(j)
                    else:
                        continue
                count = {}
                for c,a in enumerate(temp2):  # i는 숫자 a는 1행 
                    for b in temp2[c+1:]:
                        if a>b:
                            count[b,a] = count.get((b,a),0)+1
                        else:
                            count[a,b] = count.get((a,b),0)+1
                word_df = pd.DataFrame.from_dict(count,orient="index")  
                df = pd.concat([df,word_df])

            df.reset_index(inplace=True)
            df[1] = pd.DataFrame(df["index"].tolist())[0]
            df[2] = pd.DataFrame(df["index"].tolist())[1]
            df = df[df[1]!=df[2]]
            df= pd.DataFrame(df.groupby("index")[0].sum())

            list1 = []
            for i in range(len(df)):
                list1.append([df.index[i][0],df.index[i][1],df[0][i]])

            df2 = pd.DataFrame(list1,columns=["term1","term2","freq"])
            df3 = df2.sort_values(by=["freq"],ascending=False)
            df3 = df3.reset_index(drop=True)

            i =1
            while len((np.where(df3["freq"]>=i))[0])>100:
                i +=1
            freq_num=i

            G_centrality = nx.Graph()
            for i in range(len((np.where(df3["freq"]>=freq_num))[0])):
                G_centrality.add_edge(df3["term1"][i],df3["term2"][i],weight=int(df3["freq"][i]))

            dgr = nx.degree_centrality(G_centrality)      #연결 중심성
            btw = nx.betweenness_centrality(G_centrality) #매개 중심성
            cls = nx.closeness_centrality(G_centrality)   #근접 중심성
            egv = nx.eigenvector_centrality(G_centrality, tol=1.0e-3) #고유벡터 중심성
            pgr = nx.pagerank(G_centrality) #페이지랭크 안됨 

            sorted_dgr = sorted(dgr.items(), key=operator.itemgetter(1), reverse=True)
            sorted_btw = sorted(btw.items(), key=operator.itemgetter(1), reverse=True)
            sorted_cls = sorted(cls.items(), key=operator.itemgetter(1), reverse=True)
            sorted_egv = sorted(egv.items(), key=operator.itemgetter(1), reverse=True)
            sorted_pgr = sorted(pgr.items(), key=operator.itemgetter(1), reverse=True)

            G= nx.Graph()

            for i in range(len(sorted_pgr)):
                G.add_node(sorted_pgr[i][0],nodesize=sorted_dgr[i][1])
            for i in range(len((np.where(df3["freq"]>=freq_num))[0])):
                G.add_weighted_edges_from([(df3["term1"][i],df3["term2"][i],int(df3["freq"][i]))])

            sizes = [G.nodes[node]["nodesize"]*2000 for node in G]

            ## 마이너스 폰트 깨지는 문제에 대한 대처
            mpl.rcParams['axes.unicode_minus'] = False
            font_fname = "C:\\Windows\\Fonts\\NanumGothicCoding-bold.ttf"
            fontprop = fm.FontProperties(fname=font_fname,size=10).get_name()

            options={
                "edge_color":'#FFDEA2',
                "width":1,
                "with_labels":True,
                "font_weight":"bold",
            }

            plt.figure(figsize=(16,8)); 
            nx.draw_networkx(G,node_size=sizes,pos=nx.kamada_kawai_layout(G),**options,font_family=fontprop)
            ax = plt.gca()
            ax.collections[0].set_edgecolor("#555555")

            plt.savefig("C:\\Users\\KISDI\\LDA\\html\\LDA\\NTIS\\Topic_Network_NTIS_middle\\"+"Topic_"+str(topic_ver+1) +"\\Topic_" + str(topic_num+1)  +".png", bbox_inches='tight')
        except:
            pass

# # 하나씩

topic_ver = int(input("불러올 토픽번호를 입력하세요 :")) 

DATA_DIR = "C:\\Users\\KISDI\\LDA\\data\\LDA\\NTIS\\corpus_topic\\"
# 코퍼스 다운로드
corpus = []
for n, line in enumerate(open(DATA_DIR + "LDA_topic_"+str(topic_ver)+".txt", encoding='CP949')):
    doc=line.strip().split()
    corpus.append(doc)
# 모델 다운로드
mdl=tp.LDAModel.load("./model/middle_model/LDA_MODEL_topic_"+str(topic_ver)+".bin")

wd_network = pd.DataFrame(index=range(len(corpus)),columns=["corpus","mdl.docs","Topic","Topic_prob"])

for u in tqdm(range(len(corpus))):
    wd_network.iloc[u][0] =  ' '.join(map(str, corpus[u]))

for j in tqdm(range(len(mdl.docs))):
    wd_network.iloc[j][1] = mdl.docs[j]

for z in tqdm(range(len(mdl.docs))):
    wd_network.iloc[z][2] = mdl.docs[z].get_topics(top_n=mdl.k)[0][0] + 1

for c in tqdm(range(len(mdl.docs))):
    wd_network.iloc[c][3] = mdl.docs[c].get_topics(top_n=mdl.k)[0][1]

topic_num = int(input("MIDDLE 토픽번호를 입력하세요 :")) 

using_word= []

for i in range(500):
    using_word.append(mdl.get_topic_words(topic_num-1,top_n=500)[i][0]) 

df = pd.DataFrame()
if len(wd_network[wd_network["Topic"]==topic_num].sort_values("Topic_prob",ascending=False)) < 100:
    a = len(wd_network[wd_network["Topic"]==topic_num].sort_values("Topic_prob",ascending=False))
else :
    a = 100 
for i in range(a):
    temp = wd_network[wd_network["Topic"]==topic_num].sort_values("Topic_prob",ascending=False).iloc[i][0]
    temp2 = []
    for j in (temp.split(" ")):
        if j in using_word:
            temp2.append(j)
        else:
            continue
    count = {}
    for c,a in enumerate(temp2):  # i는 숫자 a는 1행 
        for b in temp2[c+1:]:
            if a>b:
                count[b,a] = count.get((b,a),0)+1
            else:
                count[a,b] = count.get((a,b),0)+1
    word_df = pd.DataFrame.from_dict(count,orient="index")  
    df = pd.concat([df,word_df])

df.reset_index(inplace=True)
df[1] = pd.DataFrame(df["index"].tolist())[0]
df[2] = pd.DataFrame(df["index"].tolist())[1]
df = df[df[1]!=df[2]]
df= pd.DataFrame(df.groupby("index")[0].sum())

list1 = []
for i in range(len(df)):
    list1.append([df.index[i][0],df.index[i][1],df[0][i]])

df2 = pd.DataFrame(list1,columns=["term1","term2","freq"])
df3 = df2.sort_values(by=["freq"],ascending=False)
df3 = df3.reset_index(drop=True)

i =1
while len((np.where(df3["freq"]>=i))[0])>100:
    i +=1
freq_num=i

G_centrality = nx.Graph()
for i in range(len((np.where(df3["freq"]>=freq_num))[0])):
    G_centrality.add_edge(df3["term1"][i],df3["term2"][i],weight=int(df3["freq"][i]))

dgr = nx.degree_centrality(G_centrality)      #연결 중심성
btw = nx.betweenness_centrality(G_centrality) #매개 중심성
cls = nx.closeness_centrality(G_centrality)   #근접 중심성
egv = nx.eigenvector_centrality(G_centrality, tol=1.0e-3) #고유벡터 중심성
pgr = nx.pagerank(G_centrality) #페이지랭크 안됨 

sorted_dgr = sorted(dgr.items(), key=operator.itemgetter(1), reverse=True)
sorted_btw = sorted(btw.items(), key=operator.itemgetter(1), reverse=True)
sorted_cls = sorted(cls.items(), key=operator.itemgetter(1), reverse=True)
sorted_egv = sorted(egv.items(), key=operator.itemgetter(1), reverse=True)
sorted_pgr = sorted(pgr.items(), key=operator.itemgetter(1), reverse=True)

G= nx.Graph()

for i in range(len(sorted_pgr)):
    G.add_node(sorted_pgr[i][0],nodesize=sorted_dgr[i][1])
for i in range(len((np.where(df3["freq"]>=freq_num))[0])):
    G.add_weighted_edges_from([(df3["term1"][i],df3["term2"][i],int(df3["freq"][i]))])

sizes = [G.nodes[node]["nodesize"]*2000 for node in G]

## 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False
font_fname = "C:\\Windows\\Fonts\\NanumGothicCoding-bold.ttf"
fontprop = fm.FontProperties(fname=font_fname,size=10).get_name()

options={
    "edge_color":'#FFDEA2',
    "width":1,
    "with_labels":True,
    "font_weight":"bold",
}

plt.figure(figsize=(16,8)); 
nx.draw_networkx(G,node_size=sizes,pos=nx.spring_layout(G,k=3.5,iterations=100),**options,font_family=fontprop)
ax = plt.gca()
ax.collections[0].set_edgecolor("#555555")

plt.savefig("C:\\Users\\KISDI\\LDA\\html\\LDA\\NTIS\\Topic_Network_NTIS_middle\\"+"Topic_"+str(topic_ver) +"\\Topic_" + str(topic_num)  +".png", bbox_inches='tight')