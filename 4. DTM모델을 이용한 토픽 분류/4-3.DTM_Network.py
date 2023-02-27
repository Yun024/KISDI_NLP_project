#!/usr/bin/env python
# coding: utf-8
# Package Load
import pickle
import pyLDAvis
import math
import copy
import os
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
version_name=str(datetime.today().strftime("%Y%m%d")) # 버전 정보 변수
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import pandas as pd
import networkx as nx
import operator
import numpy as np
import scipy
import itertools
import nltk
import matplotlib.font_manager as fm
import matplotlib as mpl
#pip install scipy networkx==2.6.3  # coo_array에러나면 실행

os.chdir("C:\\Users\\newcomer02\\NTIS_Project")

user_name = input("사용자명을 설정해주세요(영어로만 설정해주세요) : ")

# ## Option1 : 기존에 학습된 모델을 불러오기
# input : model 폴더명, 없으면 오류 발생 (원하는 버전 모델 있으면 지정 필요, 없으면 Enter -> Default)
# model 폴더 안에 bin파일, 없으면 오류 발생
                                     
dtm_ver=input("불러올 DTM모델 명을 입력하세요 : ")
if dtm_ver=='':
    print("\nDefault DTM모델을 불러옵니다.")
    mdl=tp.DTModel.load("./model/DTM_model_41.bin")
else:
    print("\n기존에 학습된 %s모델을 불러옵니다."%(dtm_ver))
    mdl=tp.DTModel.load("./model/"+dtm_ver+".bin")

# 토픽수, 연도, 학습수 사용자 입력 
num_topics=int(input("토픽 갯수 : "))
start_year = int(input("NTIS 시작 연도 : "))
num_time=int(input("NTIS 최종 연도 : ")) - start_year + 1

# # 파일 불러오기 
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

wd_network_dtm=input("워드네트워크를 만들 데이터를 입력하세요(Default: 엔터키): ")
if wd_network_dtm=='':
    print("\nDefault 워드네트워크 데이터를 불러옵니다.")
    wd_network_dtm= pd.read_csv("./data/Default/NTIS/DTM/wd_network_dtm_"+str(num_topics) + ".csv",encoding="cp949",index_col=0)

else:
    print("\n%s 에 워드네트워크 데이터를 불러옵니다."%(wd_network_dtm))
    wd_network_dtm= pd.read_csv("./data/LDA/"+data_ver+"/" + wd_network + ".csv",encoding="cp949",index_col=0)
#    wd_network_dtm_41

wd_network_dtm


# # 단어 네트워크

topic_num=int(input("분석하고 싶은 토픽 번호(1~41)를 입력하십시오:"))
timepoint=int(input("분석하고 싶은 timepoint(0~4)를 입력하십시오 ex)2017:0 :"))
if timepoint == 0:
    wd_network  = wd_network_dtm[wd_network_dtm["timepoint"]==0]
elif timepoint == 1:
    wd_network  = wd_network_dtm[wd_network_dtm["timepoint"]==1]
elif timepoint == 2:
    wd_network  = wd_network_dtm[wd_network_dtm["timepoint"]==2]
elif timepoint == 3:
    wd_network  = wd_network_dtm[wd_network_dtm["timepoint"]==3]
elif timepoint == 4:
    wd_network  = wd_network_dtm[wd_network_dtm["timepoint"]==4]

# wd_network = pd.DataFrame(index=range(len(corpus)),columns=["corpus","timepoint","mdl.docs","Topic","Topic_prob"])

# for i in tqdm(range(len(corpus))):
#     wd_network.iloc[i][0] =  ' '.join(map(str, corpus[i]))

# for u  in tqdm(range(len(corpus))):
#     wd_network.iloc[u][1] = corpus[u].timepoint
    
# for j in tqdm(range(len(mdl.docs))):
#     wd_network.iloc[j][2] = mdl.docs[j]

# for z in tqdm(range(len(mdl.docs))):
#     wd_network.iloc[z][3] = mdl.docs[z].get_topics(top_n=mdl.k)[0][0] + 1

# for c in tqdm(range(len(mdl.docs))):
#     wd_network.iloc[c][4] = mdl.docs[c].get_topics(top_n=mdl.k)[0][1]

# wd_network_dmt = wd_network.copy()

# wd_network_dtm.to_csv("./data/Default/NTIS/DTM/wd_network_dtm_"+str(num_topics)+".csv",encoding="cp949")

## 토픽 별 상위 1000개 데이터를 이용한 사전 생성 
using_word= []
for i in range(500):
    using_word.append(mdl.get_topic_words(topic_num-1,top_n=500,timepoint=timepoint)[i][0]) 

df = pd.DataFrame()
if len(wd_network[wd_network["Topic"]==topic_num].sort_values("Topic_prob",ascending=False)) < 100:
    a = len(wd_network[wd_network["Topic"]==topic_num].sort_values("Topic_prob",ascending=False))
else :
    a = 100 
for i in tqdm(range(a)):
    temp = wd_network[wd_network["Topic"]==topic_num].sort_values("Topic_prob",ascending=False).iloc[i][0]
    temp2 = []
    for j in (temp.split(" ")):
        if j in using_word:
            temp2.append(j)
        else:
            continue
    count = {}
    for c,a in tqdm(enumerate(temp2)):  # i는 숫자 a는 1행 
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
print(freq_num)

G_centrality = nx.Graph()
for i in tqdm(range(len((np.where(df3["freq"]>=freq_num))[0]))):
    G_centrality.add_edge(df3["term1"][i],df3["term2"][i],weight=int(df3["freq"][i]))

dgr = nx.degree_centrality(G_centrality)      #연결 중심성
btw = nx.betweenness_centrality(G_centrality) #매개 중심성
cls = nx.closeness_centrality(G_centrality)   #근접 중심성
egv = nx.eigenvector_centrality(G_centrality) #고유벡터 중심성
pgr = nx.pagerank(G_centrality) #페이지랭크 안됨 

sorted_dgr = sorted(dgr.items(), key=operator.itemgetter(1), reverse=True)
sorted_btw = sorted(btw.items(), key=operator.itemgetter(1), reverse=True)
sorted_cls = sorted(cls.items(), key=operator.itemgetter(1), reverse=True)
sorted_egv = sorted(egv.items(), key=operator.itemgetter(1), reverse=True)
sorted_pgr = sorted(pgr.items(), key=operator.itemgetter(1), reverse=True)

G= nx.Graph()


for i in tqdm(range(len(sorted_pgr))):
    G.add_node(sorted_pgr[i][0],nodesize=sorted_dgr[i][1])

for i in tqdm(range(len((np.where(df3["freq"]>=freq_num))[0]))):
    G.add_weighted_edges_from([(df3["term1"][i],df3["term2"][i],int(df3["freq"][i]))])
    
sizes = [G.nodes[node]["nodesize"]*2000 for node in G]


# ## 텍스트 네트워크 폰트 설정 및 그래프 그리기
# #### <에러나면 아래로 가서 패키지 실행 후 올라오기>

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

#pos=nx.fruchterman_reingold_layout(G)
#pos=nx.spectral_layout(G)
#pos=nx.random_layout(G)
#pos=nx.shell_layout(G)
#pos=nx.circular_layout(G)
#pos=nx.spring_layout(G,k=3.5,iterations=100)
#pos=nx.kamada_kawai_layout(G)

plt.figure(figsize=(16,8)); 
nx.draw_networkx(G,node_size=sizes,pos=nx.kamada_kawai_layout(G),**options,font_family=fontprop)
ax = plt.gca()
ax.collections[0].set_edgecolor("#555555")
#plt.savefig("temp.png", bbox_inches='tight')
plt.show()

#pos=nx.fruchterman_reingold_layout(G)
#pos=nx.spectral_layout(G)
#pos=nx.random_layout(G)
#pos=nx.shell_layout(G)
#pos=nx.circular_layout(G)
#pos=nx.spring_layout(G,k=3.5,iterations=100)
#pos=nx.kamada_kawai_layout(G)

plt.figure(figsize=(16,8)); 
nx.draw_networkx(G,node_size=sizes,pos=nx.spring_layout(G,k=3.5,iterations=100),**options,font_family=fontprop)
ax = plt.gca()
ax.collections[0].set_edgecolor("#555555")
# plt.savefig("C:\\Users\\KISDI\\LDA\\html\\LDA\\NTIS\\NTIS_WD_NETWORK_TOPIC_"+ str(topic_num) + ".png", bbox_inches='tight')
plt.show()

print("** degree **")
for x in range(len(G)):
    print(sorted_dgr[x])

print("** betweenness **")
for x in range(len(G)):
    print(sorted_btw[x])

print("** closeness **")
for x in range(len(G)):
    print(sorted_cls[x])


# ## 최초 1회 실행

print ('버전: ', mpl.__version__)
print ('설치 위치: ', mpl.__file__)
print ('설정 위치: ', mpl.get_configdir())
print ('캐시 위치: ', mpl.get_cachedir())

print ('설정파일 위치: ', mpl.matplotlib_fname())

[(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]

plt.rc("font", family="NanumGothicCoding")
print(plt.rcParams["font.family"])

#단어끼리 서로 빈도를 세는 데이터셋을 만들었을 때 Gaphi로 시각화하는 것 전단계: graphml 확장자 형식으로 만들기
class MakeGraphml:
    def make_graphml(self, pair_file, graphml_file):
        out = open(graphml_file, 'w', encoding = 'utf-8')
        entity = []
        e_dict = {}
        count = []
        for i in range(len(pair_file)):
            e1 = pair_file.iloc[i,0]
            e2 = pair_file.iloc[i,1]
            #frq = ((word_dict[e1], word_dict[e2]),  pair.split('\t')[2])
            frq = ((e1, e2), pair_file.iloc[i,2])
            if frq not in count: count.append(frq)   # ((a, b), frq)
            if e1 not in entity: entity.append(e1)
            if e2 not in entity: entity.append(e2)
        print('# terms: %s'% len(entity))
        #create e_dict {entity: id} from entity
        for i, w in enumerate(entity):
            e_dict[w] = i + 1 # {word: id}
        out.write(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlnshttp://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">" +
            "<key id=\"d1\" for=\"edge\" attr.name=\"weight\" attr.type=\"double\"/>" +
            "<key id=\"d0\" for=\"node\" attr.name=\"label\" attr.type=\"string\"/>" +
            "<graph id=\"Entity\" edgedefault=\"undirected\">" + "\n")
        # nodes
        for i in entity:
            out.write("<node id=\"" + str(e_dict[i]) +"\">" + "\n")
            out.write("<data key=\"d0\">" + i + "</data>" + "\n")
            out.write("</node>")
        # edges
        for y in range(len(count)):
            out.write("<edge source=\"" + str(e_dict[count[y][0][0]]) + "\" target=\"" + str(e_dict[count[y][0][1]]) + "\">" + "\n")
            out.write("<data key=\"d1\">" + str(count[y][1]) + "</data>" + "\n")
            #out.write("<edge source=\"" + str(count[y][0][0]) + "\" target=\"" + str(count[y][0][1]) +"\">"+"\n")
            #out.write("<data key=\"d1\">" + str(count[y][1]) +"</data>"+"\n")
            out.write("</edge>")
        out.write("</graph> </graphml>")
        print('now you can see %s' % graphml_file)
        #pairs.close()
        out.close()


gm = MakeGraphml()

graphml_file = 'wd_network.graphml'

gm.make_graphml(df3.iloc[0:len((np.where(df3["freq"]>=5))[0]),:], graphml_file)


# # 자동

year_range = input("사용년도의 범위를 입력하시오 ex)2017~2021:5 :")
topic_len = input("토픽의 개수를 입력하시오:")

wd_network_dtm

for timepoint in tqdm(range(int(year_range))):

    if timepoint == 0:
        wd_network  = wd_network_dtm[wd_network_dtm["timepoint"]==0]
    elif timepoint == 1:
        wd_network  = wd_network_dtm[wd_network_dtm["timepoint"]==1]
    elif timepoint == 2:
        wd_network  = wd_network_dtm[wd_network_dtm["timepoint"]==2]
    elif timepoint == 3:
        wd_network  = wd_network_dtm[wd_network_dtm["timepoint"]==3]
    elif timepoint == 4:
        wd_network  = wd_network_dtm[wd_network_dtm["timepoint"]==4]  

    for topic_num in range(int(topic_len)):
        using_word= []
        for i in range(500):
            using_word.append(mdl.get_topic_words(topic_num,top_n=500,timepoint=timepoint)[i][0]) 
            
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
        egv = nx.eigenvector_centrality(G_centrality) #고유벡터 중심성
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
        
        plt.savefig("C:\\Users\\newcomer02\\NTIS_Project\\data\\Default\\NTIS\\DTM\\Topic_"+ str(timepoint+2017) + "\\Topic" + str(topic_num+1) +".png", bbox_inches='tight')
