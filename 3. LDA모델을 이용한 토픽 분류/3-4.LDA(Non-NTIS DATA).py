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

import matplotlib.font_manager as fm
import matplotlib as mpl

import networkx as nx
import operator
import scipy
import itertools
#pip install scipy networkx==2.6.3  # coo_array에러나면 실행

os.chdir("C:\\Users\\newcomer02\\NTIS_Project")


# # 기존에 학습된 모델 불러오기 

# input : model 폴더명, 해당 폴더, 없으면 오류 발생 (원하는 버전 모델 있으면 지정, 없으면 Enter -> Default)
# model 폴더 안에 bin, npy 파일 필요, 없으면 오류 발생

lda_ver=input("불러올 LDA모델이 담긴 ./data/LDA 내 폴더 명을 입력하세요 (Default: 엔터키) : ")
topic_num = int(input("모델의 토픽 수를 입력하세요. (※ 권장값 = 180개)"))
if lda_ver=='':
    print("\nDefault LDA모델을 불러옵니다.")
    mdl=tp.LDAModel.load("./model/LDA_MODEL_20.bin")
#    lda_info =np.load("./data/LDA/Default/ict_index.npy")
else:
    print("\n%s 에 학습된 LDA모델을 불러옵니다."%(lda_ver[-8:]))
    mdl=tp.LDAModel.load("./model/"+lda_ver+"/LDA_MODEL_"+str(topic_num)+".bin")
#    lda_info =np.load("./data/LDA/"+lda_ver+"/ict_index.npy")

# # 파일 불러오기 
folder_name=input("사용할 데이터 폴더명을 입력하세요(Default: 엔터키): ")
lda_txt="LDA_PRED_"+ folder_name
DATA_DIR = "./data/Default/"+ folder_name+ "/"
# NKIS, NTIS, REPORT, GOV
print(DATA_DIR + lda_txt)

# input : data/LDA 폴더에 txt 형태 코퍼스 파일, 없으면 오류 발생

corpus=[]
for n, line in tqdm(enumerate(open(DATA_DIR+lda_txt+".txt", encoding='CP949'))):
    doc=line.strip().split()
    corpus.append(doc)
print("코퍼스 파일 로드가 완료되었습니다.")


# # corpus파일 전처리
##단어 없는 것 제거
temp = []
for cnt in range(len(corpus)):
    if len(corpus[cnt])  != 0 :
        temp.append(corpus[cnt])      

corpus = temp

## 불용어

from string import ascii_lowercase
from string import ascii_uppercase
alphabet_list = list(ascii_lowercase) + list(ascii_uppercase)
alphabet_list.append("ㆍ")
#+ list(topic2["0"])
#print(len(topic2["0"]))
# 52부터 topic2 51까지 알파벳 대소문자 
#alphabet_list = alphabet_list[:550] #숫자만 바꿔주면됨 
len(alphabet_list)

unusing_dic = pd.DataFrame()
unusing_dic["term"] = alphabet_list

temp_f =[]
remove_set =unusing_dic["term"].tolist()

for temp_1 in tqdm(corpus) :
    temp_t = [temp_2 for temp_2 in temp_1 if temp_2 not in remove_set]
    temp_f.append(temp_t)
    
corpus = temp_f


# # 모델 밖의 문헌 생성 후 추론하기
doc_insts = []
for i in range(len(corpus)) :
        doc_insts.append(mdl.make_doc(corpus[i]))
topic_dist, ll = mdl.infer([doc_insts][0])


# # 년도별 토픽 분포 그래프 

#cc = pd.read_pickle("./data/Default/"+ folder_name + "/HAN_Filtered_" + folder_name+ ".pkl")
#cc =cc[cc["filtered_ICT"]==1][["year"]].reset_index(drop=True)
#cc.to_csv("./data/Default/"+ folder_name + "/HAN_year_" +folder_name+ ".csv")

if folder_name == "GOV":
    df = pd.read_pickle("./data/Default/"+ folder_name + "/HAN_Filtered_" + folder_name+ ".pkl")
    df = df[df["filtered_ICT"]==1]
    df["year"] = 2023
    df = pd.DataFrame(df["year"]).reset_index(drop=True)
else:
    df = pd.read_csv("./data/Default/"+folder_name+"/HAN_year_" + folder_name+ ".csv",index_col=0)
    df = pd.DataFrame(df["year"].replace(2022,2021)) 

## 2022년 데이터 2021데이터로 변경 

df.value_counts()

year_range= int(input("몇개년 데이터인지 입력하시오 ex)5개년:5, 1개년:1 :"))
start_num = int(input("시작년도를 입력하시오:"))
end_num = int(input("마지막년도를 입력하시오:"))

year_list = []
for i in range(year_range):
    globals()["year_{}".format(str(i+start_num))] = []
for i in range(year_range):
    year_list.append(globals()["year_{}".format(str(i+start_num))])

j = 0
for i in range(start_num,(end_num+1)):
    year_list[j] = df[df==i].dropna().index.tolist()
    j +=1 

idx_list = []
for i in range(year_range):
    globals()["year_{}".format(str(i+start_num))] = []
for i in range(year_range):
    idx_list.append(globals()["year_{}".format(str(i+start_num))])
for i in range(len(year_list)):
    for j in year_list[i]:
        idx_list[i].append(topic_dist[j])

df_list = [] 
for i in range(year_range):
    globals()["year_{}".format(str(i+start_num))] = pd.DataFrame()
for i in range(year_range):
    df_list.append(globals()["year_{}".format(str(i+start_num))])
for i in range(year_range):
    for j in range(len(idx_list[i])):
        df_list[i] = pd.concat([df_list[i], pd.DataFrame(idx_list[i][j])],axis=1)

total_dist = pd.DataFrame()
for i in range(year_range):
    total_dist=pd.concat([total_dist,pd.DataFrame(df_list[i].sum(axis=1))],axis=1)
total_dist.columns = [i for i in range(start_num,end_num+1)]
dist=total_dist/np.array(total_dist.sum())
dist

#검산
dist.sum(axis=0) # 컬럼 별 1이 나오면 됨

topics_d = []
for i in range(mdl.k):
    topic_d = str("topic_") + str(i+1)
    topics_d.append(topic_d)


# ## 5개년 합산 평균 분포 그래프 
dist_one = pd.DataFrame(dist.sum(axis=1)/5)
for i in dist_one.columns.tolist():
    # output type : Average ICT NTIS/NEWS 연도별 html 파일 (html/DTM 폴더)

    fig = make_subplots(
            rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1
        )
    fig.add_bar(x=[str(i+1) for i in range(len(topics_d))], y=dist_one[i], name ="REPORT", row=1, col=1)
    fig.update_layout(title=go.layout.Title(text=folder_name+ " Average LDA Topic Distribution",
                                            font=go.layout.title.Font(size=20)),
                     yaxis_title="Prob",height=600)

    fig.update_xaxes(visible=True, showticklabels=True)
    fig.update_yaxes(visible=True, showticklabels=True)
    fig.show()
    #fig.write_html('./html/DTM/Average_DTM'+str(num_topics)+'.html')
    #fig.write_html('./html/DTM/Average_DTM'+str(23)+'.html')


# ## 5개년 년도 별 분포 그래프

for i in dist.columns.tolist():
    # output type : Average ICT NTIS/NEWS 연도별 html 파일 (html/DTM 폴더)

    fig = make_subplots(
            rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1
        )
    fig.add_bar(x=topics_d, y=dist[i], name ="NKIS", row=1, col=1)
    fig.update_layout(title=go.layout.Title(text=folder_name+" Average LDA Topic Distribution "+ str(i) ,
                                            font=go.layout.title.Font(size=20)),
                     yaxis_title="Prob",height=1000)

    fig.update_xaxes(visible=True, showticklabels=True)
    fig.update_yaxes(visible=True, showticklabels=True)
    fig.show()
    #fig.write_html('./html/DTM/Average_DTM'+str(num_topics)+'.html')
    #fig.write_html('./html/DTM/Average_DTM'+str(23)+'.html')


# # 5개년 상위 토픽별 꺾은선 그래프 

topn = pd.DataFrame(dist.sum(axis=1)).reset_index()
topn.set_index(["index"],inplace=True)
topn.sort_values(0,inplace=True,ascending=False)

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

zz = topn[:int(input("상위 토픽개수를 정해주시오:"))].index.tolist()

# zz = topn[6:].index.tolist() 
# dist = dist.drop([22,24,40,30,39,11])
# print(len(zz),len(dist))
total_ntis = []
for i in zz:
    trace = go.Scatter(y = dist.iloc[i], mode = 'lines+markers',
                        hovertext='NKIS '+ str(i+1), 
                       hoverinfo='text',
                       name = 'NKIS_Topic '+ str(i+1),
                       marker=dict(symbol='cross', color=color[i]),
                      )
    total_ntis.append(trace)

layout = go.Layout(title=folder_name + ' LDA 상위 '+str(len(zz))+'개 토픽별 트렌드',legend=dict(font=dict(size=15)),margin=dict(l=20, r=20, t=60, b=100),paper_bgcolor="White",
                   autosize=True,title_font_size=20,font=dict(size=15),hoverlabel=dict(
    
    font_size=16,
    font_family="Rockwell"
),
                xaxis=dict({"tickvals":[0,1,2,3,4],
                        "ticktext":dist.columns[:5].tolist(),
                       "title":"Year"}),
                yaxis=dict({"title":"Prob"}),
                height=500,width=500)

gen_ntis = go.Figure(data=total_ntis, layout=layout)
pyo.iplot(gen_ntis)


# # 년도에 따른 토픽 별 대표문서 보기

doc_list = []
for i in range(int(year_range)):
    globals()["year_{}".format(str(i+2017))] = []
for i in range(int(year_range)):
    doc_list.append(globals()["year_{}".format(str(i+2017))])
    for j in range(mdl.k):
        globals()["topic_{}".format(str(j+1))] = pd.DataFrame(columns=["index","prob"])
    for j in range(mdl.k):
        doc_list[i].append(globals()["topic_{}".format(str(j+1))])

for i in tqdm(range(len(year_list))):
    for j in year_list[i]:
        #if mdl.docs[j].get_topics()[0][1] > 0.5:
            doc_list[i][doc_insts[j].get_topics()[0][0]] =            pd.concat([doc_list[i][doc_insts[j].get_topics()[0][0]],
                   pd.DataFrame([(j,doc_insts[j].get_topics()[0][1])],columns=["index","prob"])])

for i in tqdm(range(len(doc_list))):
    for j in range(len(doc_list[i])):
        doc_list[i][j] = doc_list[i][j].sort_values("prob",ascending=False)

tt  = int(input("상위 몇개 토픽을 볼 것인지 입력하시오:"))

for i in tqdm(range(mdl.k)) : 
    print("\n\n\n topic_" + str(i+1))
    for u in range(len(doc_list)):
        print("\n\nYear_"+str(u+2017))
        for j in range(tt):
            try:    
                print("\nindex : " + str(doc_list[u][i]["index"].iloc[:j+1].tolist()[j]))
                print("prob : " + str(doc_list[u][i]["prob"].iloc[:j+1].tolist()[j]))
                print(doc_insts[doc_list[u][i]["index"].iloc[:j+1].tolist()[j]])
            except IndexError:
                print("\n해당 토픽과 년도에 존재하는 문서가 없음")

cc = pd.read_pickle("./data/Default/"+ folder_name + "/HAN_Filtered_" + folder_name+ ".pkl")
cc =cc[cc["filtered_ICT"]==1].reset_index(drop=True)

a = cc.iloc[33,:]["contents"].replace("\r","")
b = a.replace("\n","")
b.replace("tag","")

## 문서가 제대로 들어가있는지 확인하는 반복문
for j in range(5):
    z = 0
    for i in range(41):
        z +=len(doc_list[j][i])
    print(z)


# # 년도에 따른 토픽 별 텍스트 네트워크 

wd_network = pd.DataFrame(index=range(len(corpus)),columns=["corpus","mdl.docs","Topic","Topic_prob"])

for i in tqdm(range(len(corpus))):
    wd_network.iloc[i][0] =  ' '.join(map(str, corpus[i]))

for j in tqdm(range(len(doc_insts))):
    wd_network.iloc[j][1] = doc_insts[j]

for z in tqdm(range(len(doc_insts))):
    wd_network.iloc[z][2] = doc_insts[z].get_topics(top_n=mdl.k)[0][0] + 1

for c in tqdm(range(len(doc_insts))):
    wd_network.iloc[c][3] = doc_insts[c].get_topics(top_n=mdl.k)[0][1]

wd_network["year"] = df["year"].astype(int).copy()
if folder_name != "GOV":
    wd_network["year"] = wd_network["year"].replace(2022,2021)
else:
    pass

wd_network
wd_network["Topic"].nunique() #존재하지 않는 토픽이 많음을 확인
wd_network_lda = wd_network.copy() ##최초 1회만 실행

year_range = input("사용년도의 범위를 입력하시오 ex)2017~2021:5 :")
topic_len = input("토픽의 개수를 입력하시오:")


# # 보고서 쓰기위한 워드 데이터프레임 생성

# from collections import Counter

# df = pd.DataFrame()
# for c in range(2017,2022):
#     wd_network = wd_network_lda[wd_network_lda["year"]==c]
#     final = []
#     for j in range(topic_num):
#         temp= []
#         temp2 = []
#         using_word = np.array(mdl.get_topic_words(j,500))[:,0].tolist()
#         wd = wd_network[wd_network["Topic"]==j+1]
#         if len(wd)>0:
#             for i in range(len(wd)):
#                 temp.append(wd["corpus"].iloc[i].split(" "))
#             temp = sum(temp,[])
#             all_word =Counter(temp).most_common()
#             for i in (np.array(all_word)[:,0].tolist()):
#                 if i in using_word:
#                     temp2.append(i)
#                     if len(temp2)==10:
#                         break
#             final.append(temp2)
#         else:
#             final.append("")
#     df = pd.concat([df,pd.Series(final)],axis=1)

# df.columns=range(2017,2022)
# df.index = df.index+1

# df.to_excel("NKIS토픽별 주요 단어.xlsx",index=False)
# df


# ## 자동
for timepoint in tqdm(range(int(year_range))):

    if timepoint == 0:
        wd_network  = wd_network_lda[wd_network_lda["year"]==2017]
    elif timepoint == 1:
        wd_network  = wd_network_lda[wd_network_lda["year"]==2018]
    elif timepoint == 2:
        wd_network  = wd_network_lda[wd_network_lda["year"]==2019]
    elif timepoint == 3:
        wd_network  = wd_network_lda[wd_network_lda["year"]==2020]
    elif timepoint == 4:
        wd_network  = wd_network_lda[wd_network_lda["year"]==2021]
        
        
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

            plt.savefig("C:\\Users\\newcomer02\\NTIS_Project\\data\\Default\\NTIS\\LDA\\Topic_Network_NKIS\\Topic" + str(topic_num+1) +"_"+str(timepoint+2017)+".png", bbox_inches='tight')
        except:
            pass


# # 년도 구분하지 않은 텍스트 네트워크
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

        plt.savefig(".\\data\\Default\\"+folder_name+"\\LDA\\Topic_Network_"+folder_name+"\\Topic" + str(topic_num+1)+".png", bbox_inches='tight')
    except:
        pass

print("** degree **")
for x in range(len(G)):
    print(sorted_dgr[x])

print("** betweenness **")
for x in range(len(G)):
    print(sorted_btw[x])

print("** closeness **")
for x in range(len(G)):
    print(sorted_cls[x])


# ## 깨진 파일 1개 자동

wd_network_lda = wd_network.copy() ## 최초 한번만 실행 그 후로 실행x
topic_num=int(input("분석하고 싶은 토픽 번호를 입력하십시오:"))-1
timepoint = int(input("보고싶은 년도를 입력해주시오: ex)2017:0 ~ 2021:4 :"))
if timepoint == 0:
    wd_network  = wd_network_lda[wd_network_lda["year"]==2017]
elif timepoint == 1:
    wd_network  = wd_network_lda[wd_network_lda["year"]==2018]
elif timepoint == 2:
    wd_network  = wd_network_lda[wd_network_lda["year"]==2019]
elif timepoint == 3:
    wd_network  = wd_network_lda[wd_network_lda["year"]==2020]
elif timepoint == 4:
    wd_network  = wd_network_lda[wd_network_lda["year"]==2021]

using_word= []
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
#plt.savefig(".\\data\\Default\\NTIS\\LDA\\Topic_Network_NKIS\\Topic"+ str(topic_num+1) +"_"+ str(timepoint+2017) +".png", bbox_inches='tight')
plt.show()


# ### var진행하기 위한 dist파일 쓰기 

# dist["Topic"] = topics_d
# dist = dist.set_index("Topic")
# dist.to_csv("dist_NKIS.csv")


# # VAR word_dist 데이터 추출 과정
# wd_network.to_csv("wd_network_NKIS_41.csv",index=False)

from collections import Counter
wd_network=input("워드네트워크를 만들 데이터를 입력하세요(Default: 엔터키): ")
if wd_network=='':
    print("\nDefault 워드네트워크 데이터를 불러옵니다.")
    wd_network= pd.read_csv("./data/Default/NTIS/LDA/wd_network_NKIS_"+str(topic_num) + ".csv")

else:
    print("\n%s 에 워드네트워크 데이터를 불러옵니다."%(wd_network))
    wd_network= pd.read_csv("./data/LDA/"+data_ver+"/" + wd_network + ".csv",encoding="cp949",index_col=0)
#    wd_network_41

word_dist = pd.DataFrame()
for i in range(topic_num):
    dtfrm = pd.DataFrame(mdl.get_topic_words(i,1000000),columns=["word","prob"])
    dtfrm["topic"] = i+1
    dtfrm = dtfrm.sort_values("prob",ascending=False)
    word_dist = pd.concat([word_dist,dtfrm])
word_dist = word_dist[["word","topic","prob"]]
word_dist

for c in tqdm(range(start_num,end_num+1)):
    final = pd.DataFrame()
    for i in range(topic_num):
        temp = []
        using_word = word_dist[word_dist["topic"]==i+1]["word"].tolist()
        zzz=wd_network[(wd_network["year"]==c) & (wd_network["Topic"]==i+1)]
        if len(zzz) > 0: 
            for j in range(len(zzz)):
                aa = zzz["corpus"].iloc[j].split(" ")
                for u in aa:
                    if u in using_word:
                        temp.append(u)
                    else:
                        continue
            df = pd.DataFrame(Counter(temp).most_common())
            df["topic"]= i+1
            df[c]= df[1]/sum(df[1])
            final = pd.concat([final,df])
            final.pop(1)
    if c == start_num:
        final_dist = final.copy()
    else: 
        final_dist = pd.merge(final_dist,final, on=["topic",0])
        

final_dist["labels"]= final_dist["topic"].apply(lambda x: "topic" + str(x)+": " + ','.join(pd.DataFrame(mdl.get_topic_words(x-1,5))[0].tolist()))
final_dist.columns = ["word","topic"] + list(range(start_num,end_num+1)) + ["labels"]
final_dist = final_dist[["word","labels","topic"]  + list(range(start_num,end_num+1))]


index_list = []
for i in range(topic_num):
    index_list = index_list + list(range(sum(final_dist["topic"]==(i+1))))
final_dist["index"] = index_list
final_dist=final_dist.set_index("index")
final_dist
final_dist.to_pickle("WORDS_NKIS_" + str(topic_num)+".pkl")
pd.read_pickle("WORDS_NKIS_"+str(topic_num)+ ".pkl")


# # GOV 문서 별 토픽확률 및 토픽 분포 병합

zz = pd.read_excel("./data/Default/GOV/GOV_2022.xlsx")
cc = pd.read_excel("./data/Default/GOV/HAN_Filtered_GOV.xlsx",index_col=0)
zz["contents"] = cc["contents"]
zz["content"] = cc["content"]
zz["ICT_prob"] = cc["ICT_prob"]
zz["filtered_ICT"] = cc["filtered_ICT"]
zz = zz.sort_values("ICT_prob",ascending=False)

a,b,c,d,e,f = [],[],[],[],[],[]
for i in range(len(zz[zz["filtered_ICT"]==1])):
    a.append(sum(doc_insts[i].get_topics(3),())[::2][0])
    b.append(sum(doc_insts[i].get_topics(3),())[::2][1])
    c.append(sum(doc_insts[i].get_topics(3),())[::2][2])
    d.append(sum(doc_insts[i].get_topics(3),())[1::2][0])
    e.append(sum(doc_insts[i].get_topics(3),())[1::2][1])
    f.append(sum(doc_insts[i].get_topics(3),())[1::2][2])

a = pd.DataFrame(a)
b = pd.DataFrame(b)
c = pd.DataFrame(c)
d = pd.DataFrame(d)
e = pd.DataFrame(e)
f = pd.DataFrame(f)

dd = pd.concat([a,b,c,d,e,f],axis=1)

dd.columns=["first_topic","second_topic","third_topic","first_prob","second_prob","third_prob"]
dd["prob_sum"] = dd.iloc[:,3:].sum(axis=1)

zz.reset_index(inplace=True)
df = pd.concat([zz[zz["filtered_ICT"]==1],dd],axis=1)
df.to_excel("GOV_Topic.xlsx")