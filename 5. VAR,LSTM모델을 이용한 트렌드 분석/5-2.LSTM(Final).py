#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import os
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mutual_info_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from math import log2
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from datetime import datetime
import plotly.offline as pyo

version_name=str(datetime.today().strftime("%Y%m%d"))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.chdir("C:\\Users\\newcomer02\\NTIS_Project")


# # 1. 토픽분포 예측
topic_num=int(input("토픽 개수를 입력하세요 : "))
start_year=int(input("시작 년도를 입력하세요 : "))

folder_name = input("./data/prediction 내 사용할 데이터 폴더명을 입력하세요 (Default:엔터키): ")
path='./data/Default/'+folder_name+'/prediction/'+folder_name+'_topic_dist_'+str(topic_num)+'.pkl'
print("\n%s 데이터의 토픽 분포를 예측합니다."%(folder_name))


def kl_divergence(p, q):
    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

def Real_Predict_Measure(path):
    DATA = pd.read_pickle(path)

    DATA = DATA.T
    
    test_X = []
    test_Y = []
    train_X = []
    train_Y = []

    for k in range(len(DATA)-4):
        for i in range(len(DATA.columns)):
            train_X.append(list(DATA.iloc[k:k+3,i]))
            train_Y.append(DATA.iloc[k+3,i])

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)

    for i in range(len(DATA.columns)):
        test_X.append(list(DATA.iloc[1:4,i]))
    test_X = np.array(test_X)

    for i in range(len(DATA.columns)):
        test_Y.append(DATA.iloc[4,i])
    test_Y = np.array(test_Y)
    
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[1],1))
    
    model = Sequential()
    model.add(LSTM(10, 
                   input_shape=(train_X.shape[1], train_X.shape[2]), 
                   activation='relu', 
                   return_sequences=False)
              )
    model.add(Dense(5))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(train_X, train_Y, epochs=50, batch_size=2)
    
    test_X = test_X.reshape((len(DATA.columns),3,1))
    yhat = model.predict(test_X)
    
    Y_col = []
    for i in yhat:
        for j in i:
            Y_col.append(j)
    
    Y_test = list(test_Y)
    
    mdl = model
    mse = mean_squared_error(Y_test, Y_col)
    r2 = r2_score(Y_test, Y_col)
    kld = kl_divergence(Y_test, Y_col)
    DATA = DATA.T
    DATA[str(len(DATA.columns))] = Y_col
    
    return DATA, Y_col, Y_test, mse, r2, kld, mdl

# import random as rn
# import tensorflow as tf
# seed_num = 10
# np.random.seed(seed_num)
# rn.seed(seed_num)
# tf.random.set_seed(seed_num)

DATA, Y_col, Y_test, mse, r2, kld, mdl = Real_Predict_Measure(path)

print("MSE : %f"%(mse))
print("\nR-Square : %f"%(r2))
print("\nKL-Divergence : %f"%(kld))

DATA.columns= [0,1,2,3,4,"predict_prob"]

plt.style.use('seaborn')

def All_Plot1(DATA, Y_col, Y_test):
    top_list = []
    for i in range(len(DATA)):
        top_list.append('Topic '+str(i+1))

    top_pred_list = []
    for i in top_list:
        top_pred_list.append(i+' Pred')

    new_df = pd.DataFrame()
    for i in range(len(DATA)):
        new_df['Topic '+str(i+1)] = DATA.iloc[i]

    year_list = []
    for i in range(len(DATA.columns)-1):
        year_list.append(start_year+i)

    yr_topic_norm = []
    for i in range(len(DATA.columns)-1):
        yr_topic_norm.append(list(DATA[i]))

    topic_dist_norm = []
    for j in range(len(DATA)):
        topic_yr = []
        for i in range(len(DATA.columns)-1):
            topic_yr.append(yr_topic_norm[i][j])
        topic_dist_norm.append(topic_yr)

    top_id = []
    for i in range(len(DATA)):
        top_id.append(i+1)

    index = np.arange(len(DATA))
    bar_width = 0.35
    plt.figure(figsize=(15,3))
    p1 = plt.bar(index, Y_test, bar_width)
    p2 = plt.bar(index+bar_width, Y_col, bar_width)
    plt.title('LSTM '+str(start_year+len(DATA.columns)-2)+' Topic Distribution') 
    plt.xlabel('Topic', fontsize=10)
    plt.ylabel('Probability', fontsize=10)
    plt.xticks(index, top_id, fontsize=10)
    plt.legend((p1[0], p2[0]), ('Prediction', 'Real'), fontsize=10)
    #plt.savefig(DIR+'/All_Topics_Verification')
    plt.show()

    top_pred = []
    for j in range(len(DATA)):
        topic_yr = []
        for i in range(len(DATA.columns)-2):
            topic_yr.append(topic_dist_norm[j][i])
        topic_yr.append(Y_col[j])
        top_pred.append(topic_yr)
        
    for i in range(len(DATA)):
        plt.title(folder_name+'_Topic ' + str(i+1) + ' Distribution')
        plt.grid()
        labels = top_list

        plt.plot(year_list, topic_dist_norm[i], label = top_pred_list[i], linestyle = 'dotted')
        p1 = plt.scatter(year_list, topic_dist_norm[i])
        plt.plot(year_list, top_pred[i], label = top_list[i])
        p2 = plt.scatter(year_list, top_pred[i])

        plt.plot()
        plt.legend(loc='upper left')
        plt.xticks(np.arange(start_year,start_year+len(DATA.columns)-1),labels = [i for i in list(range(start_year,start_year+len(DATA.columns)-1))])
        plt.tick_params(axis='x', direction='in', length=3, pad=6, labelsize=14, top=True)
        #plt.savefig(DIR+'/'+version+'_topic_' + str(i+1)+'_verification')
        plt.show()
        plt.close()

All_Plot1(DATA, Y_col, Y_test)

def New_Predict_Measure(path):
    DATA = pd.read_pickle(path)

    DATA = DATA.T
    
    test_X = []
    test_Y = []
    train_X = []
    train_Y = []
    
    for k in range(len(DATA)-3):
        for i in range(len(DATA.columns)):
            train_X.append(list(DATA.iloc[k:k+3,i]))
            train_Y.append(DATA.iloc[k+3,i])
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)

    for i in range(len(DATA.columns)):
        test_X.append(list(DATA.iloc[len(DATA)-3:len(DATA),i]))
    test_X = np.array(test_X)
    
    print(train_X.shape)
    print(train_Y.shape)
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[1],1))
    
    model = Sequential()
    model.add(LSTM(10, 
                   input_shape=(train_X.shape[1], train_X.shape[2]), 
                   activation='relu', 
                   return_sequences=False)
              )
    model.add(Dense(5))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(train_X, train_Y, epochs=50, batch_size=5)
    
    test_X= test_X.reshape((len(DATA.columns),3,1))
    yhat = model.predict(test_X)
    
    Y_col = []
    for i in yhat:
        for j in i:
            Y_col.append(j)
    
    DATA = DATA.T
#    DATA[len(DATA.columns)] = Y_col
    mdl = model
    
    return DATA, mdl

DATA, mdl = New_Predict_Measure(path)

def Make_Predict_Data(DATA,step):
    temp_data = DATA
    temp_col = len(DATA.columns)
    for j in range(step) :
        temp_x =[]
        for i in range(len(DATA)) :
            temp_x.append(list(temp_data.iloc[i,-3:]))
        temp_x = np.array(temp_x).reshape((topic_num,3,1))
        temp_y = mdl.predict(np.array(temp_x))
        temp_data[temp_col+j] = temp_y
    return temp_data

pred_data = Make_Predict_Data(DATA,2)
pred_data.columns= [0,1,2,3,4,5,6]

plt.style.use('seaborn')

def All_Plot2(DATA,step):
    DIR = "C:\\Users\\KISDI\\LDA\\html\\LSTM\\"+folder_name+"\\"
    top_list = []
    for i in range(len(DATA)):
        top_list.append('Topic '+str(i+1))

    top_pred_list = []
    for i in top_list:
        top_pred_list.append(i+' Pred')

    new_df = pd.DataFrame()
    for i in range(len(DATA)):
        new_df['Topic '+str(i+1)] = DATA.iloc[i]
    
    year_list = []
    for i in range(len(DATA.columns)):
        year_list.append(start_year+i)

    yr_topic_norm = []
    for i in range(len(DATA.columns)):
        yr_topic_norm.append(list(DATA[i]))

    topic_dist_norm = []
    for j in range(len(DATA)):
        topic_yr = []
        for i in range(len(DATA.columns)-1):
            topic_yr.append(yr_topic_norm[i][j])

        topic_dist_norm.append(topic_yr)

    topic_sum = []
    for i in topic_dist_norm:
        c = 0
        c+=sum(i)
        topic_sum.append(c)

    topic_sort = sorted(topic_sum, reverse=True)    

    idx_sort = []
    for i in topic_sort:
        idx_sort.append(topic_sum.index(i))

    top_id = []
    for i in range(len(DATA)):
        top_id.append(i+1)

    top_pred = []
    for j in range(len(DATA)):
        topic_yr = []
        for i in range(len(DATA.columns)-1):
            topic_yr.append(topic_dist_norm[j][i])
        topic_yr.append(Y_col[j])
        top_pred.append(topic_yr)
    
    for i in range(len(DATA)):
        plt.title(folder_name+'_Topic ' + str(i+1) + ' Prediction')
        plt.grid()
        labels = top_list

        plt.plot(year_list, top_pred[i], label = top_pred_list[i], linestyle = 'dotted')
        p2 = plt.scatter(year_list, top_pred[i])
        plt.plot(year_list[:-step], topic_dist_norm[i][:-step+1], label = top_list[i])
        p1 = plt.scatter(year_list[:-step], topic_dist_norm[i][:-step+1])

        plt.plot()
        plt.legend(loc='upper left')
        plt.xticks(np.arange(start_year,start_year+len(DATA.columns)),labels = [i for i in list(range(start_year,start_year+len(DATA.columns)))])
        #plt.savefig(DIR + 'topic_' + str(i+1)+'_lstm')
        plt.show()
        plt.close()

All_Plot2(pred_data,2)

DATA_LABEL = []
for i in range(len(DATA)):
    DATA_LABEL.append('topic'+str(i+1))
    
DATA['LABEL']=DATA_LABEL
SORTED_DATA=DATA.sort_values(by=[len(DATA.columns)-2],axis=0,ascending=False)

# output type : ICT NTIS/NEWS 연도별 html 파일 (html/DTM 폴더)

def ntis_time(DATA,top_n):
    color=['brown','red','darkviolet','deeppink','forestgreen', 'fuchsia','indigo','lawngreen', 'lightslategray','yellow','silver','skyblue','tomato'', turquoise','yellowgreen']
    plot=[]
    
    for i in range(top_n):
        plot.append(go.Scatter(x = np.array(range(2014,2014+len(DATA.columns)-1)), y = SORTED_DATA.iloc[i,:-2], line=dict(color=color[i],width=4),mode = 'lines+markers', name = SORTED_DATA.iloc[i,-1]))
        plot.append(go.Scatter(x = np.array(range(2014,2014+len(DATA.columns))), y = SORTED_DATA.iloc[i,:-1], line=dict(color=color[i],dash='dashdot',width=4),mode = 'lines+markers', name = 'predicted_'+SORTED_DATA.iloc[i,-1]))
        
    layout = go.Layout(title='NTIS 토픽별 트렌드(LSTM)',
                       legend=dict(x=0,y=-1.7),margin=dict(l=20, r=20, t=60, b=300),paper_bgcolor="White",
                       autosize=True,title_font_size=30,font=dict(size=15),hoverlabel=dict(
                                   
        font_size=16,
        font_family="Rockwell"
    ),
                    xaxis=dict({"tickvals":list(range(2014,2014+len(DATA.columns))),
                            "ticktext":[str(i) for i in list(range(2014,2014+len(DATA.columns)))],
                           "title":"연도"}),
                    yaxis=dict({"title":"토픽 비중"}),
                    height=1000)
    
    gen_ntis = go.Figure(data=plot, layout=layout)
    pyo.iplot(gen_ntis)
    
    gen_ntis.write_html(DIR+'/'+"ntis_trace_상위"+str(top_n)+".html")

ntis_time(DATA,5)

# # 단어분포 예측

if version=='':
    data=pd.read_pickle('./data/Default/NTIS/LDA/VAR,LSTM/WORDS_'+str(doc_type)+"_"+str(topic_num)+'.pkl')
else:
    data=pd.read_pickle('./data/PREDICTION/'+version+'/WORDS_'+str(topic_num)+'_t.pkl')
    
TOPN=int(input("예측에 사용할 상위 단어 갯수를 지정하세요 (권장값=1000) : "))
data=data[data.index<TOPN]


# # 마지막년도와 1년후 예측
def lstm_mode(data):
    
    for y in range(topic_num):
        ex_da=data[data['topic']==y+1]
        train_X=ex_da.iloc[:,3:-2].values
        train_Y=ex_da.iloc[:,-2:-1]
        test_X=ex_da.iloc[:,4:-1].values
        test_Y=ex_da.iloc[:,-1:]

        train_X=train_X.reshape(train_X.shape[0], train_X.shape[1],1)
        test_X=test_X.reshape(test_X.shape[0], test_X.shape[1],1)
        
        model = Sequential()
        model.add(LSTM(10, 
                       input_shape=(train_X.shape[1], train_X.shape[2]), 
                       activation='relu', 
                       return_sequences=False)
                  )
        model.add(Dense(5))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')
        history = model.fit(train_X, train_Y, epochs=30, batch_size=5)
        yhat = model.predict(test_X)
        
        name1='predict_lstm'+str(data.columns[-1])
        ex_da[name1]=yhat
        norm=[]
        for i in ex_da[name1]:
            if i <=0:
                norm.append(0)
            else:
                norm.append(i)
        ex_da[name1]=norm
        
        ###########################################
        
        train_X=ex_da.iloc[:,3:-2].values
        train_Y=ex_da.iloc[:,-2:-1]
        test_X=ex_da.iloc[:,4:-1].values
        test_Y=ex_da.iloc[:,-1:]

        train_X=train_X.reshape(train_X.shape[0], train_X.shape[1],1)
        test_X=test_X.reshape(test_X.shape[0], test_X.shape[1],1)
        
        model = Sequential()
        model.add(LSTM(10, 
                       input_shape=(train_X.shape[1], train_X.shape[2]), 
                       activation='relu', 
                       return_sequences=False)
                  )
        model.add(Dense(5))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')
        history = model.fit(train_X[1], train_Y[1], epochs=30, batch_size=5)
        yhat = model.predict(test_X)
        
        name2='predict_lstm'+str(int(data.columns[-1])+1)
        
        ex_da[name2]=yhat
        norm=[]
        for i in ex_da[name2]:
            if i <=0:
                norm.append(0)
            else:
                norm.append(i)
        ex_da[name2]=norm
        
        #############################################

        if y==0:
            final=ex_da
        else:
            final=pd.concat([final, ex_da])
    return final

#final_data=lstm_mode(data)
#final_data.to_csv("final_data_1.csv",index=False)
final_data = pd.read_csv("final_data_1.csv")


# # step을 이용한 미래예측 

step = int(input("몇년을 예측하고 싶은가요:"))

def lstm_step_mode(data,step,train):
    
    for t in range(topic_num):

        df_X = []
        df_Y = []
        ex_da=data[data['topic']==(t+1)]
        for i in range(train):
            df_X.append(ex_da.iloc[:,(3+i):(6+i)])
            df_Y.append(ex_da.iloc[:,6+i:7+i])

        for i in range(train):
            df_X[i] = df_X[i].values.reshape(df_X[i].shape[0],df_X[i].shape[1],1)

        model = Sequential()
        model.add(LSTM(10,
                      input_shape=(df_X[0].shape[1],df_X[0].shape[2]),
                      activation="relu",
                      return_sequences=False)
                 )
        model.add(Dense(5))
        model.add(Dense(1))

        model.compile(optimizer="adam",loss="mse")
        for i in range(train):
            model.fit(df_X[i],df_Y[i],epochs=30,batch_size=5)


        temp_col = ex_da.columns.tolist()

        for i in range(step):
            test_X = ex_da.iloc[:,-3:].values.reshape(df_X[0].shape[0],df_X[0].shape[1],1)
            yhat = model.predict(test_X)
            ex_da[temp_col[-1]+(i+1)] = yhat

        if t==0:
            final = ex_da
        else:
            final = pd.concat([final,ex_da])
    return final

# final_data = lstm_step_mode(data,step,len(data.columns)-6)
### 2017~2021 기준 6은 2회학습 , 7은 1회학습
### 년도가 늘어나면 학습횟수가 1회가 증가함

# final_data.to_csv("final_data_2.csv",index=False)
final_data = pd.read_csv("final_data_2.csv")
final_data

# 2018년도와 predict_18년도의 값을 비교하여 해당 데이터에 VAR성능이 어느 정도로 잘 나오는지 확인
# 만약, Spearsman_corr 결과, 낮은 값들이 많다면 향후 년도 예측의 성능이 낮을 수 있음

def spearsman_corr(fi_da):
    topic_corr=[]
    to_data=[]
    year=fi_da.columns[-2]
    predict_year=fi_da.columns[-1]
    
    for i in tqdm(range(1,len(fi_da['topic'].unique())+1)):
        real=fi_da[fi_da['topic']==i][['word',year]]
        real=real.sort_values(by=[year],ascending=False)
        real['Number']=list(range(len(real)))
        predict=fi_da[fi_da['topic']==i][['word',predict_year]]
        predict=predict.sort_values(by=[predict_year],ascending=False)
        predict['Number']=list(range(len(predict)))
        predict.columns=['predict_word',predict_year,'predict_Number']
        predict
        for su in range(len(real)):
            wo=list(real['word'])
            data=predict[predict['predict_word']==wo[su]]
            if su==0:
                fo=data
            else:
                fo=pd.concat([fo,data])
        fina=pd.concat([real,fo],axis=1)
        to_data.append(fina)
        topic_corr.append(fina[['Number','predict_Number']].corr(method='spearman').iloc[0,1])
        
    plt.figure(figsize=(25, 10))
    to=pd.DataFrame(columns=['corr'])
    
    to['corr']=topic_corr
    to.index=list(range(1,len(fi_da['topic'].unique())+1))
    to['corr'].sort_values(ascending=False).plot(kind='bar')
    
    plt.title('TOPIC')
    plt.ylabel('spearsman_corr')
    #plt.savefig(DIR+'/'+data_type+'_topic_' + str(i+1)+'_spearman correlaion')
    return to_data,to

# 랭킹 변동 확인
# 스피어만 상관계수의 범위는 -1 ~ 1 까지로, 비교적 높은 결과가 나옴 -> VAR로 향후 년도를 예측하는게 의미가 있다고 판단 

da,to=spearsman_corr(final_data.iloc[:,:-1])
np.mean(to)


# # 토픽 별 단어 미래 예측
def topic_word_plot2(data,topic,top_N,step):  

    DIR = "C:\\Users\\newcomer02\\NTIS_Project\\data\\Default\\NTIS\\LDA\\VAR,LSTM\\html\\"+doc_type + "\\topic_word_prediction_html\\"
    data = data[data["topic"] == topic]
    data = data.set_index("word")
    real_data = data.iloc[:,2:-(step)].T
    predict_data = data.iloc[:,-(step)-1:].T
    ye = real_data.index.append(predict_data.index).tolist()

    col= real_data.columns.tolist()
    color=['brown','red','darkviolet','deeppink','forestgreen', 'fuchsia','indigo','lawngreen', 'lightslategray','yellow','silver','skyblue','tomato'', turquoise','yellowgreen']
    plot=[]
    
    for N in range(top_N):
        #기존의 데이터
        plot.append(go.Scatter(y=list(real_data[col[N]]), x=real_data.index.tolist(),name=col[N],line=dict(color=color[N], width=2),marker = dict(color=color[N]),mode='lines+markers'))
        #예측데이터
        plot.append(go.Scatter(y=list(predict_data[col[N]]), x=predict_data.index.tolist(),name="예측"+str(col[N]),line=dict(color=color[N],dash="dashdot", width=2),marker = dict(color=color[N]),mode='lines+markers'))
    
    layout = go.Layout(title=data["labels"][0],autosize=True,
                       xaxis=dict(
                            tickvals=ye,
                           title="연도"),
                       yaxis=dict(title = "Topic내 단어 순위",linewidth=2))

    fig=go.Figure(data=plot,layout=layout)
    
    
    plotly.offline.iplot({
            "data": plot,
            "layout": go.Layout(autosize=True,height=500,width=700,title=data["labels"][0],legend=dict(font=dict(size=15)),margin=dict(l=20, r=20,t=100),xaxis = dict(title = "연도",linewidth=0.5,
                                tickvals=ye
),yaxis=dict(title = "Topic내 단어 순위",linewidth=2))})#,auto_open=True,filename= DIR +"topic_" + str(topic) +"_lstm.html" ,image="png",image_filename = "topic_" + str(topic) +"_lstm")

for i in range(topic_num):
    topic_word_plot2(data=final_data,topic=i+1,top_N=5,step=step)
    
final_data[final_data["topic"]==9]

def topic_di(data,word,number_of_topics):
   
    length=len(data['topic'].unique())
    ds=[]
    fib=[]
    ra=[]
    
    name=list(data.labels.unique())
    for rank in tqdm(range(1,length+1)):
        word_da=data[data['topic']==rank].iloc[0:,3:]     
        
        ye=list(word_da.columns[0:-2])
        ye.append(str(int(word_da.columns[-3])+1))
        
        for i in range(len(word_da.columns)):
            word_da.iloc[:,i:i+1]=word_da.iloc[:,i:i+1].rank(ascending=False) 
        word_da['word']=data[data['topic']==rank].iloc[0:,0:1]     

        ran=list(word_da.iloc[:,-3:-2].sum(axis=1))
        word_da['rank']=ran
        word_da['rank']=word_da['rank'].rank(ascending=True) 
        word_da.sort_values(by=['rank'],axis=0,inplace=True)
        word_da=word_da.reset_index().iloc[:,1:-1]
        word_das=word_da.set_index('word').T
        
        ds.append(word_das)
   
    for nu,dd in enumerate(range(len(ds))):
        if word in ds[dd].columns:
            ra.append(nu)
            fib.append(ds[dd])
            
    if len(fib)!=0:
   
        for su,dat in enumerate(fib):
            if su==0:
                dat=pd.DataFrame(dat[word])
                dat.columns=[word+'--'+str(name[su-1])]
                finalss=dat
            else:
                dat=pd.DataFrame(dat[word])
                dat.columns=[word+'--'+str(name[su-1])]
                finalss=pd.concat([finalss,dat],axis=1)


        finalss=finalss.T.sort_values(by=data.columns[-5]).T
        
        print('찾을수있는 number_of_topics의 최대길이 :',len(finalss.columns))

        color=['fuchsia','brown','darkgray','darkviolet','deeppink','forestgreen','indigo','lawngreen', 'lightslategray','silver','skyblue','tomato'', turquoise','yellow','yellowgreen']

        plot=[]

        for N in range(number_of_topics):
            yy=[]
            i=0
            for ys in range(len(ye)-1):
                yy.append(i)
                i+=1    

            plot.append(go.Scatter(y=list(finalss.iloc[0:len(yy),N]), x=ye[0:-1],name=str(list(finalss.columns)[N]),line=dict(color=color[N], width=4),marker = dict(color=color[N]),mode='lines+markers'))

            fin=finalss.drop([data.columns[-3]])

            plot.append(go.Scatter(y=list(fin.iloc[:,N]), x=ye,name='예측_'+word,line=dict(color=color[N],dash='dashdot',width=4),marker = dict(color=color[N]),mode ='lines+markers'))

        plotly.offline.iplot({
            "data": plot,
            "layout": go.Layout(title=word, xaxis = dict(title = "년도",linewidth=0.5),

        yaxis=dict(autorange='reversed',title = "Topic별 단어 순위",linewidth=2),legend=dict(
        yanchor="top",
        y=-0.5,
        xanchor="left",
        x=0
    ))})

        plotly.offline.init_notebook_mode(connected=False)

        plotly.offline.plot({
            "data": plot,
            "layout": go.Layout(title=word,xaxis = dict(title = "년도",linewidth=0.5),

        yaxis=dict(autorange='reversed',title = "Topic내 해당 단어 순위",linewidth=2),legend=dict(
        yanchor="top",
        y=-0.5,
        xanchor="left",
        x=0
    ))}, auto_open=False )#,filename=DIR+'/'+word)

    else:
        print('해당 단어는 데이터에 존재하지 않습니다.')

target_word=input("분석하고자 하는 단어를 입력하세요 : ")
max_length = 5
try:
    topic_di(data=final_data,word=target_word,number_of_topics=max_length)
except IndexError:
    max_length -=1
    topic_di(data=final_data,word=target_word,number_of_topics=max_length)
