#!/usr/bin/env python
# coding: utf-8

# Package Load
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
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from math import log2
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly
import plotly.graph_objs as go
import plotly.offline as pyo
from datetime import datetime
version_name=str(datetime.today().strftime("%Y%m%d")) # 버전 정보

# pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org statsmodels
os.chdir("C:\\Users\\newcomer02\\NTIS_Project")


# # 1. 토픽분포 예측

topic_num=int(input("토픽 개수를 입력하세요 : "))
start_year=int(input("시작 년도를 입력하세요 : "))
year_length = int(input("데이터가 몇개년으로 이루어져 있는지 입력하세요:"))

folder_name = input("./data/prediction 내 사용할 데이터 폴더명을 입력하세요 (Default:엔터키): ")
path='./data/Default/'+folder_name+'/prediction/'+folder_name+'_topic_dist_'+str(topic_num)+'.pkl'    
print("\n%s 데이터의 토픽 분포를 예측합니다."%(folder_name))

def kl_divergence(p, q):
    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

# 모델 학습 및 성능 평가
# 추후, 년도가 추가 된다면 2018년 뒤에 2019, 2020, ...이 붙을 수 있지만, 
# word, labels, topic, 2014, 2015, ... 의 컬럼 순서는 유지 되어야 함

def Real_Predict_Measure(path):
    DATA = pd.read_pickle(path)
    #del DATA['topic']
    
    Y_col = list(DATA[DATA.columns[len(DATA.columns)-1]])

    dist_refined_valid=DATA.iloc[:,0:len(DATA.columns)-1].values

    dist_refined_valid=dist_refined_valid.T.astype(float)

    var_model_valid = VAR(endog=dist_refined_valid)

    var_model_fit_valid = var_model_valid.fit(2)

    new_dist_valid = var_model_fit_valid.forecast(var_model_valid.y, steps=1)

    predict_last_year=new_dist_valid.tolist()[0]

    DATA['predict_prob'] = predict_last_year

    var_model_valid.y.tolist()

    new_dist_valid.tolist()[0]

    Y_test = list(new_dist_valid[0])
    
    mse = metrics.mean_squared_error(Y_col, Y_test)
    r2 = metrics.r2_score(Y_col, Y_test)
    kld = kl_divergence(Y_col, Y_test)
    
    return DATA, Y_col, Y_test, mse, r2, kld

DATA, Y_col, Y_test, mse, r2, kld = Real_Predict_Measure(path)
#DATA, Y_col, Y_test, mse, r2 = Real_Predict_Measure(path)
print("MSE : %f"%(mse))
print("\nR-Square : %f"%(r2))
print("\nKL-Divergence : %f"%(kld))

DATA.columns= [0,1,2,3,4,"predict_prob"]

def All_Plot1(DATA, path):
    
    plt.style.use('default')
    DIR = "C:\\Users\\newcomer02\\NTIS_Project\\data\\Default\\NTIS\\LDA\\VAR,LSTM\\NKIS\\topic_distribution\\"
    top_list = []
    for i in range(len(DATA)):
        top_list.append('topic '+str(i+1))

    top_pred_list = []
    for i in top_list:
        top_pred_list.append(i+' Pred')

    new_df = pd.DataFrame()
    for i in range(len(DATA)):
        new_df['topic '+str(i+1)] = DATA.iloc[i]

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

    index = np.arange(len(DATA))
    bar_width = 0.35
    plt.figure(figsize=(15,3))
    p1 = plt.bar(index, Y_test, bar_width)
    p2 = plt.bar(index+bar_width, Y_col, bar_width)
    plt.title('VAR '+str(start_year+len(DATA.columns)-2)+' Topic Distribution')
    plt.xlabel('Topic', fontsize=10)
    plt.ylabel('Probability', fontsize=10)
    plt.xticks(index, top_id, fontsize=10)
    plt.legend((p1[0], p2[0]), ('Prediction', 'Real'), fontsize=10)
    # plt.savefig(DIR+"/All_Topics_Verification")
    plt.show()
    
    top_pred = []
    for j in range(len(DATA)):
        asdf = []
        for i in range(len(DATA.columns)-2):
            asdf.append(topic_dist_norm[j][i])
        asdf.append(Y_test[j])
        top_pred.append(asdf)

    for i in range(len(DATA)):
        plt.title('Topic ' + str(i+1) + '_var_Distribution')
        plt.grid()
        labels = top_list

        plt.plot(year_list, top_pred[i], label = top_pred_list[i], linestyle = 'dotted')
        p2 = plt.scatter(year_list, top_pred[i])
        plt.plot(year_list, topic_dist_norm[i], label = top_list[i])
        p1 = plt.scatter(year_list, topic_dist_norm[i])

        plt.plot()
        plt.legend(loc='upper left')
        plt.xticks(np.arange(start_year,start_year+len(DATA.columns)-1),labels = [i for i in list(range(start_year,start_year+len(DATA.columns)-1))])
        plt.tick_params(axis='x', direction='in', length=3, pad=6, labelsize=14, top=True)
        #plt.savefig(DIR + 'topic_' + str(i+1)+'_var')
        plt.show()
        plt.close()

All_Plot1(DATA, path)

def New_Predict_Measure(path,step):
    DATA = pd.read_pickle(path)
    #del DATA['topic']

    dist_refined_valid=DATA.iloc[:,0:len(DATA.columns)].values
    dist_refined_valid=dist_refined_valid.T.astype(float)
    var_model_valid = VAR(endog=dist_refined_valid)
    var_model_fit_valid = var_model_valid.fit(2)
    len(var_model_valid.y.tolist())
    new_dist_valid = var_model_fit_valid.forecast(var_model_valid.y, steps=step)
    col = len(DATA.columns)
    for i in range(step) :
        predict_next_year=new_dist_valid.tolist()[i]
        DATA[col+i] = predict_next_year

    return DATA

DATA = New_Predict_Measure(path,2)

plt.style.use('seaborn')

def All_Plot2(DATA,step):
    #DIR = "C:\\Users\\newcomer02\\NTIS_Project\\data\\Default\\NTIS\\LDA\\VAR,LSTM\\NKIS\\topic_prediction\\"
    top_list = []
    for i in range(len(DATA)):
        top_list.append('topic '+str(i+1))

    top_pred_list = []
    for i in top_list:
        top_pred_list.append(i+' Pred')

    new_df = pd.DataFrame()
    for i in range(len(DATA)):
        new_df['topic '+str(i+1)] = DATA.iloc[i]
    
    year_list = []
    for i in range(len(DATA.columns)):
        year_list.append(start_year+i)

    yr_topic_norm = []
    for i in range(len(DATA.columns)):
        yr_topic_norm.append(list(DATA.iloc[:,i]))

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
        topic_yr.append(Y_test[j])
        top_pred.append(topic_yr)
    
    for i in range(len(DATA)):
        plt.title(folder_name+'_Topic ' + str(i+1) + '_var_Prediction')
        #plt.grid()
        labels = top_list

        plt.plot(year_list, top_pred[i], label = top_pred_list[i], linestyle = 'dotted')
        p2 = plt.scatter(year_list, top_pred[i])
        plt.plot(year_list[:-step], topic_dist_norm[i][:-step+1], label = top_list[i])
        p1 = plt.scatter(year_list[:-step], topic_dist_norm[i][:-step+1])

        plt.plot()
        plt.legend(loc='upper left')
        plt.xticks(np.arange(start_year,start_year+len(DATA.columns)),labels = [i for i in list(range(start_year,start_year+len(DATA.columns)))])
        plt.tick_params(axis='x', direction='in', length=3, pad=6, labelsize=14, top=True)
        #plt.savefig(DIR + 'topic_' + str(i+1) + '_var')
        plt.show()
        plt.close()

All_Plot2(DATA,2)

DATA_LABEL = []
for i in range(len(DATA)):
    DATA_LABEL.append('topic'+str(i+1))
    
DATA['LABEL']=DATA_LABEL
SORTED_DATA=DATA.sort_values(by=[len(DATA.columns)-2],axis=0,ascending=False)

# output type : ICT NTIS/NEWS 연도별 html 파일 (html/DTM 폴더)

def ntis_time(DATA,top_n):
    color=[ 'aqua', 'aquamarine','beige', 'bisque', 'black', 'blanchedalmond', 'blue',
            'blueviolet', 'brown', 'burlywood', 'cadetblue',
            'chartreuse', 'chocolate','coral', 'cornflowerblue',
            'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
            'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen','darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
            'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
            'darkslateblue', 'darkslategray', 'darkslategrey',
            'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
            'dimgray', 'dimgrey', 'dodgerblue', 'firebrick']
    plot=[]
    
    for i in range(top_n):
        plot.append(go.Scatter(x = np.array(range(2017,2017+len(DATA.columns)-4)), y = SORTED_DATA.iloc[i,:-2], line=dict(color=color[i],width=4),mode = 'lines+markers', name = SORTED_DATA.iloc[i,-1]))
        plot.append(go.Scatter(x = np.array(range(2017,2017+len(DATA.columns)-3)), y = SORTED_DATA.iloc[i,:-1], line=dict(color=color[i],dash='dashdot',width=4),mode = 'lines+markers', name = 'predicted_'+SORTED_DATA.iloc[i,-1]))
        

    
    layout = go.Layout(title='NTIS 토픽별 트렌드(VAR)',
                       legend=dict(x=0,y=-1.7),margin=dict(l=20, r=20, t=60, b=300),paper_bgcolor="White",
                       autosize=True,title_font_size=30,font=dict(size=15),hoverlabel=dict(
                                   
        font_size=16,
        font_family="Rockwell"
    ),
                    xaxis=dict({"tickvals":list(range(2014,2014+len(DATA.columns))),
                            "ticktext":[str(i) for i in list(range(2017,2017+len(DATA.columns)))],
                           "title":"연도"}),
                    yaxis=dict({"title":"토픽 비중"}))
    
    gen_ntis = go.Figure(data=plot, layout=layout)
    pyo.iplot(gen_ntis)
    
    # gen_ntis.write_html(DIR+'/'+"ntis_trace_상위"+str(top_n)+".html")

ntis_time(DATA,10)


# # 2. 단어 분포 예측

# input : pkl 파일, 없으면 에러 (data/PREDICTION 폴더)
data=pd.read_pickle('./data/Default/'+folder_name+'/prediction/WORDS_'+ folder_name +"_"+ str(topic_num)+'.pkl')
TOPN=int(input("예측에 사용할 상위 단어 갯수를 지정하세요 (권장값=1000) : "))
data=data[data.index<TOPN]

# ## - 존재하는 마지막 년도 예측

def var_mode(data):
    
    topic_list = list(range(topic_num))
    for y in tqdm(range(topic_num)):
        ex_da=data[data['topic']==y+1]
        if len(ex_da) > 1:
            valid_da=ex_da.iloc[:,3:-1].values
            valid_da=valid_da.T
            var_model_valid = VAR(endog=valid_da)
            var_model_fit_valid = var_model_valid.fit(1)
            new_dist_valid = var_model_fit_valid.forecast(var_model_fit_valid.endog, steps=1)
            name1='predict_var'+str(data.columns[-1])
            ex_da[name1]=new_dist_valid.tolist()[0]
            norm=[]
            for i in ex_da[name1]:
                if i <=0:
                    norm.append(0)
                else:
                    norm.append(i)
            ex_da[name1]=norm
            if y==topic_list[0]:
                final=ex_da
            else:
                final=pd.concat([final, ex_da])
        else:
            topic_list.pop(topic_list.index(y))
    return final[final["predict_var2021"]!=0]

#data 인풋 데이터, t 시퀀스, predict_year= 2018년도까지 데이터가 있다면,
# 2018년도를 예측하게 해서 해당 데이터의 var성능을 평가

first_df=var_mode(data)

def Topic_Predict_Measure(fir_df):
    for u in range(topic_num):
        df=fir_df[fir_df["topic"]==u+1]
        if len(df)>0:
            word_col = df.iloc[:,-2:-1]
            word_col = word_col[word_col.columns[0]].tolist()
            word_test = df.iloc[:,-1:]
            word_test = word_test[word_test.columns[0]].tolist()

            cond1 = list(filter(lambda x:word_test[x]==0, range(len(word_test))))
            cond2 = list(filter(lambda x: word_col[x]==0, range(len(word_col))))
            cond = cond1 +cond2
            if len(cond) != 0:
                j = 0
                for k in cond:
                    del word_test[k-j] 
                    del word_col[k-j] 
                    j +=1


            mse = metrics.mean_squared_error(word_col, word_test)
            r2 = metrics.r2_score(word_col, word_test)
            kld = kl_divergence(word_col, word_test)

            print("\nTopic"+ str(u+1))
            print("MSE : %.8f" %(mse))
            print("R-Square: %f"%(r2))
            print("kL-Divergence : %f"%(kld))

Topic_Predict_Measure(first_df)

# 2018년도와 predict_18년도의 값을 비교하여 해당 데이터에 VAR성능이 어느 정도로 잘 나오는지 확인
# 만약, Spearsman_corr 결과, 낮은 값들이 많다면 향후 년도 예측의 성능이 낮을 수 있음

def spearsman_corr(fi_da):
    topic_corr=[]
    to_data=[]
    year=fi_da.columns[-2]
    predict_year=fi_da.columns[-1]
    
    for i in tqdm(range(1,topic_num+1)):
        real=fi_da[fi_da['topic']==i][['word',year]]
        if len(real) > 0:
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
    to['corr'].plot(kind='bar')
    #to['corr'].sort_values(ascending=False).plot(kind='bar')
    
    plt.title('TOPIC')
    plt.ylabel('spearsman_corr')
    #plt.savefig(DIR+'/'+data_type+'_topic_' + str(i+1)+'_spearman correlaion')
    return to_data,to

# 랭킹 변동 확인
# 스피어만 상관계수의 범위는 -1 ~ 1 까지로, 비교적 높은 결과가 나옴 -> VAR로 향후 년도를 예측하는게 의미가 있다고 판단 

da,to=spearsman_corr(first_df)
np.mean(to)


# # 토픽 별 단어 미래 예측

step = input("몇년 후 까지 예측하고 싶은지 입력하시오: ")
def var_mode_next_year(data,step):
    from statsmodels.tsa.vector_ar.var_model import VAR
    
    topic_list = list(range(topic_num))
    for y in tqdm(range(topic_num)):
        ex_da=data[data['topic']==y+1]
        if len(ex_da)>1: 
            valid_da=ex_da.iloc[:,3:].values
            valid_da=valid_da.T
            var_model_valid = VAR(endog=valid_da)
            var_model_fit_valid = var_model_valid.fit(1)
            new_dist_valid = var_model_fit_valid.forecast(var_model_fit_valid.endog, steps=step)

            for i in range(step):
                next_year='predict_var'+str(int(data.columns[-1])+(1+int(i)))
                ex_da[next_year]=new_dist_valid.tolist()[i]
                norm=[]
                for k in ex_da[next_year]:
                    if k <=0:
                        norm.append(0)
                    else:
                        norm.append(k)
                ex_da[next_year]=norm

            if y==topic_list[0]:
                final=ex_da
            else:
                final=pd.concat([final, ex_da])
        else:
            topic_list.pop(topic_list.index(y))
    return final

final_df=var_mode_next_year(data,int(step))


# # Topic별 단어 미래 예측 시각화
# 확인하고자 하는 topic 번호 및 상위 몇 개의 단어를 보고싶은지 설정하여 시각화

import plotly
import plotly.graph_objs as go

def topic_word_plot2(data,topic,top_N,step): 
    
    DIR = "C:\\Users\\newcomer02\\NTIS_Project\\data\\Default\\NTIS\\LDA\\VAR,LSTM\\NKIS\\topic_word_prediction_html\\"
    step = int(step)

    obj_col = data.columns.tolist()[:3].copy()
    num_col = []
    for i in range(start_year,(start_year+year_length+step)):
        num_col.append(i)
    data.columns = obj_col + num_col
    
    df = data[data["topic"]==topic].iloc[:top_N,:]
    if len(df) > 0:
        df = df.set_index("word")
        df = df.iloc[:,2:].T

        ye = df.index.tolist()

        col = list(df.columns)
        color=['brown','red','darkviolet','deeppink','forestgreen', 'fuchsia','indigo','lawngreen', 'lightslategray','yellow','silver','skyblue','tomato', 'turquoise','yellowgreen','black','chocolate','darkgoldenrod']

        plot=[]

        for N in range(min(top_N,len(col))):
            #기존의 데이터
            plot.append(go.Scatter(y=list(df[col[N]].iloc[:-step]), x=ye[:-step],name=col[N],line=dict(color=color[N],width=2),marker = dict(symbol="cross",color=color[N]),mode='lines+markers'))
            plot.append(go.Scatter(y=list(df[col[N]].iloc[-step-1:]), x=ye[-step-1:],name="예측"+str(col[N]),line=dict(color=color[N],dash="dashdot",width=2),marker = dict(symbol="cross",color=color[N]),mode='lines+markers'))

        layout = go.Layout(title=data[data["topic"]== topic]["labels"][0],autosize=True,
                           xaxis=dict(
                                tickvals=ye,
                               title="연도"),
                           yaxis=dict(title = "Topic내 단어 순위",linewidth=4))

        fig=go.Figure(data=plot,layout=layout)

        plotly.offline.iplot({
            "data": plot,
            "layout": go.Layout(autosize=True,height=500,width=700,title=data[data["topic"]== topic]["labels"][0],legend=dict(font=dict(size=15)),margin=dict(l=20, r=20,t=100),xaxis = dict(title = "연도",linewidth=0.5,
                                tickvals=ye
        ),yaxis=dict(title = "Topic내 단어 순위",linewidth=2))})#,auto_open=True,filename=DIR+"topic_"+str(topic)+"_var.html",image="png",image_filename="topic_" + str(topic) + "_var")

for i in range(topic_num):
    topic_word_plot2(final_df,i+1,5,step)

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
            #number_of_topics=len(finalss.columns)

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
    ))}, auto_open=False)
        
    else:
        print('해당 단어는 데이터에 존재하지 않습니다.')

target_word=input("분석하고자 하는 단어를 입력하세요 : ")
max_length = 2
try:
    topic_di(data=final_df,word=target_word,number_of_topics=max_length)
except IndexError:
    max_length -=1
    topic_di(data=final_df,word=target_word,number_of_topics=max_length)

#final_df.to_excel("C:\\Users\\KISDI\\연세대_FINAL\\html\\VAR\\KISAU_20210808\\var_43_ntis.xlsx")
#final_df
