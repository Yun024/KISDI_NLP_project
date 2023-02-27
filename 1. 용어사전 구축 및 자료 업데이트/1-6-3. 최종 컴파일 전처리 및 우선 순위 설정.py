#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from eunjeon import Mecab
import numpy as np
from tqdm import tqdm

mecab = Mecab(dicpath="c:/mecab/mecab-ko-dic")
mecab.morphs("자동체결장치는 세포핵 단백질 사이의 기능적 연결을 개발자가 방문자가 효과를 암웹 직생하다")

data_use=pd.read_csv("C:/mecab/user-dic/custom.csv")
data_use.iloc[208]


data_use["term"]=data_use["term"].str.replace(pat=r'[\n]',repl=r'',regex=True)
data_use["term"]=data_use["term"].str.replace(pat=r'[\r]',repl=r'',regex=True)

data_use["term_1"]=data_use["term_1"].str.replace(pat=r'[\n]',repl=r'',regex=True)
data_use["term_1"]=data_use["term_1"].str.replace(pat=r'[\r]',repl=r'',regex=True)

data_use.to_csv("C:/mecab/user-dic/custom.csv",index=False)


# ## window powershell 관리자 모드로 실행 (아래 순서대로 복붙가능)
# 
# ### 1. cd C:\mecab
# ### 2. tools\add-userdic-win.ps1
# ### 3. 에러가 발생할 경우 Set-ExecutionPolicy Unrestricted 입력 후 y 입력
# ### 4. 다시올라가서 2. 내용 입력

data=pd.read_csv("C:/mecab/mecab-ko-dic/user-custom.csv")
data["2953"]
data["2953"]= (((data["2953"]-data["2953"].min()) * (5000 - 0))/ (data["2953"].max() - data["2953"].min())) + 0
data=data.astype({'2953':'int'})
data

dic_len=len(data["term"])
for i in tqdm(range(0,dic_len)):
    data["2953"][i] = len(data["term"][i])
data["2953"]= data["2953"].rank(method="dense",ascending=False)
data=data.astype({'2953':'int'})
data.to_csv("C:/mecab/mecab-ko-dic/user-custom.csv",index=False)

# ## window powershell 관리자 모드로 실행 (아래 순서대로 복붙가능)
# 
# ### 1. cd C:\mecab
# ### 2. tools\compile-win.ps1

