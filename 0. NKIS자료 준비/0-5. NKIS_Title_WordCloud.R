### 변수 초기화
rm(list=ls())

### 작업 디렉토리 변경
getwd()
setwd("C:/Users/newcomer02/Desktop/working/R/정제데이터")
dir()

###### 라이브러리 설치 및 불러오기
#install_github("junhewk/RcppMeCab")
library(wordcloud)
library(RColorBrewer)
library(rvest)
library(stringr)
library(dplyr)
library(KoNLP)
library(remotes)
library(RcppMeCab)

### 세종사전 설치
#useSejongDic()


###### 데이터 불러온 후 전처리
### 불러오기
data <- read.table(choose.files(),fill=T)  %>% unlist() %>% na.omit()
data.use <- Filter(function(x){nchar(x)>=2},data) 
data <- read.table("nkis_final.txt")

### 데이터 확인
data.title <- data$보고서명
data.title %>% class()
data.abstr <- data$국문초록 %>% unlist()
data.abstr %>% head() 

### 전처리
data.use <- Filter(function(x){nchar(x)>=2},data)
pword <- sapply(data.title,extractNoun,USE.NAMES = F)

###필터링
data <- unlist(pword)
#write(data,"C:/Users/newcomer02/Desktop/working/R/원본데이터/title.txt")
data <- Filter(function(x){nchar(x)>=2},data)
data <- gsub("\\d+","",data)
data <- gsub("\\n","",data)
data <- gsub("\\.","",data)
data <- gsub("\n","",data)
data <- gsub(" ","",data)
data <- gsub("-","",data)
# data <- gsub("of","",data)
# data <- gsub("and","",data)
# data <- gsub("the","",data)

 
###### 워드 클라우드
data_cnt <- table(data)
data_cnt <-sort(data_cnt, decreasing=T) %>% head(600)
#write.csv(data_cnt,"title_table.csv")
#write.table(data_cnt,"title_table.txt")

palete <- brewer.pal(9, "Set1")
x11()
wordcloud(names(data_cnt),freq=data_cnt,scale=c(10,0.5),rot.per=0.25,min.freq=3,
          random.order=F,random.color=T,colors=palete)
### brewer의 모든 색깔을 보여주는 함수 
display.brewer.all()

memory.limit(size=6143)
memory.size(max=T) = 6143
memory.limit(size=NA)

