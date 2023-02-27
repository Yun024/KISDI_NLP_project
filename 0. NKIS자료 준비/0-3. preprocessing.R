rm(list=ls())
getwd()
setwd("C:/Users/newcomer02/Desktop/working/R/원본데이터")
dir()

###### 라이브러리 설치 및 불러오기
library(rvest)
library(stringr)
library(dplyr)

data <- read.csv("NKIS_2017_2022.csv")
data.use <- data
data.use %>% head()
data.use %>% names()


##### NA값 제거
# data[data %>% is.na]                ## NA확인
# data[data$보고서유형 %>% is.na(),]  ## NA확인
data.use <- data.use %>% na.omit()


##### 엑셀의 셀당 글자 수 제한(32767)을 넘겨 다음행으로 넘어가는 데이터의 전처리 
### 다음 행으로 넘어간 URL을 찾고 정해진 위치에 넣어주는 반복문
limit<- data.use[,1] %>% as.numeric() %>% is.na() %>% which()

for(i in limit){
  j <- 1
  b <- ""
  while(substr(b,1,5) != "https"){
    b <- data.use[i,j]
    j <- j +1
  }
  data.use[i-1,16]<- data.use[i,j-1]
}

### 잘 들어갔는지 확인 
# data[3037,16]
# data[4192,16]
# data[4461,16]
# data[6735,16]
# data.use[3037,16]
# data.use[4192,16]
# data.use[4461,16]
# data.use[6735,16]


##### 불필요하게 늘어난 열 제거 
data.use <- data.use[,1:16]
data.use %>% head()


### paste문을 쓰는 방법도 있지만 데이터 손실이 일어나기 때문에 불러오는 방법 채택
# abstract<- vector()
# for (i in 1:133){
#   abstract <- paste0(abstract,b[i])
# }
# abstract

### 분리 혹은 특수기호로 인해 불러지지 않은 국문초록 URL을 통해 불러오기
#특수기호 94건, 분리 4건
data.abstr<- data.use %>% filter(국문초록=="#NAME?")
for (i in limit){
  data.abstr<- rbind(data.abstr,data.use[i-1,])
}

data.abstr$국문초록 %>% tail(4)
data.abstr %>% names()
data.abstr$Url주소

zz<-data.abstr %>% nrow()
for (k in 1:zz){
  html_news <- read_html(data.abstr[k,16],encoding="cp-949")
  data.abstr[k,15] <- (html_news %>% html_nodes(".excerpt") %>% html_text(trim=T))[1]
}
data.abstr[,15] %>% tail(4)
#write.table(data.abstr,"abstr.txt") ### TXT파일로 저장
#data.abstr<-read.table("abstr.txt")


##### 전처리 최종 데이터 통합 
### 반복문과 filter함수를 통해 이미 있는 행 제거
for (u in 1:zz){
  a <- (data.abstr[u,1])
  data.use<-data.use %>% filter(X != a)
}

### rbind를 이용한 통합
data.use <- rbind(data.use,data.abstr)
data.use <- data.use[data.use$X %>% as.numeric() %>% order(),]

### 국문초록이 분리되며 생긴 행 확인 후 제거 
data.use$X %>% tail(4)
row <- data.use %>% nrow()
data.use <- data.use[-(row-3):-row,]


### 인덱스 및 번호 재설정 
rownames(data.use) = NULL
data.use$X <-data.use %>% rownames()
data.use %>% head()
data.use %>% tail()

### 표준분류에 붙어있는 태그 제거하기 
data.use %>% names()
data.use[,13]
for(i in 1:(data.use %>% nrow())){
  data.use[i,13] <- gsub("\t","",data.use[i,13])
  data.use[i,13] <- gsub("\n","",data.use[i,13])
}

### 자료유형이 세미나인 데이터는 불필요하므로 제거
data.use <- data.use %>% filter(자료유형!="세미나")


### 출판년도가 2017년 이전인 데이터 제거 
data.use$출판년도 <- data.use$출판년도 %>% as.numeric()
data.use$출판년도 %>% table()
data.use <- data.use %>% filter(출판년도 >= 2017) 



### 표준분류를 대분류와 소분류로 나누고 변수순서 조정 
data.use$대분류 <- lapply(data.use$표준분류 %>% strsplit(">"),"[",1) %>% unlist()
data.use$소분류 <- lapply(data.use$표준분류 %>% strsplit(">"),"[",2) %>% unlist() %>% str_trim()
data.use <-data.use %>% relocate(c(대분류,소분류),.after=표준분류)
data.use %>% names()
data.use %>% head()


### 데이터 확인 후 이상한건 수동으로 제거
#write.table(data.use,"nkis_final.txt")
#zz <- read.table("nkis_final.txt")
zz[767,]
data.use[767,] 

