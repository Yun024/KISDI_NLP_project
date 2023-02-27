rm(list=ls())
getwd()
setwd("C:/Users/newcomer02/Desktop/working/R/결과물")
dir()

##### 라이브러리 불러오기
library(dplyr)
library(stringr)

### EDA를 구하는데 사용되는 데이터 불러오기
data <- read.table(choose.files())
data %>% head()
data %>% names()

### 사용할 변수 추출
data.use <- data[,c(5,10,14,15)]
data.use %>% head()
data.use %>% summary()


### 문자를 factor화 해서 간편하게 요약 보기 
for (i in 1:4){
  data.use[,i] <- data.use[,i] %>% as.character()
}

for (i in 1:4){
  data.use[,i] <- data.use[,i] %>% as.factor()
}
data.use %>% summary()
data.use$대분류 %>% summary()
data.use$소속기관 %>% summary()


##### Group by 
### 출판년도
year.agency<-data.use %>% group_by(출판년도) %>% count(소속기관) %>% as.data.frame()
year.cate<-data.use %>% group_by(출판년도) %>% count(대분류) %>% as.data.frame()

### 대분류
large.small <- data.use %>% group_by(대분류) %>% count(소분류) %>% as.data.frame()
large.agency <- data.use %>% group_by(대분류) %>% count(소속기관) %>% as.data.frame()


### 파일 저장
write.csv(year.agency,"출판년도_별_소속기관.csv")
write.csv(year.cate,"출판년도_별_대분류.csv")
write.csv(large.small,"대분류_별_소분류.csv")
write.csv(large.agency,"대분류_별_소속기관.csv")



  