### 변수초기화
rm(list=ls())

### 경로설정
getwd()
setwd("C:/Users/newcomer02/Desktop/working/R")
dir()

### 라이브러리 불러오기
library(rvest)
library(stringr)
library(dplyr)


### URL 데이터 불러오기(Urlcrawler_NKIS.R에서 불러온 Url)
url_pol <- read.table("Policy_url_2017.txt") 
url_re <- read.table("Research_url_2017.txt")
url <- rbind(url_pol,url_re)

### NKIS webcrawler
i <- 6072  
zz <- url %>% count() %>% as.numeric()   #반복문end값에 넣을 숫자 할당
use.data <- data.frame()
## "공동책임자"가 있는 경우와 없는 경우를 if문으로 구분하여 웹크롤링 진행 
for (i in 1:zz){
  url_news <- url[i,]
  html_news <- read_html(url_news,encoding="cp-949")
  if(c(html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[4]/div/strong/span') %>% html_text(trim=T))=="공동책임자"){
    use.data[i,1] <- html_news %>% html_nodes(xpath='//*[@id="reportName"]') %>% html_text(trim=T)
    use.data[i,2] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[2]/div/span/span') %>% html_text(trim=T)
    use.data[i,3] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[3]/div[1]/span/span') %>% html_text(trim=T)
    use.data[i,4] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[3]/div[2]/span/span') %>% html_text(trim=T)
    use.data[i,5] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[4]/div[1]/span/span') %>% html_text(trim=T)
    use.data[i,6] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[5]/div[1]/span/span') %>% html_text(trim=T)
    use.data[i,7] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[5]/div[2]/span/span') %>% html_text(trim=T)
    use.data[i,8] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[6]/div[2]/span/span') %>% html_text(trim=T)
    use.data[i,9] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[7]/div/span/span') %>% html_text(trim=T)
    use.data[i,10] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[9]/div[1]/span/span') %>% html_text(trim=T)
    use.data[i,11] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[9]/div[2]/span/span') %>% html_text(trim=T)
    use.data[i,12] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[10]/div[1]/span/span') %>% html_text(trim=T)
    use.data[i,13] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[10]/div[2]/span/span') %>% html_text(trim=T)
    use.data[i,14] <- (html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[3]/dl/dd[1]/div') %>% html_text(trim=T))[1]
    use.data[i,15] <- url[i,]
  }else{
    use.data[i,1] <- html_news %>% html_nodes(xpath='//*[@id="reportName"]') %>% html_text(trim=T)
    use.data[i,2] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[2]/div/span/span') %>% html_text(trim=T)
    use.data[i,3] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[3]/div[1]/span/span') %>% html_text(trim=T)
    use.data[i,4] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[3]/div[2]/span/span') %>% html_text(trim=T)
    use.data[i,5] <- ""
    use.data[i,6] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[4]/div[1]/span/span') %>% html_text(trim=T)
    use.data[i,7] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[4]/div[2]/span/span') %>% html_text(trim=T)
    use.data[i,8] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[5]/div[2]/span/span') %>% html_text(trim=T)
    use.data[i,9] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[6]/div/span/span') %>% html_text(trim=T)
    use.data[i,10] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[8]/div[1]/span/span') %>% html_text(trim=T)
    use.data[i,11] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[8]/div[2]/span/span') %>% html_text(trim=T)
    use.data[i,12] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[9]/div[1]/span/span') %>% html_text(trim=T)
    use.data[i,13] <- html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[2]/ul/li[9]/div[2]/span/span') %>% html_text(trim=T)
    use.data[i,14] <- (html_news %>% html_nodes(xpath='//*[@id="mainContents"]/section/div[3]/dl/dd[1]/div') %>% html_text(trim=T))[1]
    use.data[i,15] <- url[i,]
  } 
}
use.data %>% head()
use.data %>% tail()

### 변수명 변경
names(use.data) <- c("보고서명","영문보고서명","연구책임자","소속기관","공동책임자","내부연구참가자","외부연구참가자",
                     "공개여부","출판년도","보고서유형","연구유형","표준분류","자료유형","국문초록","Url주소")
                     
### 최종데이터 csv로 추출                                       
write.csv(use.data,"NKIS_2017~2022.csv")

