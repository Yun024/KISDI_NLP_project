
### 변수초기화
rm(list=ls())

### 경로설정
getwd()
setwd("C:/Users/newcomer02/Desktop/working")
dir()

### 라이브러리 불러오기
library(dplyr)
library(stringr)
library(httr)
library(rvest)
library(jsonlite)
---------------------------------------------
---------------------------------------------
---------------------------------------------

###### NKIS데이터 불러오기
### 인증키 불러오기
mykey <- Sys.getenv('NKIS_TOKEN')

### 고정주소 변수할당
##연구보고서
URL <- 'https://nkis.re.kr:4445/nkisApi/search/ReportList.do'
##정책보고서
URL1 <- 'https://nkis.re.kr:4445/nkisApi/search/ResearchList.do'

### Open API 불러오기 
res <- GET(url = URL1,
           query = list(serviceKey = mykey %>% I(), #개인키
                        pageNo = '65',  #몇페이지로
                        rowCnt = '100', #개수
                        pblYyBegin = '2017', #시작년도
                        pblYyEnd = '2022'))  #마지막년도

### cat함수를 통해 불러온 파일 출력
res %>% content(as="text",encoding="euc-kr") %>% cat() 

### total_count를 통해 논문의 수 확인 후 전체 저장
res.count <- res %>% read_html() %>% html_elements(css='total_count') %>% html_text() %>% as.numeric()
res.all <- GET(url = URL1,
           query = list(serviceKey = mykey %>% I(),
                        rowCnt = res.count,
                        pblYyBegin = '2017',
                        pblYyEnd = '2022'))

### 최종 Url 크롤링 
org_link <- res.all %>% read_html() %>% html_elements(css='org_link') %>% html_text(trim=TRUE)

### 데이터 쓰기 
#write(org_link,"Research_url_2017.txt")
#write(org_link,"Policy_url_2017.txt")


