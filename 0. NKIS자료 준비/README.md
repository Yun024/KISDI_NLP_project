# NKIS자료 준비 

```
- 경제·인문사회연구회와 소관정부출연연구기관의 연구성과물
- 언어 R을 사용하여 진행 
```

## ※ NKIS데이터 크롤링 

- 오픈API를 이용하여 연구보고서와 정책보고서 각각의 Url_crawler 제작  *[바로가기](https://github.com/Yun024/NLP_ICT_Trend/blob/main/0.%20NKIS%EC%9E%90%EB%A3%8C%20%EC%A4%80%EB%B9%84/0-1.%20URLcrawler_NKIS.R)*
- Url_crawler를 이용힌 Web_crawler를 제작하고 if문과 for문을 이용한 데이터 추출 진행  *[바로가기](https://github.com/Yun024/NLP_ICT_Trend/blob/main/0.%20NKIS%EC%9E%90%EB%A3%8C%20%EC%A4%80%EB%B9%84/0-2.%20webcrawler_NKIS.R)*

## ※ 전처리 *[바로가기](https://github.com/Yun024/NLP_ICT_Trend/blob/main/0.%20NKIS%EC%9E%90%EB%A3%8C%20%EC%A4%80%EB%B9%84/0-3.%20preprocessing.R)*

- `NA제거` `특수기호 제거` `2017이전 년도 데이터 제거` `포털에서부터 이상하게 입력된 데이터 수동제거`
- '국문초록'이 셀당 글자 수(32767)를 넘기는 경우 전처리 
- `표준 분류 세분화` `인덱스 재설정` 
- 원본 데이터 8521행, 134열 -> 최종 데이터 7451행, 15열 

## ※ EDA *[바로가기](https://github.com/Yun024/NLP_ICT_Trend/blob/main/0.%20NKIS%EC%9E%90%EB%A3%8C%20%EC%A4%80%EB%B9%84/0-5.%20EDA.R)*

- NKIS데이터의 논문제목을 이용한 워드 클라우드 *[바로가기](https://github.com/Yun024/NLP_ICT_Trend/blob/main/0.%20NKIS%EC%9E%90%EB%A3%8C%20%EC%A4%80%EB%B9%84/0-4.%20NKIS_Title_WordCloud.R)*

![image](https://user-images.githubusercontent.com/52143231/221649558-26f50765-6658-4fd1-be70-4ba8a829db67.png)

- NKIS데이터의 년도 별 대분류 추이 그래프 

![image](https://user-images.githubusercontent.com/52143231/221649952-c5db3bdf-836b-48fc-b57a-840bf9367f23.png)

- NKIS데이터의 5개년 대분류 산출물 비율

![image](https://user-images.githubusercontent.com/52143231/221650119-53d84d33-4477-4035-948f-97dc23fe6487.png)
