# HAN모델을 이용한 ICT분류
[HAN이미지파일 삽입]

- 문서 내 연속된 문장과 문장 내 연속된 단어의 계층적 구조의 특성을 반영
- 반영된 특성을 통해 중요한 단어와 문장에 가중치를 부여하여 문서 분류 성능을 높인 모델
- 크게 단어 시퀀스와 문장 시퀀스로 구성되어 있으며, 각각의 시퀀스는 인코더와 가중치를 부여하는 부분으로 구성됨

## HAN모델 학습 및 생성 *[바로가기](https://github.com/Yun024/NLP_ICT_Trend/blob/main/2.%20HAN%EB%AA%A8%EB%8D%B8%EC%9D%84%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20ICT%EB%B6%84%EB%A5%98/2-1.%20Train_Classifier_HAN(oversampling).py)*

- ICT 문서의 정의
  + `과학기술표준분류-대1,2,3` 중 하나라도 '정보/통신'인 문서
  + `중점과학기술분류-대`가 ICT·SW에 해당하는 문서
  + `적용분야` 중 하나라도 '출판·영상, 방송통신 및 정보서비스업'인 문서 
  + `6T관련기술-대`가 'IT(정보기술)'에 해당하는 문서
  
- 데이터 샘플링
  + train : val : test = 5 : 3 : 2 로 랜덤 추출하여 데이터 구성 
  + ICT-data가 Non-ICT-data에 비해 적은 모습으로 데이터 불균형을 보이기 때문에 'Oversampling'을 통해 데이터 불균형 해소
  
- HAN 모델 구성
  + 문장 내 최대 단어 개수 : 150
  + 문서 내 최대 문장 수 : 30
  + 학습률 : 0.001
  + Word sequence & Sentence sequence 의 GRU Cell Hidden Size : 10
  + 각 시퀀스의 Timedistributed 층의 개수 : 20
  + 출력층 활성화함수 : softmax 함수 
  
## HAN모델을 이용한 데이터 별 ICT분류 *[바로가기](https://github.com/Yun024/NLP_ICT_Trend/blob/main/2.%20HAN%EB%AA%A8%EB%8D%B8%EC%9D%84%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20ICT%EB%B6%84%EB%A5%98/2-2.%20Classifier_HAN(Final).py)*

[그림 3-1 이미지 삽입]
- 정밀도(Precision)와 재현율(Recall)의 합이 가장 높을 때의 값을 임계점으로 결정 : 0.8774
  + 정밀도 : 모델이 ICT문서라고 분류한 것 중 실제로 ICT문서인 빙류
  + 재현율 : 실제로 ICT문서인 비율 중 모델이 ICT문서라고 분류한 비율 
- HAN모델에서 연구목표는 많은 ICT문서를 확보하는 것이기 때문에 ICT_prob(문서가 ICT일 확률)가 0.8774(임계점)가 아닌 0.5로 정의하여 연구 진행 
- 데이터 분류결과 
  + ICT: Non-ICT  = `NTIS` 27:73 `NKIS` 20:80 `KISDI REPORT` 69:31 `GOV` 29:71


## LDA, DTM모델에 사용할 TXT파일 생성 *[바로가기](https://github.com/Yun024/NLP_ICT_Trend/blob/main/2.%20HAN%EB%AA%A8%EB%8D%B8%EC%9D%84%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20ICT%EB%B6%84%EB%A5%98/2-3.%20Generate_txt(Final).py)*

## TXT파일을 이용한 워드클라우드  *[바로가기](https://github.com/Yun024/NLP_ICT_Trend/blob/main/2.%20HAN%EB%AA%A8%EB%8D%B8%EC%9D%84%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20ICT%EB%B6%84%EB%A5%98/2-4.%20WordCloud.py)*

[워드클라우드 이미지 삽입]
- LDA모델을 진행하는데 있어 `min_cf` `min_df` `rm_top`를 어떻게 구성할것인지 구상해보기 위해 진행
