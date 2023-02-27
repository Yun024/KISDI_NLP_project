# 토픽모델링 기반 ICT분야 빅데이터 트렌드 분석

## ※ 분석배경 및 필요성

- 4차 산업혁명 및 ICT 분야의 급격한 변화로 인해 사회적으로 다양한 이슈가 발생하고 있음
- 정부차원에서 4차 산업혁명의 핵심 기술에 대한 정책수요 증가
- 위의 상황에서 발생한 사회적 이슈에 대해 정부출연연구기관이 어느정도 인지·분석하고 정책에 반영하고 있는지 확인할 방법이 필요함
- 현재 정책 시행의 적시성을 객관적으로 검증하고 판별하여 미래 정책이슈를 구체적으로 예측할 수 있는 알고리즘이 필요함

## ※ 연구목표

국내외적으로 빠르게 변화하는 ICT 흐름을 파악하기 위해 LDA 토픽모델링 기법을 적용하여 국내 연구자들이 수행하고 있는 ICT분야 국가연구개발사업 과제정보에 대한 주요 연구 토픽과 연구 동향을 탐색한 후 이를 미래동향까지 예측이 가능한 알고리즘 구현 

## ※ 사용기술 

- Language
  + `Python` `R`
- Model
  + `HAN` `LDA` `VAR` `LSTM`

## ※ 데이터
|번호|활용 데이터|형식|행|출처|
|:------:|------|------|------|------|
|1|국가과학기술지식정보서비스 데이터(NTIS)|csv|286,028건|*[Url](https://m.ntis.go.kr/ThMain.do)*|
|2|국가정책연구포털 데이터(NKIS)|csv|7,451건|*[Url](https://www.nkis.re.kr/main.do)*|
|3|정보통신정책연구원 내 보고서(KISDI REPORT)|csv|910건|*[Url](https://www.kisdi.re.kr/index.do)*|
|4|2023 정부 120대 국정과제목록(GOV)|csv|120건|*[Url](https://www.korea.kr/main.do)*|

## ※ Contents

### 0. NKIS자료 준비 *[바로가기](https://github.com/Yun024/KISDI_NLP_project/blob/main/0.%20NKIS%EC%9E%90%EB%A3%8C%20%EC%A4%80%EB%B9%84/README.md)*

- 크롤링
- 전처리
- EDA

### 1. 용어사전 구축 및 자료 업데이트 *[바로가기](https://github.com/Yun024/KISDI_NLP_project/blob/main/1.%20%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84%20%EA%B5%AC%EC%B6%95%20%EB%B0%8F%20%EC%9E%90%EB%A3%8C%20%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8/README.md)*

- 데이터 구조 변경 
- 사전 생성
- 사전 수정 및 컴파일 

### 2. HAN모델을 이용한 ICT분류 *[바로가기](https://github.com/Yun024/KISDI_NLP_project/tree/main/2.%20HAN%EB%AA%A8%EB%8D%B8%EC%9D%84%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20ICT%EB%B6%84%EB%A5%98)*

- HAN모델 학습 및 구축
- HAN모델을 이용한 ICT분류 
- LDA모델에 사용할 TXT파일 생성
- TXT파일을 이용한 워드클라우드 

### 3. LDA모델을 이용한 토픽 분류 *[바로가기](https://github.com/Yun024/KISDI_NLP_project/blob/main/3.%20LDA%EB%AA%A8%EB%8D%B8%EC%9D%84%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20%ED%86%A0%ED%94%BD%20%EB%B6%84%EB%A5%98/README.md)*

- LDA모델 학습 및 구축
- LDA모델을 이용한 토픽 분류 후 분석 
- 토픽 별 텍스트 네트워크 

### 4. DTM모델을 이용한 토픽 분류 *[바로가기](https://github.com/Yun024/KISDI_NLP_project/blob/main/4.%20DTM%EB%AA%A8%EB%8D%B8%EC%9D%84%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20%ED%86%A0%ED%94%BD%20%EB%B6%84%EB%A5%98/README.md)*

- DTM모델 학습 및 구축
- DTM모델을 이용한 토픽 분류 후 분석 
- 토픽 별 텍스트 네트워크 

### 5. VAR,LSTM모델을 이용한 트렌드 분석 *[바로가기](https://github.com/Yun024/KISDI_NLP_project/blob/main/5.%20VAR%2CLSTM%EB%AA%A8%EB%8D%B8%EC%9D%84%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20%ED%8A%B8%EB%A0%8C%EB%93%9C%20%EB%B6%84%EC%84%9D/README.md)*

- VAR모델을 이용한 토픽 별 분포 및 주요단어 예측
- LSTM모델을 이용한 토픽 별 분포 및 주요단어 예측 


### 6. LDA모델을 이용한 세분화 분류 *[바로가기](https://github.com/Yun024/KISDI_NLP_project/blob/main/6.%20LDA%EB%AA%A8%EB%8D%B8%EC%9D%84%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20%EC%84%B8%EB%B6%84%ED%99%94%20%EB%B6%84%EB%A5%98/README.md)*

- 각 토픽 별 LDA모델을 통한 중분류 진행 
- 중분류 토픽 분류 후 분석 

## ※ 활용 방안

- ICT의 여러 주제 별 분포를 파악하여 향후 연구 기획방향을 설정할 수 있음 
- 데이터 간 비교를 통해 각 포털 별 어떠한 토픽에 집중하고 있는지 확인이 가능하고 이에 대한 피드백이 가능함
- 데이터 베이스 내에서 유사한 자료를 찾을 수 있는 검색 엔진 혹은 대시보드로 활용할 수 있음 

## ※ 보완점 

- 전문적인 단어, 신조어 등 용어 사전에 존재하지 않아 실제로 문서에 등장하지만 분석에는 활용되지 않은 단어들을 활용하는 방안을 고려해야 함 
- HAN분류모델의 ICT문서의 정의 부분을 연구자 임의의 판단으로 결정하였기 때문에 이에 대한 타당성을 보완할 필요가 있음 
- 현 연구는 한국어 위주의 문서에 대한 분석이 가능하기 때문에 외국어 문서에 대한 분석도 가능하도록 보완할 필요가 있음
