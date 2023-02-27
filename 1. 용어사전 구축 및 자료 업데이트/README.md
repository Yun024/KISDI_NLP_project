# 용어사진 구축 및 자료 업데이트

```
- 차년도 데이터에 대한 전처리 및 통합과정 정립 
- 한글 텍스트 분석의 핵심인 형태소 사전을 구축하는 과정 
- 구축한 사전을 이용한 Mecab사전 적용(Compile)방법 
```

## ※ 신규 데이터 전처리 *[바로가기](https://github.com/Yun024/NLP_ICT_Trend/blob/main/1.%20%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84%20%EA%B5%AC%EC%B6%95%20%EB%B0%8F%20%EC%9E%90%EB%A3%8C%20%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8/1-1.%20Add_DATA.py)*

- 각 데이터별로 추가된 데이터의 변수명을 변경하여 통일
- 'contents'의 전문이 영어로 되어있는 행 제거 후 기존 데이터와 통합

## ※ 사전 구축 *[바로가기](https://github.com/Yun024/NLP_ICT_Trend/blob/main/1.%20%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84%20%EA%B5%AC%EC%B6%95%20%EB%B0%8F%20%EC%9E%90%EB%A3%8C%20%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8/1-6-1.%20New%EC%82%AC%EC%A0%84.py)*

- NIADic, TTA용어사전, NTIS의ko_key를 베이스로 사용 
- 특수문자 제거, 공백 제거, 중복 값 제거의 과정을 거쳐 최종 단어 사전 구축
- 각각 92만건, 2만5천건, 122만건 => 106만 건

## ※ 우선순위 설정 및 컴파일 *[바로가기](https://github.com/Yun024/NLP_ICT_Trend/blob/main/1.%20%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84%20%EA%B5%AC%EC%B6%95%20%EB%B0%8F%20%EC%9E%90%EB%A3%8C%20%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8/1-6-3.%20%EC%B5%9C%EC%A2%85%20%EC%BB%B4%ED%8C%8C%EC%9D%BC%20%EC%A0%84%EC%B2%98%EB%A6%AC%20%EB%B0%8F%20%EC%9A%B0%EC%84%A0%20%EC%88%9C%EC%9C%84%20%EC%84%A4%EC%A0%95.py)*

- 단어의 품사에 따라 길이가 다르거나 복합 명사의 경우도 형태소 분석기가 인식할 수 있도록 우선순위 설정 
- 우선순위는 단어의 길이가 짧을수록 후순위, 길수록 선순위가 되도록 Min-Max-Scaling를 적용하여 설정 
- window powershell의 관리자 모드를 이용하여 Compile을 진행하여 용어사전 적용  
- ex)공격, 공격하다, 공격함/ 딥 러닝, 머신 러닝, 데이터 플랫폼 


`최종 용어사전 CSV파일 형태`

<img src = "https://user-images.githubusercontent.com/52143231/221653433-bbf14897-d5f5-4301-b68a-b0c878fb9979.png" width = 800, height = 150>






