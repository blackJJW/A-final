# 뉴스 감성 분석을 통한 주가예측

---

- **인원** : 4명
- **기간** : 2021.10.14 ~ 2021.11.30
- **사용언어** : Python
- **사용 라이브러리**
    - `konlpy`
    - `pandas`, `numpy`, `csv`, `json`
    - `matplotlib`, `scipy`
    - `selenium`, `chromedriver_autoinstaller`, `Beautifulsoup`
    - `sklearn`, `xgboost`, `torch`
    - `tqdm`, `os`, `shutil`, `time`, `re`, `datetime`, `collections`
    - `tkinter`
- 작업 툴
    - `VScode`
    - `Spyder`
    - `Colab`
    - `Jupyter Notebook`

## 프로젝트 목적

---

- 원하는 기업의 뉴스를 크롤링 할 수 있어야 한다.
- 여러 형태소 분석기를 사용하여 최적의 분석기를 선택한다.
- Machine Learning, Deep Learning 의 여러가지 기술들을 사용, 비교하여 최고의 성능을 나타내는 모델을 찾아낸다.
- 사용자가 사용할 수 있게 서비스가 가능하도록 한다.

## 팀원 및 역할 분담

---

- 팀원 : 조세연, 정진우, 이예지, 한상백
- 조세연 - 팀장
    - 자료 수집, 논문 연구
    - 데이터 크롤링
    - 데이터 전처리
    - 웹 페이지 서비스 구현
- 정진우
    - 자료 수집, 논문 연구
    - 크롤링 프로그램 작성
    - 데이터 크롤링
    - 데이터 전처리
    - 프로그램 구조 계획
    - 프로그램 코드 작성
    - 영상 편집
- 이예지
    - 자료 수집, 논문 연구
    - 프로그램 기획 및 계획
    - 데이터 크롤링,
    - 데이터 전처리
    - 웹 페이지 서비스 구현
- 한상백
    - 자료 수집, 논문 연구
    - 데이터 크롤링
    - 데이터 전처리
    - 프로그램 구조 계획
    - 웹 페이지 서비스 구현

## 전체적인 Process

---

![Untitled](img/Untitled.png)

## 데이터 처리 과정

---

![Untitled](img/Untitled%201.png)

## 한계점 및 시행착오

---

### 한계점

- 주식시장에 특화된 감성사전이 아닌 이전 주가 지표들을 활용한 라벨링에 의해 구축된 것이라 극성 정확도가 낮다.
- 종목별로 성능이 좋은 모델과 보조지표가 다르다.
- 다음날 종가에 대해 상승, 하락이라는 2가지의 예측 결과만을 보인다.
- 실제 주식시장에는 더 많은 변수가 존재하고 이를 모두 반영하지 못했다.
- 데이터가 많아진다고 해서 정확도가 확연하게 커지지 않는다.

### 시행착오

- 원래 목표는 웹페이지를 구현하여 누구나 예측 서비스를 사용 가능하도록 하는 것이었지만, 웹페이지를 구현하지 못했다.
    - 기능이 작동하도록 하기 위해 tkinter를 이용하여 사용자가 쉽게 사용가능한 프로그램으로 완성하였다.

## 향후 과제

---

- 주식시장에 특화된 감성 사전 구축이 필요, 지속적인 확장이 있어야 한다.
- 보편적으로 활용할 수 있는 예측 모델 개발이 요구되며, 다양한 종목들에 대한 분석이 필요
- 얼마만큼 오르고, 내릴 것인지에 대한 분석이 필요
- 실제 투자를 하여 지속적인 예측모델의 개선이 필요
- 등락폭이 큰 종목을 가지고 분석하여 개선을 해야 한다.

## 결과 자료

---


![A-team 결과 ppt](pdf/new_stock_prediction.pdf)
![A-team 결과 ppt google drive](https://drive.google.com/file/d/1mXmaurSsZf5xGDKVsjhImYvcaP7WI_XY/view?usp=sharing)
[![시연영상]()](https://drive.google.com/file/d/1qnK7A0GaMw1zhXHgt26dS85Rou1XlrnY/view?usp=sharing)