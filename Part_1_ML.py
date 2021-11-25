# Importing Classifier Modules
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#cross_validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import sys
from tqdm import tqdm
#-------------------------------------------
#     training, test 데이터 크기 , 개수
#     모델 별 (LogisticRegression, KNeighborsClassifier,
#              DecisionTreeClassifier, RandomForestClassifier,
#              GaussianNB)
#
#             CV score
#             classification_report
#             confusion matrix
#             Accuracy Score
#-------------------------------------------

#------------------------------------------
#       model name :
#       첫날을 제외한 거래일수 :
#       누적 수익률 :
#       1일 평균 수익률 :
#------------------------------------------

class Set_Data:
    def __init__(self, stock_file_name):
        self.stock_file_name = stock_file_name
        
        self.refine_data()
        
    def refine_data(self):
        company_stock = pd.read_csv('./data/stock/'+self.stock_file_name, encoding="cp949")
        company_stock_1 = company_stock.drop(columns=["등락률","거래대금","시가총액","상장주식수"])
        
        # 날짜 데이터를 datetime 형식으로 바꾸고 순서 재정렬
        company_stock_1['일자'] = company_stock_1['일자'].map(lambda x : datetime.strptime(x, "%Y/%m/%d"))
        company_stock_1 = company_stock_1.sort_values('일자')

        #컬럼명을 영어로 바꿈
        company_stock_1.columns = ['date', 'close', 'diff' , 'start', 'high' ,'low', 'volume']
        company_stock_1 = company_stock_1.set_index('date') # date를 index로 설정

        company_stock_1.to_csv('./data/stock/refined/'+self.stock_file_name+'_refined.csv', encoding='cp949')
    
class ML_Part_1:
    def __init__(self, stock_file_name):
        self.df =  pd.read_csv('./data/stock/refined/'+stock_file_name+'_refined.csv', encoding='cp949')
        self.stock_file_name = stock_file_name
        
        self.ready()
        self.cross_accuracy()
        self.income_rate()
        
    def ready(self):
        def up_down(x):
            if x >= 0:
                return 1 #내일의 종가가 오르거나 그대로면 1
            else :
                return 0
            
        self.df = self.df.set_index('date')
        self.df['fluctuation'] = (self.df['close'].shift(-1)-self.df['close']).apply(up_down)
        self.df.drop('diff', axis=1, inplace=True)

        rp =int(len(self.df)*0.7)

        print('기존의 데이터 갯수 :', len(self.df))
        print('최소한의 데이터 갯수 : ', len(self.df.iloc[len(self.df)-rp:]))
        print('\n')
        self.df_part = self.df.iloc[len(self.df)-rp:]

        today = self.df_part.iloc[-1]
        company_stock_1_df = self.df_part.iloc[:-1]
        target = company_stock_1_df['fluctuation']
        company_stock_1_df = company_stock_1_df.drop('fluctuation', axis = 1)

        self.train, self.test, self.train_target, self.test_target = train_test_split(company_stock_1_df, target, test_size = 0.3, shuffle=False ) 
    def cross_accuracy(self):
        #cross_val_score에서 분류모형의 scoring은 accuracy이다.
        kfold = KFold(n_splits = 3, shuffle = False, random_state = None)

        # 분류모형
        logistic = LogisticRegression()
        knn = KNeighborsClassifier()
        decisiontree = DecisionTreeClassifier(random_state=None)
        forest = RandomForestClassifier(random_state=None)
        naive = GaussianNB()
        # SVM은 매개변수와 데이터 전처리 부분에서 신경써야함. 따라서 현재 사용하지 않는다.
        # 추후 매개변수를 선택하는 알고리즘을 짠 후 사용하도록 하자

        self.models = [{'name' : 'Logistic', 'model' : logistic}, {'name' : 'KNN', 'model' : knn},
                {'name' : 'DecisonTree', 'model' : decisiontree}, {'name' : 'RandomForest', 'model' : forest},
                {'name' : 'NaiveBayes', 'model' : naive}]

        temp = sys.stdout
        sys.stdout = open('./data/report/cv_acc/'+self.stock_file_name+'_cv_accuracy_report.txt', 'w')

        def cv_accuracy(models):
            for m in tqdm(models):
                print("Model {} CV score : {:.4f}".format(m['name'], 
                                                          np.mean(cross_val_score(m['model'], 
                                                                                  self.train, self.train_target, cv=kfold))))
        cv_accuracy(self.models)
        
        for m in tqdm(self.models) : 
            model = m['model']
            model.fit(self.train, self.train_target)

            predicted = model.predict(self.test)

        #Accuracy : 전체 샘플 중 맞게 예측한 샘플수의 비율
        #Precision(정밀도) : postive라고 예측한 것 중에서 실제 postive인 것
        #Recall(재현율) : 실제 postive중에 예측한 postive 비율
            print('\n')
            print ('model name : {}'.format(m['name']))
            print (metrics.classification_report(self.test_target, predicted))

        #confusion_matrix에서
        #행은 실제값, 열은 예측한 값으로 0 1 순서대로 임
            print('Confusion Matrix') 
            print (metrics.confusion_matrix(self.test_target, predicted))

            print ('Accuracy Score : {:.4f}\n'.format(metrics.accuracy_score(self.test_target, predicted)))
        sys.stdout.close()
        sys.stdout = temp

        
    def income_rate(self):
        def rate_of_return():
            df['percent'] = round((df.close-df.close.shift(1))/df.close.shift(1)*100, 2) 
            #round(0.4457, 2) > 0.4475를 소수점 아래 둘째 자리로 반올림한다.
            for i in tqdm(range(len(df)-1)):
                if (df.loc[i, 'predicted'] == 0):
                    df.loc[i+1, 'percent'] = df.loc[i+1, 'percent']
        
        temp = sys.stdout           
        sys.stdout = open('./data/report/income_rate/'+self.stock_file_name+'_income_rate_report.txt', 'w')
        
        for m in tqdm(self.models) : 
            model = m['model']
            model.fit(self.train, self.train_target)

            predicted = model.predict(self.test)

            df = pd.concat([self.test.reset_index().drop('date', axis=1), pd.DataFrame(predicted, columns = ['predicted'])], axis=1)

            rate_of_return()

            df.dropna(inplace = True)

            print('model name : {}'.format(m['name']))
            print('첫날을 제외한 거래일수 : {}'.format(len(df)))
            print('누적 수익률 : {}'.format(round(df['percent'].sum(), 2)))
            print('1일 평균 수익률 : {}\n'.format(round(df['percent'].sum()/(len(df)-1),2)))
        
        sys.stdout.close()
        sys.stdout = temp