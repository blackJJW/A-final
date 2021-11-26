import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import sys
from tqdm import tqdm

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

class Set_Data:
    def __init__(self, df):
        print("extra_fea - Set_Data  Start")
        self.stock_file_name = df
        
        self.refine_data()
        
    def refine_data(self):
        print("extra_fea - Set_Data - refine_data Start")
        company_stock = self.stock_file_name
        print("----- dropping columns Start -----")
        try:
            company_stock_1 = company_stock.drop(columns=["Unnamed: 0","등락률","거래대금","시가총액","상장주식수"])
        except:
            company_stock_1 = company_stock.drop(columns=["등락률","거래대금","시가총액","상장주식수"])
        print("----- dropping columns Done -----")
        
        print("----- transforming datetime / sorting Start -----")
        # 날짜 데이터를 datetime 형식으로 바꾸고 순서 재정렬
        company_stock_1['일자'] = company_stock_1['일자'].map(lambda x : datetime.strptime(x, "%Y/%m/%d"))
        company_stock_1 = company_stock_1.sort_values('일자')
        print("----- transforming datetime / sorting Done -----")

        print("----- changing column names Start -----")
        #컬럼명을 영어로 바꿈
        company_stock_1.columns = ['date', 'close', 'diff' , 'start', 'high' ,'low', 'volume']
        company_stock_1 = company_stock_1.set_index('date') # date를 index로 설정
        print("----- changing column names Done -----")
        self.df = company_stock_1
        print("extra_fea - Set_Data - refine_data Done")
        
    def return_df(self):
        print("extra_fea - Set_Data  Done")
        return(self.df)
                
class Extra_Features_1:
    def __init__(self, df):
        print("extra_fea - Extra_Features_1  Start")
        self.df = df
        self.day_list = [3, 7, 15, 30]

        self.ma()
        self.ema()
        self.ppo()
        self.rsi()
        #self.high_low()
        self.cci()
        self.macd()
        
    #이동평균선
    def ma(self):
        for day in self.day_list:
            #이동평균 칼럼 추가 당일 포함 3일 7일 15일 30일
            self.df['MA_{}'.format(int(day))] = self.df['close'].rolling(window=int(day)).mean()
            
    #지수이동평균선
    def ema(self):
        for day in self.day_list:
            self.df['EWM_{}'.format(int(day))] = self.df['close'].ewm(span=int(day)).mean()

    #이격도
    def ppo(self):
        for day in self.day_list:
            self.df['PPO_{}'.format(int(day))] = (self.df['close'] / self.df['MA_{}'.format(int(day))])*100
            
    #rsi        
    def rsi(self):
        def U(x):
            if x >= 0:
                return x
            else :
                return 0
        def D(x):
            if x <= 0:
                return x*(-1)
            else :
                return 0
            
        self.df['diff_rsi'] = (self.df['close'].shift(1) - self.df['close'])
        self.df['AU'] = self.df['diff_rsi'].apply(U).rolling(window=14).mean() 
        self.df['AD'] = self.df['diff_rsi'].apply(D).rolling(window=14).mean() 
        self.df['RSI'] = self.df['AU']/(self.df['AU']+self.df['AD'])
        self.df.drop(['diff_rsi', 'AU', 'AD'], axis=1, inplace = True)
    '''    
    #모멘텀 스토캐스틱
    def high_low(self):
        day = 5
        self.df['high_st'] = np.nan
        self.df['low_st'] = np.nan
        self.df = self.df.reset_index()
        
        for i in range(len(self.df)-day+1):
            self.df.loc[i, 'high_st']= self.df[i:i+day]['high'].max()
            self.df.loc[i, 'low_st']= self.df[i:i+day]['low'].min()

        self.df['high_st_4'] = self.df['high_st'].shift(4)
        self.df['low_st_4'] = self.df['low_st'].shift(4)

        self.df['fast_K'] = (self.df['close']-self.df['low_st_4'])/(self.df['high_st_4']-self.df['low_st_4'])
        self.df['fast_D'] = self.df['fast_K'].rolling(3).mean()
        self.df['slow_K'] = self.df['fast_D']
        self.df['slow_D'] = self.df['slow_K'].rolling(3).mean()
        self.df = self.df.set_index('date')
        
        self.df = self.df.drop(['high_st', 'low_st', 'high_st_4', 'low_st_4', 'fast_K', 'fast_D'], axis = 1)
    '''
    #CCI
    def cci(self):
        #CCI = (M-N) / (0.015*D)
        # M=특정일의 고가,저가, 종가의 평균
        # N = 일정기간동안의 단순이동평균 통상적으로 20일로 사용
        # D = M-N의 일정기간동안의 단순이동평균
        M = ((self.df.high)+(self.df.low)+(self.df.close)) / 3
        N = M.rolling(20).mean()
        D = (M-N).rolling(20).mean()
        CCI = (M - N)/ (0.015 * D)
        self.df['CCI'] = CCI
        
    #macd
    def macd(self):
        short_ = 12 
        long_ = 26 
        t = 9 

        ma_12 = self.df.close.ewm(span = short_).mean()
        ma_26 = self.df.close.ewm(span = long_).mean() # 장기(26) EMA
        macd = ma_12 - ma_26 # MACD
        macdSignal = macd.ewm(span = t).mean() # Signal
        macdOscillator = macd - macdSignal # Oscillator
        self.df['macd'] = macdOscillator
        
    def show_df(self):
        print(self.df)
        
    def return_df(self):
        print("extra_fea - Extra_Features_1  Done")
        return(self.df)
    
class ML_Part_1:
    def __init__(self, stock_file_name):
        print("extra_fea - ML_Part_1  Start")
        self.df =  pd.read_csv('./data/stock/extra_f/'+stock_file_name, encoding='cp949')
        self.stock_file_name = stock_file_name
        
        self.ready()
        self.cross_accuracy()
        self.income_rate()
        
    def ready(self):
        print("extra_fea - ML_Part_1 - ready  Start")
        print("----- setting columns Start -----")
        def up_down(x):
            if x >= 0:
                return 1 #내일의 종가가 오르거나 그대로면 1
            else :
                return 0
            
        self.df = self.df.set_index('date')
        self.df['fluctuation'] = (self.df['close'].shift(-1)-self.df['close']).apply(up_down)
        self.df.drop('diff', axis=1, inplace=True)
        print("----- setting columns Done -----")
        
        print("----- spliting train, test Start -----")
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
        print("----- spliting train, test Done -----")
        print("extra_fea - ML_Part_1 - ready  Done")
        
    def cross_accuracy(self):
        print("extra_fea - ML_Part_1 - cross_accuracy  Start")
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
        print("----- generating cv-accuracy Start -----")
        self.models = [{'name' : 'Logistic', 'model' : logistic}, {'name' : 'KNN', 'model' : knn},
                {'name' : 'DecisonTree', 'model' : decisiontree}, {'name' : 'RandomForest', 'model' : forest},
                {'name' : 'NaiveBayes', 'model' : naive}]

        temp = sys.stdout
        sys.stdout = open('./data/report/cv_acc/'+self.stock_file_name+'_extra_cv_accuracy_report.txt', 'w')

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
        print("----- generating cv-accuracy Done -----")
        print("extra_fea - ML_Part_1 - cross_accuracy  Done")
        
    def income_rate(self):
        print("extra_fea - ML_Part_1 - income_rate  Start")
        print("----- creating income ratio report Start -----")
        def rate_of_return():
            df['percent'] = round((df.close-df.close.shift(1))/df.close.shift(1)*100, 2) 
            #round(0.4457, 2) > 0.4475를 소수점 아래 둘째 자리로 반올림한다.
            for i in tqdm(range(len(df)-1)):
                if (df.loc[i, 'predicted'] == 0):
                    df.loc[i+1, 'percent'] = df.loc[i+1, 'percent']
        
        temp = sys.stdout           
        sys.stdout = open('./data/report/income_rate/'+self.stock_file_name+'_extra_income_rate_report.txt', 'w')
        
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
        print("----- creating income ratio report Done -----")
        print("extra_fea - ML_Part_1 - income_rate  Done")
        print("extra_fea - ML_Part_1  Done")
    