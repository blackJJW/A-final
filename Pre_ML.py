import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch 
from torch.autograd import Variable

mm = MinMaxScaler()
ss = StandardScaler()

class Refine_DF:
    def __init__(self, stock_pos_neg_file_name):
        self.file_name = stock_pos_neg_file_name
        
        self.refining_df()
                
    def refining_df(self):
        stock_pos_neg_file = pd.read_csv('./data/stock_pos_neg/'+self.file_name, encoding="cp949")
        stock_pos_neg_file = stock_pos_neg_file.drop(columns=["Unnamed: 0", "level_0", "index", "상장주식수", "거래대금", "시가총액"])
        stock_pos_neg_file = stock_pos_neg_file.rename(columns = {"일자":"date","종가":"close", "거래량":"volume",
                                                                  "시가":"open","고가":"high", "저가":"low","대비":"diff","등락률":"ratio"})
        stock_pos_neg_file = stock_pos_neg_file.set_index("date")
        stock_pos_neg_file = stock_pos_neg_file.dropna()
        stock_pos_neg_file_1 = stock_pos_neg_file.copy()
        
        return stock_pos_neg_file_1
        
class Extra_Features:
    def __init__(self, df):
        self.df = df
        self.day_list = [3, 7, 15, 30]

        self.ma()
        self.ema()
        self.ppo()
        self.rsi()
        self.high_low()
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
        return(self.df)
    
class Scaler:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.prob = 0.7
        
    def prep(self):
        
        self.X_ss = ss.fit_transform(self.X)
        self.y_mm = mm.fit_transform(self.y)
        
        #------Test Data------------------------------------------------
        self.X_train = self.X_ss[:int(len(self.X_ss) * self.prob), :]
        self.X_test = self.X_ss[int(len(self.X_ss) * self.prob):, :]
        self.y_train = self.y_mm[:int(len(self.y_mm) * self.prob), :]
        self.y_test = self.y_mm[int(len(self.y_mm) * self.prob):, :]
        #---------------------------------------------------------------
        
        #numpy형태에서는 학습이 불가능하기 때문에 학습할 수 있는 형태로 변환하기 위해 Torch로 변환

        self.X_train_tensors = Variable(torch.Tensor(self.X_train)) 
        self.X_test_tensors = Variable(torch.Tensor(self.X_test)) 

        self.y_train_tensors = Variable(torch.Tensor(self.y_train)) 
        self.y_test_tensors = Variable(torch.Tensor(self.y_test)) 

        self.X_train_tensors_final = torch.reshape(self.X_train_tensors, 
                                                   (self.X_train_tensors.shape[0], 1, self.X_train_tensors.shape[1])) 
        self.X_test_tensors_final = torch.reshape(self.X_test_tensors, 
                                                  (self.X_test_tensors.shape[0], 1, self.X_test_tensors.shape[1]))
        
        return self.X_train_tensors_final, self.X_test_tensors_final, self.y_train_tensors, self.y_test_tensors, int(len(self.X_ss) * self.prob)