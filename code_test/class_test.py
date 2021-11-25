#%%
import pandas as pd

import Part_1_Pre
import Part_1_LSTM
import Part_1_ML

company_data = pd.read_csv('./data/result/셀트리온_test_1_result.csv', encoding="cp949")
# %%
#-----------------Part_1_Pre------------------------------
a =  Part_1_Pre.Refine_DF('셀트리온_test_1_result.csv')
# %%
b = a.refining_df()
# %%

print(b)

# %%
c = Part_1_Pre.Extra_Features(b)

d = c.return_df()
# %%
print(d)
# %%

j = Part_1_Pre.ready_data_set(d)

X, y = j.split_X_y()
'''
#----------- data set 준비 ------------
X= d.drop(columns='close')
#0번째부터 마지막에서 앞까지 
X= X.iloc[:-1]
#y는 ratio(등락률)로 잡고, 첫번째부터 마지막까지 읽어옴
y = d.iloc[1:, 2:3]

prep_y = Part_1_Pre.Scaler(X, y)

X_train, X_test, y_train, y_test, length = prep_y.prep()

print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape) 
'''
#------------LSTM------------------------------------------------------------


# %%
prep_y = Part_1_LSTM.Scaler(X, y)
X_train, X_test, y_train, y_test, length = prep_y.prep()

print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape) 
# %%
import torch 
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt 

'''
input_size = 25  #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1  #number of stacked lstm layers
num_classes = 1 #number of output classes 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lstm1 = Part_1_LSTM.LSTM1(num_classes, input_size, hidden_size, num_layers, X_train.shape[1]).to(device)

num_epochs = 90000 #1000 epochs
learning_rate = 0.00001 #0.001 lr

Part_1_LSTM.LSTM_predict(lstm1, num_epochs, learning_rate, X_train, y_train, length, d)
'''
# %%
#-----[load stock data]---------------------------------------------------
company_stock = pd.read_csv("./data/stock/GS2005_2021.csv", encoding="cp949")

p = Part_1_ML.set_data(company_stock)
p_s = p.refine_data()

# %%
Part_1_ML.ML_Part_1(p_s)
# %%
import Part_2_Reg



# %%
a = Part_2_Reg.Prep_Regressor('셀트리온_test_1_result.csv')
b = a.refining_df()
# %%
print(b)
# %%
c = Part_2_Reg.RF_Regressor(b)

# %%
d = Part_2_Reg.XGBoost_Regressor(b)