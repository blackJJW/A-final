#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

company_data = pd.read_csv('./data/result/셀트리온_test_1_result.csv', encoding="cp949")
company_data.dropna(inplace = True)

p_company_data = company_data[['일자', '종가','sumPos', 'sumNeg', '대비', '등락률', '시가', '고가', '저가', '거래량']]
p_company_data = p_company_data.rename(columns = {"일자":"date","종가":"close", "대비":"vs", "등락률":"ratio", "시가":"current_price", "고가":"high_price", 
                                                  "저가":"low_price", "거래량":"trade_volume"})
p_company_data = p_company_data.set_index('date')
p_company_data['Pct_change'] = p_company_data['close'].pct_change()
p_company_data.dropna(inplace = True)

# This function "window_data" accepts the column number for the features (X) and the target (y)
# It chunks the data up with a rolling window of Xt-n to predict Xt
# It returns a numpy array of X any y
def window_data(df, window, feature_col_number1, feature_col_number2, feature_col_number3, 
                feature_col_number4,feature_col_number5,feature_col_number6,feature_col_number7,feature_col_number8,
                feature_col_number9, target_col_number):
    # Create empty lists "X_close", "X_polarity", "X_volume" and y
    X_close = []
    X_sumPos = []
    X_sumNeg = []
    X_vs=[]
    X_ratio =[]
    X_current_price=[]
    X_high_price = []
    X_low_price = []
    X_trade_volume = []
    
    
    
    y = []
    for i in range(len(df) - window):
        
        # Get close, ts_sumPos, ts_sumNeg, and target in the loop
        ts_close = df.iloc[i:(i + window), feature_col_number1]
        ts_sumPos = df.iloc[i:(i + window), feature_col_number2]
        ts_sumNeg = df.iloc[i:(i + window), feature_col_number3]
        
        ts_vs = df.iloc[i:(i + window), feature_col_number4]
        ts_ratio = df.iloc[i:(i + window), feature_col_number5]
        ts_current_price = df.iloc[i:(i + window), feature_col_number6]
        
        ts_high_price = df.iloc[i:(i + window), feature_col_number7]
        ts_low_price = df.iloc[i:(i + window), feature_col_number8]
        ts_trade_volume = df.iloc[i:(i + window), feature_col_number9]
        
        
        


        target = df.iloc[(i + window), target_col_number]
        # Append values in the lists
        X_close.append(ts_close)
        X_sumPos.append(ts_sumPos)
        X_sumNeg.append(ts_sumNeg)
        X_vs.append(ts_vs)
        X_ratio.append(ts_ratio) 
        X_current_price.append(ts_current_price)
        X_high_price.append(ts_high_price) 
        X_low_price.append(ts_low_price) 
        X_trade_volume.append(ts_trade_volume) 
        y.append(target)
        
    return np.hstack((X_close,X_sumPos,X_sumNeg, X_vs, X_ratio, X_current_price, X_high_price, X_low_price, X_trade_volume)), np.array(y).reshape(-1, 1)

# Predict Closing Prices using a 3 day window of previous closing prices
window_size = 5

# Column index 0 is the `Close` column
# Column index 1 is the `sumPos` column
# Column index 2 is the `sumNeg` column
feature_col_number1 = 0
feature_col_number2 = 1
feature_col_number3 = 2
feature_col_number4 = 3
feature_col_number5 = 4
feature_col_number6 = 5
feature_col_number7 = 6
feature_col_number8 = 7
feature_col_number9 = 8





target_col_number = 0
X, y = window_data(p_company_data, window_size, feature_col_number1, 
                   feature_col_number2, feature_col_number3,feature_col_number4, feature_col_number5, 
                   feature_col_number6, feature_col_number7, feature_col_number8, feature_col_number9, 
                   target_col_number)

# Use 70% of the data for training and the remainder for testing
X_split = int(0.7 * len(X))
y_split = int(0.7 * len(y))

X_train = X[: X_split]
X_test = X[X_split:]
y_train = y[: y_split]
y_test = y[y_split:]

# Scaling Data with MinMaxScaler
# value 0 ~ 1
# scale both features and target sets

# Use the MinMaxScaler to scale data between 0 and 1.
x_train_scaler = MinMaxScaler()
x_test_scaler = MinMaxScaler()
y_train_scaler = MinMaxScaler()
y_test_scaler = MinMaxScaler()

# Fit the scaler for the Training Data
x_train_scaler.fit(X)
y_train_scaler.fit(y)

# Scale the training data
X_train = x_train_scaler.transform(X_train)
y_train = y_train_scaler.transform(y_train)

# Scale the y_test data
X_test = x_train_scaler.transform(X_test)
y_test = y_train_scaler.transform(y_test)

'''
# Fit the scaler for the Testing Data
x_test_scaler.fit(X_test)
y_test_scaler.fit(y_test)

# Scale the y_test data
X_test = x_test_scaler.transform(X_test)
y_test = y_test_scaler.transform(y_test)
'''


# Reshape the features for the model
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model. 

# Define the LSTM RNN model.
model = Sequential()

number_units = 15
dropout_fraction = 0.2

# Layer 1
model.add(LSTM(
    units=number_units,
    return_sequences=True,
    input_shape=(X_train.shape[1], 1))
    )
    
model.add(Dropout(dropout_fraction))

# Layer 2
# The return_sequences parameter needs to set to True every time we add a new LSTM layer, excluding the final layer.
model.add(LSTM(units=number_units, return_sequences=True))
model.add(Dropout(dropout_fraction))

# Layer 3
model.add(LSTM(units=number_units))
model.add(Dropout(dropout_fraction))

# Output layer
model.add(Dense(1))

# Compile the model
model.compile(optimizer="adam", metrics = ['accuracy'], loss="mean_squared_error")

# Summarize the model
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=50, shuffle=False, batch_size=5, verbose=1)

# Evaluate the model for loss
model.evaluate(X_test, y_test)

# Make some predictions
predicted = model.predict(X_test)

# Evaluating the model
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predicted)))
print('R-squared :', metrics.r2_score(y_test, predicted))

# Recover the original prices instead of the scaled version
predicted_prices = y_train_scaler.inverse_transform(predicted)
real_prices = y_train_scaler.inverse_transform(y_test.reshape(-1, 1))

# Create a DataFrame of Real and Predicted values
stocks = pd.DataFrame({
    "Real": real_prices.ravel(),
    "Predicted": predicted_prices.ravel()
}, index = p_company_data.index[-len(real_prices): ]) 

print(stocks)

plt.figure(figsize=(20,10)) #plotting

plt.plot(stocks['Real'], label='Actual Data') #actual plot
plt.plot(stocks['Predicted'], label='Predicted Data') #predicted plot
plt.title('Time-Series Prediction')
plt.legend()
plt.show() 
#%%

y_pred = predicted
y_pred_price = y_train_scaler.inverse_transform(y_pred)
y_test_price = y_train_scaler.inverse_transform(y_test)
print(y_pred_price)



# %%
print(len(y_pred_price))
print(len(y_test_price))
# %%
l =[]
for i in range(len(y_test_price)-1):
    if y_test_price[i] - y_pred_price[i+1] >= 0:
        l.append(0)
    else:
        l.append(1)

print(l)
# %%
s =[]
for i in range(len(y_test_price)-1):
    if y_test_price[i] - y_test_price[i+1] >=0:
        s.append(0)
    else:
        s.append(1)
        
print(s)

# %%
t = []
for i in range(len(l)):
    if l[i] == s[i] :
        t.append(1)
        
    else:
        t.append(0)

print(t)
# %%
from collections import Counter

Counter(t)

# %%
a = Counter(t)
# %%
a[1] / (a[0]+a[1])
# %%
print('Dummy Classifier의 정확도: {0: .4f}'.format(accuracy_score(y_test, predicted)))