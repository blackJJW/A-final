import datetime
import numpy as np
import pandas_datareader as pdr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from flask import Flask, jsonify

import config
from views import main_views


app = Flask(__name__)
app.config.from_object(config)


#blueprint
app.register_blueprint(main_views.bp)



# Predict
@app.route('/<string:stock>/<float:test>/<int:days>')
def pred(stock,test,days):
    dt=pdr.DataReader(stock,'yahoo','2021-1-1',datetime.datetime.now(),api_key='911ee28d70118f9cea5a84d2b8f1436fa32d3116')
    print(dt)
    dt.reset_index(inplace=True)
    dt.set_index("Date",inplace=True)
    dt=dt[['High','Low','Open','Close','Volume','Adj Close']]
    no_days=int(days)
    #today = df_part.iloc[-1]
    #company_stock_1_df = df_part.iloc[:-1]
    #target = company_stock_1_df['fluctuation']
    #company_stock_1_df = company_stock_1_df.drop('fluctuation', axis = 1)
    #train, test, train_target, test_target = train_test_split(company_stock_1_df, target, test_size = 0.3, shuffle=False ) 
    dt['new_close']=dt['Adj Close'].shift(-no_days)
    x=dt.drop(['Adj Close','new_close'],axis=1)
    y=dt['new_close'].dropna()
    x1=x[:-no_days]
    x2=x[-no_days:]
    scaler=StandardScaler()
    scaler.fit(x1)
    x1=scaler.transform(x1)
    x2=scaler.transform(x2)
    x_tr,x_ts,y_tr,y_ts=train_test_split(x1,y,test_size=0.25)
    algo=LinearRegression()
    algo.fit(x_tr,y_tr)
    acu=algo.score(x_ts,y_ts)
    prd=algo.predict(x2)
    result={'stock':stock,'test_size':test,'no_of_days':days,'accuracy':acu,'prediction':list(prd)}
    return jsonify(result)

        
    

if __name__ == "__main__":
    app.run(debug=True)
