#%%
import pandas as pd
import numpy as  np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
#%%
model = xgb.XGBClassifier()

param_grid = {'booster' :['gbtree'],
              'silent' : [True],
              'max_depth': [5, 6, 8],
              'min_child_weight' : [1, 2, 3, 4, 5],
              'gamma' : [0, 1, 2, 3],
              'nthread' : [4],
              'colsample_bytree' :[0.5, 0.8],
              'colsample_bylevel' :[0.9],
              'n_estimators' :[50],
              'objective' :['binary:logistic'],
              'random_state':[2]}

#cv = KFold(n_splits=6, random_state=1)
#gcv = GridSearchCV(model, param_grid=param_grid, cv = cv, scoring='f1', n_jobs=4)

company_data = pd.read_csv('./data/result/셀트리온_test_1_result.csv', encoding="cp949")

p_company_data = company_data[['일자', '종가','sumPos', 'sumNeg', '대비', 
                               '등락률', '시가', '고가', '저가', '거래량']]
p_company_data = p_company_data.rename(columns = {"일자":"date","종가":"close", "대비":"vs", "등락률":"ratio",
                                                  "시가":"current_price", "고가":"high_price", 
                                                  "저가":"low_price", "거래량":"trade_volume"})
p_company_data = p_company_data.set_index('date')
p_company_data['Pct_change'] = p_company_data['close'].pct_change()
p_company_data.dropna(inplace = True)

#train_X, test_X, train_y, test_y = train_test_split(company_data, test_size=0.3, random_state=0)

#gcv.fit(train_X.values, train_y.values)

#%%
p_company_data

# %%
