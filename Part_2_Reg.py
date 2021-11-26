#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

class Prep_Regressor:
    def __init__(self, stock_pos_neg):
        print("Part_2_Reg - Prep_Regressor Start")
        self.stock_pos_neg = stock_pos_neg
    
    def refining_df(self):
        print("Part_2_Reg - Prep_Regressor - refining_df Start")
        print("----- reading csv Start -----")
        company_data = pd.read_csv('./data/stock_pos_neg/'+self.stock_pos_neg, encoding="cp949")
        print("----- reading csv Done -----")
        company_data.dropna(inplace = True)

        print("----- setting df Start -----")
        p_company_data = company_data[['일자', '종가', 'sumPos', 'sumNeg']]
        p_company_data = p_company_data.rename(columns = {"일자":"date","종가":"close"})
        p_company_data = p_company_data.set_index('date')
        p_company_data['Pct_change'] = p_company_data['close'].pct_change()
        p_company_data.dropna(inplace = True)
        print("----- setting df Done -----")
        print("Part_2_Reg - Prep_Regressor - refining_df Done")
        print("Part_2_Reg - Prep_Regressor  Done")
        return p_company_data
    
class RF_Regressor:
    def __init__(self, f_company_data):
        print("Part_2_Reg - RF_Regressor  Start")
        self.f_company_data = f_company_data
        
        self.gen_X_y()
        self.gen_train_test_sets()
        self.fit_sets()
        self.set_model()
        self.fit_model()
        self.get_prediction()
        self.evaluate_model()
        self.inverse_predict()
        self.gen_stocks_df()
        self.show_plot()
        
    def gen_X_y(self): 
        print("Part_2_Reg - RF_Regressor - gen_X_y  Start")   
        def window_data(df, window, feature_col_number1, feature_col_number2, feature_col_number3, target_col_number):
            # Create empty lists "X_close", "X_polarity", "X_volume" and y
            X_close = []
            X_sumPos = []
            X_sumNeg = []
            y = []
            for i in range(len(df) - window):

                # Get close, ts_sumPos, ts_sumNeg, and target in the loop
                close = df.iloc[i:(i + window), feature_col_number1]
                ts_sumPos = df.iloc[i:(i + window), feature_col_number2]
                ts_sumNeg = df.iloc[i:(i + window), feature_col_number3]
                target = df.iloc[(i + window), target_col_number]

                # Append values in the lists
                X_close.append(close)
                X_sumPos.append(ts_sumPos)
                X_sumNeg.append(ts_sumNeg)
                y.append(target)

            return np.hstack((X_close,X_sumPos,X_sumNeg)), np.array(y).reshape(-1, 1)

        # Predict Closing Prices using a 3 day window of previous closing prices
        window_size = 3

        # Column index 0 is the `Close` column
        # Column index 1 is the `sumPos` column
        # Column index 2 is the `sumNeg` column
        feature_col_number1 = 0
        feature_col_number2 = 1
        feature_col_number3 = 2
        target_col_number = 0
        print("----- creating window datas Start -----")
        self.X, self.y = window_data(self.f_company_data, window_size, feature_col_number1, 
                           feature_col_number2, feature_col_number3, target_col_number)
        print("----- creating window datas Done -----")
        print("Part_2_Reg - RF_Regressor - gen_X_y  Done")

    def gen_train_test_sets(self):
        print("Part_2_Reg - RF_Regressor - gen_train_test_sets  Start")
        print("----- spliting Start -----")
        # Use 70% of the data for training and the remainder for testing
        X_split = int(0.7 * len(self.X))
        y_split = int(0.7 * len(self.y))

        self.X_train = self.X[: X_split]
        self.X_test = self.X[X_split:]
        self.y_train = self.y[: y_split]
        self.y_test = self.y[y_split:]
        print("----- spliting Done -----")
        print("Part_2_Reg - RF_Regressor - gen_train_test_sets  Done")
        
    def fit_sets(self):
        print("Part_2_Reg - RF_Regressor - fit_sets  Start")
        # Scaling Data with MinMaxScaler
        # value 0 ~ 1
        # scale both features and target sets

        # Use the MinMaxScaler to scale data between 0 and 1.
        self.x_train_scaler = MinMaxScaler()
        self.x_test_scaler = MinMaxScaler()
        self.y_train_scaler = MinMaxScaler()
        self.y_test_scaler = MinMaxScaler()

        print("----- fit Start -----")
        # Fit the scaler for the Training Data
        self.x_train_scaler.fit(self.X_train)
        self.y_train_scaler.fit(self.y_train)

        # Scale the training data
        self.X_train_t = self.x_train_scaler.transform(self.X_train)
        self.y_train_t = self.y_train_scaler.transform(self.y_train)

        # Fit the scaler for the Testing Data
        self.x_test_scaler.fit(self.X_test)
        self.y_test_scaler.fit(self.y_test)

        # Scale the y_test data
        self.X_test_result = self.x_test_scaler.transform(self.X_test)
        self.y_test_result = self.y_test_scaler.transform(self.y_test)
        print("----- fit Done -----")
        print("Part_2_Reg - RF_Regressor - fit_sets  Done")
        
    def set_model(self):
        print("Part_2_Reg - RF_Regressor - set_model  Start")
        # Create the Random Forest regressor instance
        self.model = RandomForestRegressor(n_estimators=1000, max_depth=2, bootstrap=False, min_samples_leaf=1)
        print("Part_2_Reg - RF_Regressor - set_model  Done")

    def fit_model(self):
        print("Part_2_Reg - RF_Regressor - fit_model  Start")
        # Fit the model
        self.model.fit(self.X_train_t, self.y_train_t.ravel())
        print("Part_2_Reg - RF_Regressor - fit_model  Done")

    def get_prediction(self):
        print("Part_2_Reg - RF_Regressor - get_prediction  Start")
        # Make some predictions
        self.predicted = self.model.predict(self.X_test_result)
        print("Part_2_Reg - RF_Regressor - get_prediction  Done")

    def evaluate_model(self):
        print("Part_2_Reg - RF_Regressor - evaluate_model  Start")
        # Evaluating the model
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(self.y_test_result, self.predicted)))
        print('R-squared :', metrics.r2_score(self.y_test_result, self.predicted))

        rmse = 'RF - Root Mean Squared Error :'+str(np.sqrt(metrics.mean_squared_error(self.y_test_result, self.predicted)))+'\n'
        r_sqr = 'RF - R-squared :'+str(metrics.r2_score(self.y_test_result, self.predicted))+'\n'
        
        print("Part_2_Reg - RF_Regressor - evaluate_model  Done")
        return rmse, r_sqr
    
    def inverse_predict(self):
        print("Part_2_Reg - RF_Regressor - inverse_predict  Start")
        # Recover the original prices instead of the scaled version
        self.predicted_prices = self.y_test_scaler.inverse_transform(self.predicted.reshape(-1, 1))
        self.real_prices = self.y_test_scaler.inverse_transform(self.y_test_result.reshape(-1, 1))
        print("Part_2_Reg - RF_Regressor - inverse_predict  Done")

    def gen_stocks_df(self):
        print("Part_2_Reg - RF_Regressor - gen_stocks_df  Start")
        # Create a DataFrame of Real and Predicted values
        self.stocks = pd.DataFrame({
                    "Real": self.real_prices.ravel(),
                    "Predicted": self.predicted_prices.ravel()
                    }, index = self.f_company_data.index[-len(self.real_prices): ]) 
        print("Part_2_Reg - RF_Regressor - gen_stocks_df  Done")

    def show_plot(self):
        print("Part_2_Reg - RF_Regressor - show_plot  Start")
        plt.figure(figsize=(20,10)) #plotting
        plt.plot(self.stocks['Real'], label='Actual Data') #actual plot
        plt.plot(self.stocks['Predicted'], label='Predicted Data') #predicted plot
        plt.title('RF-Regressor Time-Series Prediction')
        plt.legend()
        plt.show() 
        print("Part_2_Reg - RF_Regressor - show_plot  Done")
        print("Part_2_Reg - RF_Regressor  Done")

class XGBoost_Regressor:
    def __init__(self, f_company_data):
        print("Part_2_Reg - XGBoost_Regressor  Start")
        self.f_company_data = f_company_data
        
        self.gen_X_y()
        self.gen_train_test_sets()
        self.fit_sets()
        self.set_model()
        self.fit_model()
        self.get_prediction()
        self.evaluate_model()
        self.inverse_predict()
        self.gen_stocks_df()
        self.show_plot()
        
    def gen_X_y(self): 
        print("Part_2_Reg - XGBoost_Regressor - gen_X_y  Start")   
        def window_data(df, window, feature_col_number1, feature_col_number2, feature_col_number3, target_col_number):
            # Create empty lists "X_close", "X_polarity", "X_volume" and y
            X_close = []
            X_sumPos = []
            X_sumNeg = []
            y = []
            for i in range(len(df) - window):

                # Get close, ts_sumPos, ts_sumNeg, and target in the loop
                close = df.iloc[i:(i + window), feature_col_number1]
                ts_sumPos = df.iloc[i:(i + window), feature_col_number2]
                ts_sumNeg = df.iloc[i:(i + window), feature_col_number3]
                target = df.iloc[(i + window), target_col_number]

                # Append values in the lists
                X_close.append(close)
                X_sumPos.append(ts_sumPos)
                X_sumNeg.append(ts_sumNeg)
                y.append(target)

            return np.hstack((X_close,X_sumPos,X_sumNeg)), np.array(y).reshape(-1, 1)

        # Predict Closing Prices using a 3 day window of previous closing prices
        window_size = 3

        # Column index 0 is the `Close` column
        # Column index 1 is the `sumPos` column
        # Column index 2 is the `sumNeg` column
        feature_col_number1 = 0
        feature_col_number2 = 1
        feature_col_number3 = 2
        target_col_number = 0
        print("----- creating window datas Start -----")
        self.X, self.y = window_data(self.f_company_data, window_size, feature_col_number1, 
                           feature_col_number2, feature_col_number3, target_col_number)
        print("----- creating window datas Done -----")
        print("Part_2_Reg - XGBoost_Regressor - gen_X_y  Done")
        
    def gen_train_test_sets(self):
        print("Part_2_Reg - XGBoost_Regressor - gen_train_test_sets  Start")
        print("----- spliting Start -----")
        # Use 70% of the data for training and the remainder for testing
        X_split = int(0.7 * len(self.X))
        y_split = int(0.7 * len(self.y))

        self.X_train = self.X[: X_split]
        self.X_test = self.X[X_split:]
        self.y_train = self.y[: y_split]
        self.y_test = self.y[y_split:]
        print("----- spliting Done -----")
        print("Part_2_Reg - XGBoost_Regressor - gen_train_test_sets  Done")
        
    def fit_sets(self):
        print("Part_2_Reg - XGBoost_Regressor - fit_sets  Start")

        # Scaling Data with MinMaxScaler
        # value 0 ~ 1
        # scale both features and target sets

        # Use the MinMaxScaler to scale data between 0 and 1.
        self.x_train_scaler = MinMaxScaler()
        self.x_test_scaler = MinMaxScaler()
        self.y_train_scaler = MinMaxScaler()
        self.y_test_scaler = MinMaxScaler()
        print("----- fit Start -----")
        # Fit the scaler for the Training Data
        self.x_train_scaler.fit(self.X_train)
        self.y_train_scaler.fit(self.y_train)

        # Scale the training data
        self.X_train_t = self.x_train_scaler.transform(self.X_train)
        self.y_train_t = self.y_train_scaler.transform(self.y_train)

        # Fit the scaler for the Testing Data
        self.x_test_scaler.fit(self.X_test)
        self.y_test_scaler.fit(self.y_test)

        # Scale the y_test data
        self.X_test_result = self.x_test_scaler.transform(self.X_test)
        self.y_test_result = self.y_test_scaler.transform(self.y_test)
        print("----- fit Done -----")
        print("Part_2_Reg - XGBoost_Regressor - fit_sets  Done")
        
    def set_model(self):
        print("Part_2_Reg - XGBoost_Regressor - set_model  Start")
        # Create the Random Forest regressor instance
        self.model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        print("Part_2_Reg - XGBoost_Regressor - set_model  Done")

    def fit_model(self):
        print("Part_2_Reg - XGBoost_Regressor - fit_model  Start")
        # Fit the model
        self.model.fit(self.X_train_t, self.y_train_t.ravel())
        print("Part_2_Reg - XGBoost_Regressor - fit_model  Done")

    def get_prediction(self):
        print("Part_2_Reg - XGBoost_Regressor - get_prediction  Start")
        # Make some predictions
        self.predicted = self.model.predict(self.X_test_result)
        print("Part_2_Reg - XGBoost_Regressor - get_prediction  Done")

    def evaluate_model(self):
        print("Part_2_Reg - XGBoost_Regressor - evaluate_model  Start")
        # Evaluating the model
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(self.y_test_result, self.predicted)))
        print('R-squared :', metrics.r2_score(self.y_test_result, self.predicted))
        
        rmse = 'XGB - Root Mean Squared Error :'+str(np.sqrt(metrics.mean_squared_error(self.y_test_result, self.predicted)))+'\n'
        r_sqr = 'XGB - R-squared :'+str(metrics.r2_score(self.y_test_result, self.predicted))+'\n'
        
        print("Part_2_Reg - XGBoost_Regressor - evaluate_model  Done")
        return rmse, r_sqr
    
    def inverse_predict(self):
        print("Part_2_Reg - XGBoost_Regressor - inverse_predict  Start")
        # Recover the original prices instead of the scaled version
        self.predicted_prices = self.y_test_scaler.inverse_transform(self.predicted.reshape(-1, 1))
        self.real_prices = self.y_test_scaler.inverse_transform(self.y_test_result.reshape(-1, 1))
        print("Part_2_Reg - XGBoost_Regressor - inverse_predict  Done")
        
    def gen_stocks_df(self):
        print("Part_2_Reg - XGBoost_Regressor - gen_stocks_df  Start")
        # Create a DataFrame of Real and Predicted values
        self.stocks = pd.DataFrame({
                    "Real": self.real_prices.ravel(),
                    "Predicted": self.predicted_prices.ravel()
                    }, index = self.f_company_data.index[-len(self.real_prices): ]) 
        print("Part_2_Reg - XGBoost_Regressor - gen_stocks_df  Done")

    def show_plot(self):
        print("Part_2_Reg - XGBoost_Regressor - show_plot  Start")
        plt.figure(figsize=(20,10)) #plotting
        plt.plot(self.stocks['Real'], label='Actual Data') #actual plot
        plt.plot(self.stocks['Predicted'], label='Predicted Data') #predicted plot
        plt.title('XGBoost-Regressor Time-Series Prediction')
        plt.legend()
        plt.show()
        print("Part_2_Reg - XGBoost_Regressor - show_plot  Done")
        print("Part_2_Reg - XGBoost_Regressor Done")



        




















