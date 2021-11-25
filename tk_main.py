import tkinter as tk
import os
import os.path
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import torch 
import torch.nn as nn

import DataFrameProcessing
import Gen_Senti_Process
import NewsArticleDFProcessing
import AnalysisNoun
import Gen_Stock_Pos_Neg

import Part_1_ML
import Part_1_Pre
import Part_1_LSTM
import Part_2_Reg

#------------- 주식데이터 다운로드 path------------------------
current_path = str(pathlib.Path(__file__).parent.absolute())
download_path = current_path+'\data\stock'
#------------------------------------------------------------
#-----------정규분포 확률 변수----------
prob = 0.75
#--------------------------------------
#----------LSTM 변수------------------------------------------------------------
input_size = 25  #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1  #number of stacked lstm layers
num_classes = 1 #number of output classes 

num_epochs = 90000 #1000 epochs
learning_rate = 0.00001 #0.001 lr
#---------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#---------------------------------------------------------------------------

def display_dir_path(dir_name): # dir 내 파일 목록 출력
    file_dir = "./data/"+dir_name 
    list_files = os.listdir(file_dir)

    a = []
    b = []
    file_list = {}
    for v in range(len(list_files)):
        a.append(v)

    for  i in  list_files :
        b.append(i)

    for x in range(len(list_files)):
        file_list[a[x]] = b[x]

    print('\n')
    print('*'*10+'데이터 파일 목록'+'*'*10)

    for y in range(len(file_list)) :
        print(y ,":", file_list[y])
        
    print('*'*35)
    
    return file_list

class Page(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
    def show(self):
        self.lift()

class Page1(Page):
   def __init__(self, *args, **kwargs):
       Page.__init__(self, *args, **kwargs)
       label_main = tk.Label(self, text="Team A Final Project : Stock Prediction", font="나눔고딕 25")
       label_main.pack(side="top", pady=150)
       
       label_team = tk.Label(self, text="이예지 정진우 조세연 한상백", font="나눔고딕 20")
       label_team.pack(side="top")

class Page2(Page):
   def __init__(self, *args, **kwargs):
       Page.__init__(self, *args, **kwargs)
           
       label_stock_main = tk.Label(self, text="KRX에서 주식 데이터를 가져와 정제", font="나눔고딕 20")
       label_stock_main.pack(side="top")
       
       label_company_name = tk.Label(self, text="회사 명 : ", font="나눔고딕 10")
       label_company_name.place(relx=0.03, rely =0.08)
       ent_company_name = tk.Entry(self)
       ent_company_name.place(relx=0.1, rely =0.08)
       
       label_company_code = tk.Label(self, text="회사 코드 : ", font="나눔고딕 10")
       label_company_code.place(relx=0.03, rely =0.12)
       ent_company_code = tk.Entry(self)
       ent_company_code.place(relx=0.1, rely =0.12)

       label_date_1 = tk.Label(self, text="시작 날짜 (ex : 20101023) : ", font="나눔고딕 10")
       label_date_1.place(relx=0.03, rely =0.16)
       ent_date_1 = tk.Entry(self)
       ent_date_1.place(relx=0.1, rely =0.20)

       label_date_2 = tk.Label(self, text="끝 날짜 (ex : 20201023) : ", font="나눔고딕 10")
       label_date_2.place(relx=0.03, rely =0.24)
       ent_date_2 = tk.Entry(self)
       ent_date_2.place(relx=0.1, rely =0.28)
       
       def download_stock():
        stock = DataFrameProcessing.Get_Stock_DF(ent_company_name.get(), ent_company_code.get(), 
                                                 ent_date_1.get(), ent_date_2.get(), download_path, prob)
        stock.download_stock_data()
              
       btn_progress_1 = tk.Button(self, text = "Progress", font="나눔고딕 10", command=download_stock)
       btn_progress_1.place(relx=0.17, rely = 0.35)
       
class Page3(Page):
   def __init__(self, *args, **kwargs):
       Page.__init__(self, *args, **kwargs)
       label_stock_main = tk.Label(self, text="여러 신문사에서 뉴스를 가져와 정제", font="나눔고딕 20")
       label_stock_main.pack(side="top")
       
       label_company_name = tk.Label(self, text="회사 명 : ", font="나눔고딕 10")
       label_company_name.place(relx=0.03, rely =0.08)
       ent_company_name = tk.Entry(self)
       ent_company_name.place(relx=0.1, rely =0.08)
       
       label_file_name = tk.Label(self, text="새로 저장할 파일 명 : ", font="나눔고딕 10")
       label_file_name.place(relx=0.03, rely =0.12)
       ent_file_name = tk.Entry(self)
       ent_file_name.place(relx=0.1, rely =0.16)
       
       label_maxpage = tk.Label(self, text="최대 페이지 수 : ", font="나눔고딕 10")
       label_maxpage.place(relx=0.03, rely =0.20)
       ent_maxpage = tk.Entry(self)
       ent_maxpage.place(relx=0.1, rely =0.24)
       
       def progress_news_article():
           NewsArticleDFProcessing.News_Act(ent_company_name.get(), ent_file_name.get(), ent_maxpage.get())
           NewsArticleDFProcessing.News_DF_Processing(ent_file_name.get())
       
       btn_progress_2 = tk.Button(self, text = "Progress", font="나눔고딕 10", command=progress_news_article)
       btn_progress_2.place(relx=0.17, rely = 0.3)
       
class Page4(Page):
   def __init__(self, *args, **kwargs):
       Page.__init__(self, *args, **kwargs)
       label_stock_main = tk.Label(self, text="senti 생성", font="나눔고딕 20")
       label_stock_main.pack(side="top")
       
       def show_stock_list():
           a = display_dir_path("stock/total_df")
           text1.delete(1.0,"end")
           
           for i in range(len(a), 0, -1):
            text1.insert(1.0, str(i-1)+':'+a[i-1]+'\n')
       
       label_stock_file_name = tk.Label(self, text="주식 파일 번호 : ", font="나눔고딕 10")
       label_stock_file_name.place(relx=0.03, rely =0.08)
       ent_stock_file_name = tk.Entry(self)
       ent_stock_file_name.place(relx=0.13, rely =0.08)
       btn_stock_file_list = tk.Button(self, text="목록", font="나눔고딕 10", command=show_stock_list)
       btn_stock_file_list.place(relx=0.28, rely=0.08)
              
       def show_news_list():
           a = display_dir_path("news/sorted_article")
           text1.delete(1.0,"end")
           
           for i in range(len(a), 0, -1):
            text1.insert(1.0, str(i-1)+':'+a[i-1]+'\n')

       label_news_file_name = tk.Label(self, text="뉴스 파일 번호 : ", font="나눔고딕 10")
       label_news_file_name.place(relx=0.03, rely =0.12)
       ent_news_file_name = tk.Entry(self)
       ent_news_file_name.place(relx=0.13, rely =0.12)
       btn_news_file_list = tk.Button(self, text="목록", font="나눔고딕 10", command=show_news_list)
       btn_news_file_list.place(relx=0.28, rely=0.12)
       
       def gen_senti():
           a = display_dir_path("stock/total_df")
           b = display_dir_path("news/sorted_article")
           
           Gen_Senti_Process.Gen_Senti(a[int(ent_stock_file_name.get())], b[int(ent_news_file_name.get())])
       
       btn_progress_2 = tk.Button(self, text = "Progress", font="나눔고딕 10", command=gen_senti)
       btn_progress_2.place(relx=0.17, rely = 0.16)
       
       text1 = tk.Text(self)
       text1.place(relx=0.35,rely=0.08, width=500,height=500)
       
class Page5(Page):
   def __init__(self, *args, **kwargs):
       Page.__init__(self, *args, **kwargs)
       label_stock_main = tk.Label(self, text="명사 분석", font="나눔고딕 20")
       label_stock_main.pack(side="top")
       #----------------------------------------------------------------------------------------------------
       label_noun = tk.Label(self, text="명사 추출", font="나눔고딕 15")
       label_noun.place(relx=0.1, rely=0.05)
       
       def show_senti_list():
            a = display_dir_path("dict")
            text2.delete(1.0,"end")

            for i in range(len(a), 0, -1):
             text2.insert(1.0, str(i-1)+':'+a[i-1]+'\n')

       label_senti_file_name = tk.Label(self, text="senti 파일 번호 : ", font="나눔고딕 10")
       label_senti_file_name.place(relx=0.03, rely =0.1)
       ent_senti_file_name = tk.Entry(self)
       ent_senti_file_name.place(relx=0.14, rely =0.1)
       btn_senti_file_list = tk.Button(self, text="목록", font="나눔고딕 10", command=show_senti_list)
       btn_senti_file_list.place(relx=0.29, rely=0.1)
       
       def ext_nouns():
           a = display_dir_path("dict")
           AnalysisNoun.Noun_Analysis_1(a[int(ent_senti_file_name.get())])
           
       btn_progress_noun = tk.Button(self, text = "Progress", font="나눔고딕 10", command=ext_nouns)
       btn_progress_noun.place(relx=0.255, rely = 0.15)
       #----------------------------------------------------------------------------------------------------
       label_noun_freq = tk.Label(self, text="명사 빈도수", font="나눔고딕 15")
       label_noun_freq.place(relx=0.1, rely=0.2)
       
       def show_noun_df_list():
            a = display_dir_path("nouns/noun_df")
            text2.delete(1.0,"end")

            for i in range(len(a), 0, -1):
             text2.insert(1.0, str(i-1)+':'+a[i-1]+'\n')

       label_noun_file_name = tk.Label(self, text="noun 파일 번호 : ", font="나눔고딕 10")
       label_noun_file_name.place(relx=0.03, rely =0.25)
       ent_noun_file_name = tk.Entry(self)
       ent_noun_file_name.place(relx=0.14, rely =0.25)
       btn_noun_file_list = tk.Button(self, text="목록", font="나눔고딕 10", command=show_noun_df_list)
       btn_noun_file_list.place(relx=0.29, rely=0.25)
       
       label_senti_file_name_2 = tk.Label(self, text="senti 파일 번호 : ", font="나눔고딕 10")
       label_senti_file_name_2.place(relx=0.03, rely =0.3)
       ent_senti_file_name_2 = tk.Entry(self)
       ent_senti_file_name_2.place(relx=0.14, rely =0.3)
       btn_senti_file_list_2 = tk.Button(self, text="목록", font="나눔고딕 10", command=show_senti_list)
       btn_senti_file_list_2.place(relx=0.29, rely=0.3)
       
       def freq_nouns():
           a = display_dir_path("dict")
           b = display_dir_path("nouns/noun_df")
           AnalysisNoun.Noun_Analysis_2(a[int(ent_senti_file_name_2.get())], b[int(ent_noun_file_name.get())])
           
       btn_progress_freq = tk.Button(self, text = "Progress", font="나눔고딕 10", command=freq_nouns)
       btn_progress_freq.place(relx=0.255, rely = 0.35)
       #----------------------------------------------------------------------------------------------------
       label_noun_pn = tk.Label(self, text="명사 긍부정 지수", font="나눔고딕 15")
       label_noun_pn.place(relx=0.1, rely=0.4)
              
       def show_noun_freq_list():
            a = display_dir_path("nouns/nouns_freq")
            text2.delete(1.0,"end")
            for i in range(len(a), 0, -1):
             text2.insert(1.0, str(i-1)+':'+a[i-1]+'\n')
             
       label_freq_file_name = tk.Label(self, text="freq 파일 번호 : ", font="나눔고딕 10")
       label_freq_file_name.place(relx=0.03, rely =0.45)
       ent_freq_file_name = tk.Entry(self)
       ent_freq_file_name.place(relx=0.14, rely =0.45)
       btn_freq_file_list = tk.Button(self, text="목록", font="나눔고딕 10", command=show_noun_freq_list)
       btn_freq_file_list.place(relx=0.29, rely=0.45)
              
       label_senti_file_name_3 = tk.Label(self, text="senti 파일 번호 : ", font="나눔고딕 10")
       label_senti_file_name_3.place(relx=0.03, rely =0.5)
       ent_senti_file_name_3 = tk.Entry(self)
       ent_senti_file_name_3.place(relx=0.14, rely =0.5)
       btn_senti_file_list_3 = tk.Button(self, text="목록", font="나눔고딕 10", command=show_senti_list)
       btn_senti_file_list_3.place(relx=0.29, rely=0.5)
              
       def np_nouns():
           a = display_dir_path("dict")
           b = display_dir_path("nouns/nouns_freq")
           AnalysisNoun.Pos_Neg_Points(a[int(ent_senti_file_name_3.get())], b[int(ent_freq_file_name.get())])
           
       btn_progress_np = tk.Button(self, text = "Progress", font="나눔고딕 10", command=np_nouns)
       btn_progress_np.place(relx=0.255, rely = 0.55)
       #----------------------------------------------------------------------------------------------------
       
       text2 = tk.Text(self)
       text2.place(relx=0.35,rely=0.08, width=500,height=500)
       
class Page6(Page):
   def __init__(self, *args, **kwargs):
       Page.__init__(self, *args, **kwargs)
       label_stock_pn_main = tk.Label(self, text="주식 + 뉴스 감성분석", font="나눔고딕 20")
       label_stock_pn_main.pack(side="top")
       #----------------------------------------------------------------------------------------------------
       label_noun = tk.Label(self, text="Stock Pos/Neg 생성", font="나눔고딕 15")
       label_noun.place(relx=0.1, rely=0.05)
              
       def show_result_list():
            a = display_dir_path("result")
            text3.delete(1.0,"end")
            for i in range(len(a), 0, -1):
             text3.insert(1.0, str(i-1)+':'+a[i-1]+'\n')
             
       label_result_file_name = tk.Label(self, text="result 파일 번호 : ", font="나눔고딕 10")
       label_result_file_name.place(relx=0.03, rely =0.1)
       ent_result_file_name = tk.Entry(self)
       ent_result_file_name.place(relx=0.14, rely =0.1)
       btn_result_file_list = tk.Button(self, text="목록", font="나눔고딕 10", command=show_result_list)
       btn_result_file_list.place(relx=0.29, rely=0.1)
              
       def show_stock_list():
            a = display_dir_path("stock")
            text3.delete(1.0,"end")
            for i in range(len(a), 0, -1):
             text3.insert(1.0, str(i-1)+':'+a[i-1]+'\n')
             
       label_stock_file_name = tk.Label(self, text="stock 파일 번호 : ", font="나눔고딕 10")
       label_stock_file_name.place(relx=0.03, rely =0.15)
       ent_stock_file_name = tk.Entry(self)
       ent_stock_file_name.place(relx=0.14, rely =0.15)
       btn_stock_file_list = tk.Button(self, text="목록", font="나눔고딕 10", command=show_stock_list)
       btn_stock_file_list.place(relx=0.29, rely=0.15)
       
       def stock_pn():
         a = display_dir_path("result")
         b = display_dir_path("stock")
         Gen_Stock_Pos_Neg.Refine_Result(a[int(ent_result_file_name.get())], b[int(ent_stock_file_name.get())])
    
       btn_progress_np = tk.Button(self, text = "Progress", font="나눔고딕 10", command=stock_pn)
       btn_progress_np.place(relx=0.255, rely = 0.20)
       
       text3 = tk.Text(self)
       text3.place(relx=0.35,rely=0.08, width=500,height=500)
    
class Page7(Page):
   def __init__(self, *args, **kwargs):
       Page.__init__(self, *args, **kwargs)
       label_part_1_ml_main = tk.Label(self, text="CV accuracy / Income rate", font="나눔고딕 20")
       label_part_1_ml_main.pack(side="top")
       #-------------------------------------------------------------------------------------------------------------
       label_noun = tk.Label(self, text="CV / Income rate 생성", font="나눔고딕 15")
       label_noun.place(relx=0.1, rely=0.05)
              
       def show_stock_list():
            a = display_dir_path("stock")
            text4.delete(1.0,"end")
            for i in range(len(a), 0, -1):
             text4.insert(1.0, str(i-1)+':'+a[i-1]+'\n')
             
       label_stock_file_name = tk.Label(self, text="stock 파일 번호 : ", font="나눔고딕 10")
       label_stock_file_name.place(relx=0.03, rely =0.1)
       ent_stock_file_name = tk.Entry(self)
       ent_stock_file_name.place(relx=0.14, rely =0.1)
       btn_stock_file_list = tk.Button(self, text="목록", font="나눔고딕 10", command=show_stock_list)
       btn_stock_file_list.place(relx=0.29, rely=0.1)
       
       def stock_ML_1():
         b = display_dir_path("stock")
         Part_1_ML.Set_Data(b[int(ent_stock_file_name.get())])
         Part_1_ML.ML_Part_1(b[int(ent_stock_file_name.get())])
   
       btn_progress_np = tk.Button(self, text = "Progress", font="나눔고딕 10", command=stock_ML_1)
       btn_progress_np.place(relx=0.255, rely = 0.15)
       
       def view_cv():
           str = ""
           b = display_dir_path("stock")
           cv_report = open('./data/report/cv_acc/'+b[int(ent_stock_file_name.get())]+'_cv_accuracy_report.txt', 'r')
           str = cv_report.readlines()
           text4.delete(1.0, 'end')
           text4.insert(1.0, str)
           cv_report.close()
       
       btn_cv = tk.Button(self, text = "CV", font="나눔고딕 10", command=view_cv)
       btn_cv.place(relx=0.235, rely = 0.2)
       
       def view_income():
           str = ""
           b = display_dir_path("stock")
           income_report = open('./data/report/income_rate/'+b[int(ent_stock_file_name.get())]+'_income_rate_report.txt', 'r')
           str = income_report.readlines()
           text4.delete(1.0, 'end')
           text4.insert(1.0, str)
           income_report.close()
           
       btn_income = tk.Button(self, text = "income", font="나눔고딕 10", command=view_income)
       btn_income.place(relx=0.266, rely = 0.2)
       
       text4 = tk.Text(self)
       text4.place(relx=0.35,rely=0.08, width=500,height=500)
       
class Page8(Page):
   def __init__(self, *args, **kwargs):
       Page.__init__(self, *args, **kwargs)
       label_part_1_LSTM_main = tk.Label(self, text="LSTM", font="나눔고딕 20")
       label_part_1_LSTM_main.pack(side="top")
       #-------------------------------------------------------------------------------------------------------------
       def show_stock_pn_list():
            a = display_dir_path("stock_pos_neg")
            text5.delete(1.0,"end")
            for i in range(len(a), 0, -1):
             text5.insert(1.0, str(i-1)+':'+a[i-1]+'\n')
             
       label_stock_np_file_name = tk.Label(self, text="stock p/n 파일 번호 : ", font="나눔고딕 10")
       label_stock_np_file_name.place(relx=0.03, rely =0.1)
       ent_stock_np_file_name = tk.Entry(self)
       ent_stock_np_file_name.place(relx=0.14, rely =0.15)
       btn_stock_np_file_list = tk.Button(self, text="목록", font="나눔고딕 10", command=show_stock_pn_list)
       btn_stock_np_file_list.place(relx=0.29, rely=0.15)
       
       def lstm():
         a = display_dir_path("stock_pos_neg")
         df = Part_1_Pre.Refine_DF(a[int(ent_stock_np_file_name.get())])
         df_1 = df.refining_df()
         df_2 = Part_1_Pre.Extra_Features(df_1)
         df_3 = df_2.return_df()
         df_4 = Part_1_Pre.ready_data_set(df_3)
         X, y = df_4.split_X_y()
         prep_y = Part_1_LSTM.Scaler(X, y)
         
         X_train, X_test, y_train, y_test, length = prep_y.prep()
         
         shape1 = "Traing Shape : "+str(X_train.shape)+", "+str(y_train.shape)+'\n'
         shape2 = "Testing Shape : "+str(X_test.shape)+", "+str(y_test.shape)+'\n'
         
         text5.delete(1.0, 'end')
         text5.insert(1.0, shape1)
         text5.insert(1.0, shape2)
         
         lstm1 = Part_1_LSTM.LSTM1(num_classes, input_size, hidden_size, num_layers, X_train.shape[1]).to(device)
         
         Part_1_LSTM.LSTM_predict(lstm1, num_epochs, learning_rate, X_train, y_train, length, df_3)
        
       btn_progress_np = tk.Button(self, text = "Progress", font="나눔고딕 10", command=lstm)
       btn_progress_np.place(relx=0.255, rely = 0.20)

       text5 = tk.Text(self)
       text5.place(relx=0.35,rely=0.08, width=500,height=500)
       
class Page9(Page):
   def __init__(self, *args, **kwargs):
       Page.__init__(self, *args, **kwargs)
       label_part_2_main = tk.Label(self, text="Regressor", font="나눔고딕 20")
       label_part_2_main.pack(side="top")
       #-------------------------------------------------------------------------------------------------------------
       def show_stock_pn_list():
         a = display_dir_path("stock_pos_neg")
         text6.delete(1.0,"end")
         for i in range(len(a), 0, -1):
          text6.insert(1.0, str(i-1)+':'+a[i-1]+'\n')
       
       label_stock_np_file_name = tk.Label(self, text="stock p/n 파일 번호 : ", font="나눔고딕 10")
       label_stock_np_file_name.place(relx=0.03, rely =0.1)
       ent_stock_np_file_name = tk.Entry(self)
       ent_stock_np_file_name.place(relx=0.14, rely =0.15)
       btn_stock_np_file_list = tk.Button(self, text="목록", font="나눔고딕 10", command=show_stock_pn_list)
       btn_stock_np_file_list.place(relx=0.29, rely=0.15)
       
       def rf_regressor():
         b = display_dir_path("stock_pos_neg")
         a = Part_2_Reg.Prep_Regressor(b[int(ent_stock_np_file_name.get())])
         c = a.refining_df()
         d = Part_2_Reg.RF_Regressor(c)
         rmse, r_sqr = d.evaluate_model()
         
         text6.delete(1.0, 'end')
         text6.insert(1.0, rmse)
         text6.insert(1.0, r_sqr)
       
       btn_progress_np = tk.Button(self, text = "RF", font="나눔고딕 10", command=rf_regressor)
       btn_progress_np.place(relx=0.255, rely = 0.2)
       
       def xgb_regressor():
         b = display_dir_path("stock_pos_neg")
         a = Part_2_Reg.Prep_Regressor(b[int(ent_stock_np_file_name.get())])
         c = a.refining_df()
         d = Part_2_Reg.XGBoost_Regressor(c)
         rmse, r_sqr = d.evaluate_model()
         
         text6.delete(1.0, 'end')
         text6.insert(1.0, rmse)
         text6.insert(1.0, r_sqr)
       
       btn_progress_np = tk.Button(self, text = "XGB", font="나눔고딕 10", command=xgb_regressor)
       btn_progress_np.place(relx=0.2, rely = 0.2)
       text6 = tk.Text(self)
       text6.place(relx=0.35,rely=0.08, width=500,height=500)
       
class MainView(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        p1 = Page1(self)
        p2 = Page2(self)
        p3 = Page3(self)
        p4 = Page4(self)
        p5 = Page5(self)
        p6 = Page6(self)
        p7 = Page7(self)
        p8 = Page8(self)
        p9 = Page9(self)
        
        buttonframe = tk.Frame(self)
        container = tk.Frame(self)
        buttonframe.pack(side="top", fill="x", expand=False)
        container.pack(side="top", fill="both", expand=True)

        p1.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p2.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p3.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p4.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p5.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p6.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p7.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p8.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p9.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

        b1 = tk.Button(buttonframe, text="Main", command=p1.show)
        b2 = tk.Button(buttonframe, text="Stock Data", command=p2.show)
        b3 = tk.Button(buttonframe, text="News Articles", command=p3.show)
        b4 = tk.Button(buttonframe, text="Senti", command=p4.show)
        b5 = tk.Button(buttonframe, text="Nouns", command=p5.show)
        b6 = tk.Button(buttonframe, text="Stock P/N", command=p6.show)
        b7 = tk.Button(buttonframe, text="CV / Income", command=p7.show)
        b8 = tk.Button(buttonframe, text="LSTM", command=p8.show)
        b9 = tk.Button(buttonframe, text="Regressor", command=p9.show)

        b1.pack(side="left")
        b2.pack(side="left")
        b3.pack(side="left")
        b4.pack(side="left")
        b5.pack(side="left")
        b6.pack(side="left")
        b7.pack(side="left")
        b8.pack(side="left")
        b9.pack(side="left")
        
        p1.show()

if __name__ == "__main__":
    root = tk.Tk()
    main = MainView(root)
    main.pack(side="top", fill="both", expand=True)
    root.wm_geometry("1000x700")

    root.mainloop()