from tkinter import *
import tkinter as tk
import tkinter.font as font
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


#%%
import datetime
from datetime import datetime
from datetime import timedelta
from os import X_OK
from keras.utils.generic_utils import default
from konlpy.tag import Kkma
import json
from collections import Counter
import time
import re
from tqdm import tqdm 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
import math
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from keras.layers import Embedding, Dense, LSTM 
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
import keras
from keras.layers import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr 
import matplotlib.pyplot as plt 

# Importing Classifier Modules
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#cross_validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn import datasets
from sklearn import metrics

import datetime
from datetime import datetime

import torch 
import torch.nn as nn 
from torch.autograd import Variable 

import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as Func


root=Tk()
root.geometry("1246x898")
root.title("Stock_Prediction")


#background
bg= PhotoImage(file="images.png",master=root)
#캔버스 생성
canvas1 = Canvas(root, width=1246, height=898)
 #캔버스를 창 너비에 맞춰 동적으로 크기를 조정한다.
canvas1.pack(fill="both", expand= True)
canvas1.create_image( 0, 0, image = bg, anchor = "nw")


fontLabel1 = font.Font(family="굴림", size=40)
canvas1.create_text( 400, 30, fill="#FFFFFF", text = "Stock_Prediction", font = fontLabel1)


Textbox1Font = font.Font(size=20)
strTextbox1 = StringVar()
textbox1 = Entry(root, textvariable=strTextbox1)
textbox1.place(x=80,y=80, width=800,height=50)
textbox1['font'] = Textbox1Font

text1 = Text(root)
text1.place(x=80,y=160, width=800,height=400)

"""
현재 주가 확인하는 코드
"""
def get_bs_obj(company_code):
    url = "https://finance.naver.com/item/main.nhn?code=" + company_code
    result = requests.get(url)
    bs_obj = BeautifulSoup(result.content, "html.parser")
    return bs_obj

#bs_obj를 받아서 price를 return
def get_price(company_code):
    bs_obj = get_bs_obj(company_code)
    no_today = bs_obj.find("p", {"class": "no_today"})
    blind_now = no_today.find("span", {"class": "blind"})
    return blind_now.text

#company_codes = ["005930","000660"]
def get_Price_from_ent():
    f = ent.get();
 
    lbl_2.configure(text = str(get_price(f)))
    return



# lbl_1 = Label(root, text = "Company Number")
# ent = Entry(root)
# lbl_2 = Label(root, text = "0")
# btn = Button(root, text = "Search", command = get_Price_from_ent)
 
# lbl_1.place(x = 20, y = 30)
# ent.place(x = 150, y = 30)
# btn.place(x = 150, y = 70)
# lbl_2.place(x = 150, y = 120)
 
 
"""
모델 불러오기
 
"""
# # Class Section
# class Window(object):
#   def __init__(self):
#     self.tk = Tk()
#     self.tk.title("Demo Window")
#     self.tk.geometry("500x500")
#     self.tk.resizable(0, 0)
#     self.tk.mainloop()
#     self.widgets()
#     pass
#   def widgets(self):
#     lbl = Label(self.tk, text="Hello")
#     lbl.pack()
#     pass
# win = Window()


"""
버튼설정하기

"""
strBtn1 = StringVar()
def funcBtn1():
    tmp = strTextbox1.get()
    text1.delete(1.0,"end")
    text1.insert(1.0, tmp)
    print(tmp)
    
strBtn1 = "1단계"
myButton1Font = font.Font(size=30)
button1 = Button( root, text = strBtn1, command = funcBtn1, font ='바탕', bd = 20, fg = '#FFFFFF', bg ="#6799FF", relief = "groove")
button1['font'] = myButton1Font
button1.place(x=900,y=100, width=240,height=120)

def funcBtn2():
    tmp = strTextbox1.get()
    text1.delete(1.0,"end")
    tmp = str(tmp)
    tmp_2 = get_price(tmp)
    text1.insert(1.0, tmp_2)
    print(tmp)

strBtn2 = "현재주가"
myButton2Font = font.Font(size=30)
button2 = Button( root, text = strBtn2, command = funcBtn2, font ='바탕', bd = 20, fg = '#FFFFFF', bg ="#6799FF", relief = "groove")
button2['font'] = myButton2Font 
button2.place(x=900,y=300, width=240,height=120)

#버튼 3번째
def funcBtn3():
    tmp = strTextbox1.get()
    text1.delete(1.0,"end")
    tmp = str(tmp)
    tmp_2 = get_price(tmp)
    text1.insert(1.0, tmp_2)
    print(tmp)
    

strBtn3 = "3단계"
myButton3Font = font.Font(size=30)
button3 = Button( root, text = strBtn3, command = funcBtn3, font ='바탕', bd = 20, fg = '#FFFFFF', bg ="#6799FF", relief = "groove")
button3['font'] = myButton3Font 
button3.place(x=900,y=500, width=240,height=120)

root.mainloop()