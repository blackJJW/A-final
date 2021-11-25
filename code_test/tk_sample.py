from tkinter import *
from datetime import datetime
import DataFrameProcessing
import os
import os.path
import pathlib

#------------- 주식데이터 다운로드 path------------------------
current_path = str(pathlib.Path(__file__).parent.absolute())
download_path = current_path+'\data\stock'
#------------------------------------------------------------
#-----------정규분포 확률 변수----------
prob = 0.75
#--------------------------------------

'''
def now_time():
    ntime = datetime.now()
    btn.config(text=ntime)



win = Tk() # 창 생성

win.geometry("1500x1500") # 창 크기

win.title("Team-A fianl project") # 창 제목

win.option_add("*Font", "맑은고딕 15") # 폰트 설정

win.configure(bg = 'red') # 배경색

btn = Button(win) # 버튼 생성
btn.config(width =6, height=3) # 버튼 크기
btn.config(text='버튼') # 버튼 내용
btn.config(command=now_time) # 버튼 기능
btn.pack() # 버튼 배치

ent = Entry(win) # 입력창 생성
ent.get() # 입력창 내용 추출
ent.pack() # 입력창 배치'
ent.config(show="*") # 입력 문자 숨기기
ent.inset(0, "temp@temp.com") # 입력창 문자열 삽입
ent.delete(0, 3) # 0~2번째 문자열 삭제

def clear(event):
    ent.delete(0, len(ent.get()))

ent.ving("<Button-1>", clear) # 입력창 클릭시 명령

lab = Label(win)
lab.config(text='라벨')

lab.config(image=img)
img = PhotoImage(file='temp.png', master=win)
img = img.subsample(3) # 1/3로 축소


lab.pack()

win.mainloop() # 창 실행
'''

#stock = DataFrameProcessing.Get_Stock_DF('씨젠', '096530', '20200116', '20211110', download_path, prob)

#stock.download_stock_data()







































































