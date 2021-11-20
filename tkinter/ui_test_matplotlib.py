import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
x=np.arange(1, 10, 1)
y=2*x**2
fig = Figure(figsize=(10, 7), dpi=100)  #그리프 그릴 창 생성
fig.add_subplot(1,1,1).plot(x, y)#창에 그래프 하나 추가
window = tk.Tk()  #Tk 객체 생성. 기본 윈도우 객체
canvas = FigureCanvasTkAgg(fig, master=window)
canvas.draw()
canvas.get_tk_widget().pack()
window.mainloop()