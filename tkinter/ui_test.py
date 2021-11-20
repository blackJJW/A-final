from tkinter import *
import tkinter.font as font

root = Tk()
root.geometry("1920x1080")

bg = PhotoImage(file = "ai_back.png")
canvas1 = Canvas( root, width = 1920, height = 1080)
canvas1.pack(fill = "both", expand = True)
canvas1.create_image( 0, 0, image = bg, anchor = "nw")

fontLabel1 = font.Font(family="바탕", size=50)
canvas1.create_text( 400, 50, fill="#FFFFFF", text = "Welcome 안녕하세요", font = fontLabel1)

Textbox1Font = font.Font(size=30)
strTextbox1 = StringVar()
textbox1 = Entry(root, textvariable=strTextbox1)
textbox1.place(x=100,y=100, width=800,height=50)
textbox1['font'] = Textbox1Font 

text1 = Text(root)
text1.place(x=100,y=200, width=800,height=500)

strBtn1 = StringVar()
def funcBtn1():
    tmp = strTextbox1.get()
    text1.delete(1.0,"end")
    text1.insert(1.0, tmp)
    print(tmp)

strBtn1 = "버튼1"
myButton1Font = font.Font(size=50)
button1 = Button( root, text = strBtn1, command = funcBtn1, font ='바탕', bd = 20, fg = '#FFFFFF', bg ="#6799FF", relief = "groove")
button1['font'] = myButton1Font
button1.place(x=1600,y=100, width=300,height=150)

def funcBtn2():
    print("funcBtn2")

strBtn2 = "버튼2"
myButton2Font = font.Font(size=50)
button2 = Button( root, text = strBtn2, command = funcBtn2, font ='바탕', bd = 20, fg = '#FFFFFF', bg ="#6799FF", relief = "groove")
button2['font'] = myButton2Font 
button2.place(x=1600,y=300, width=300,height=150)

def funcBtn3():
    print("funcBtn3")

strBtn3 = "버튼3"
myButton3Font = font.Font(size=50)
button3 = Button( root, text = strBtn3, command = funcBtn3, font ='바탕', bd = 20, fg = '#FFFFFF', bg ="#6799FF", relief = "groove")
button3['font'] = myButton3Font 
button3.place(x=1600,y=500, width=300,height=150)


root.mainloop()
