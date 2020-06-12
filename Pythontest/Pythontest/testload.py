import numpy as np
import cv2 as cv
#import matplotlib.pyplot as plt
from tkinter.ttk import Frame, Button, Style
import tkinter as tk
from tkinter import filedialog, BOTH,W
import os
import random as rng
import operator
from PIL import Image, ImageTk
from tienxuly import xulyanh

from demo.thuatToan import *

def khoangcachchenhlech(x1,y1,x2,y2):
    kc = 0
    kc = abs(x2-x1)+abs(y2-y1)
    return kc
def docdactrung1(img):
    #img = cv.resize(img,(800,400))
    gray = img
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    dst = cv.cornerHarris(gray,2,5,0.04)
    sift = cv.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    kp,des = sift.compute(gray,kp)
    img[dst>0.04*dst.max()] = [0,255,0]
    count = 0
    datagoc = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]): 
            if img[i,j,0]==0 and img[i,j,1]==255 and img[i,j,2]==0:
                if(count==0):
                   datagoc.append([i,j])
                   count += 1
                if(abs(i-datagoc[count-1][0])>20 or abs(j-datagoc[count-1][1])>20): #bo qua cac diem gan
                    can = True
                    for k in range(count):
                        if(abs(i-datagoc[k][0])<10 and abs(j-datagoc[k][1])<10):
                            can=False
                            break
                    if can:
                        datagoc.append([i,j])
                        count+=1 
    datagoc = np.array(datagoc)
    # tinh dac trung sift
    datakp = np.empty((len(kp),4),dtype = np.float32)
    for j in range(len(kp)):
         datakp[j] = [kp[j].size,kp[j].pt[0],kp[j].pt[1],kp[j].angle]
    listkp = []
    kp2=[]
    count2 = 0
    num = 0
    listed = np.zeros((len(kp),1),dtype=np.float32)

    for i in range(count):
        kc = []
        for j in range(len(kp)):
            kc.append((khoangcachchenhlech(datagoc[i,1],datagoc[i,0],datakp[j,1],datakp[j,2]),i,j))
        kc=sorted(kc,key=operator.itemgetter(0))
        can = True
        num = 0
        if i >= len(kp) : break
        while can:
            if listed[kc[num][2]]!=1:
                listkp.append(datakp[kc[num][2]])
                kp2.append(kp[kc[num][2]])
                listed[kc[num][2]]=1
                can = False
            num += 1

    for i in range(len(listkp)):
        #cv.circle(img,(int(listkp[i][1]),int(listkp[i][2])),3,(255,0,0),2)
        listkp[i][1] = listkp[i][1]*200/img.shape[1]
        listkp[i][2]= listkp[i][2]*200/img.shape[0]

    listkp=sorted(listkp,key=operator.itemgetter(1))
    return listkp
print("Dang train")
svm = np.empty((10),dtype = cv.ml_SVM)
one_res = np.empty((10),dtype = np.int32)
for i in range(10):
    if os.path.isfile(str(i)+".dat") == True:
        svm[i] = cv.ml.SVM_load(str(i)+".dat")
    elif os.path.isfile(str(i)+".txt")==True:
        with open(str(i)+".txt",'r',encoding='utf-8') as f:
            one_res[i] = int(f.read(1))
name1 = ""
name2 = ""
with open("name.txt",'r',encoding='utf-8') as f:
    name1 = f.readline()
    name2 = f.readline()
print("Da train xong")
window = tk.Tk()
window.title("AI Nhận diện chữ ký")
myLabel = tk.Label(window,text="Đường dẫn:   ")
myLabel2 = tk.Label(window,text="")
window.geometry('450x350')
myLabel.grid(column = 0,row=1)
myLabel2.place(x=90,y=25)
myLabel3 = tk.Label(window,text="" )
myLabel3.place(x=90,y=50)
myLabel4 = tk.Label(window,text="" )
myLabel4.place(x=0,y=75)
imagelabel = tk.Label(window,image="")
imagelabel.place(x=20,y=110)
def clicked_open():
    myLabel3.config(text="")
    myLabel4.config(text="Đang chọn ảnh ... ")
    window.update()
    window.filename = tk.filedialog.askopenfilename(title="Select A File", filetype=(("all files", "*"),("jpg files","*.jpg"),("png files","*.png"),("jpeg files","*.jpeg")))
    myLabel2.config(text=window.filename)
    if(window.filename != ''):
        myLabel4.config(text="Đã chọn ảnh")
        img = Image.open(window.filename)
        img = img.resize((400,225))
        image_show = ImageTk.PhotoImage(img)
        imagelabel.config(image = image_show)
        imagelabel.image = image_show
        window.update()
    else:
        myLabel4.config(text="Chưa chọn ảnh")
        imagelabel.image = ""
        window.update()
bt = Button(window,text="Open File", command=clicked_open)
def clicked_nhandang():
    if(window.filename == ''):
        myLabel4.config(text="Bạn chưa chọn ảnh !!!")
        window.update()
        return
    myLabel4.config(text="Đang kiểm tra ... ")
    window.update()
    img2 = cv.imread(window.filename)
    img = xulyanh(img2)
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    img_list,img_detect = detectOj(img)
    img3 = Image.fromarray(img_detect)
    img3 = img3.resize((400,225))
    image_show = ImageTk.PhotoImage(img3)
    imagelabel.config(image = image_show)
    imagelabel.image = image_show
    window.update()
    for k in range(len(img_list)):
        cv.imshow('Image After Detected',img_list[k])
        while(cv.getWindowImageRect('Image After Detected')!=(-1,-1,-1,-1)):
            a=cv.waitKey(1)
            if a != -1: break
        cv.destroyAllWindows()
        myLabel4.config(text="Đang kiểm tra ... ")
        window.update()
        listkp = docdactrung1(img_list[k])
        listtrain0 = []
        listtrain1 = []
        listtrain2 = []
        listtrain3 = []
        listtrain4 = []
        listtrain5 = []
        listtrain6 = []
        listtrain7 = []
        listtrain8 = []
        listtrain9 = []
        for j in range(len(listkp)):
            if listkp[j][1] <= 20:
                listtrain0.append(listkp[j])
            elif listkp[j][1] > 20 and listkp[j][1]<=40:
                listtrain1.append(listkp[j])
            elif listkp[j][1] > 40 and listkp[j][1]<=60:
                listtrain2.append(listkp[j])
            elif listkp[j][1] > 60 and listkp[j][1]<=80:
                listtrain3.append(listkp[j])
            elif listkp[j][1] > 80 and listkp[j][1]<=100:
                listtrain4.append(listkp[j])
            elif listkp[j][1] > 100 and listkp[j][1]<=120:
                listtrain5.append(listkp[j])
            elif listkp[j][1] > 120 and listkp[j][1]<=140:
                listtrain6.append(listkp[j])
            elif listkp[j][1] > 140 and listkp[j][1]<=160:
                listtrain7.append(listkp[j])
            elif listkp[j][1] > 160 and listkp[j][1]<=180:
                listtrain8.append(listkp[j])
            elif listkp[j][1] > 180 and listkp[j][1]<=200:
                listtrain9.append(listkp[j])
        datalist = []
        datalist.append(listtrain0)
        datalist.append(listtrain1)
        datalist.append(listtrain2)
        datalist.append(listtrain3)
        datalist.append(listtrain4)
        datalist.append(listtrain5)
        datalist.append(listtrain6)
        datalist.append(listtrain7)
        datalist.append(listtrain8)
        datalist.append(listtrain9)
        mot  =0
        hai =0
        respond = 0
        res = []
        for i in range(10):
            for j in range(len(datalist[i])):
                datalist[i][j] = np.array(datalist[i][j],dtype = np.float32)
                sampleMat = np.matrix([datalist[i][j]], dtype=np.float32)
                if svm[i] != None:
                    respond=svm[i].predict(sampleMat)[1]
                else:
                    respond=one_res[i]
                res.append(respond)
                if respond==1 :
                    mot+=1
                elif respond==2:
                    hai+=1
        if max(mot,hai) == mot:
            myLabel3.config(text="Chữ ký của "+name1[0:len(name1)-1]+" "+str(mot*(100/len(listkp)))+"%")
            window.update()
        elif max(mot,hai) == hai:   
            myLabel3.config(text="Chữ ký của " +name2[0:len(name2)-1]+" "+str(hai*(100/len(listkp)))+"%")
            window.update()
        myLabel4.config(text="Đã có kết quả !!!")
        window.update()
        #print(res[0:len(listkp)])
bt2 = Button(window,text="Nhận diện", command=clicked_nhandang)
bt2.grid(column = 0,row = 3)
bt.grid(column =0,row = 0)
window.mainloop()    
