import numpy as np
import cv2 as cv
#import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import random as rng
import operator
import math
def docdactrung1(url):
    img = cv.imread(url)
    img = cv.resize(img,(800,400))
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    dst = cv.cornerHarris(gray,2,5,0.04)
    img[dst>0.04*dst.max()] = [0,255,0]
    count = 0
    datagoc = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]): 
            if img[i,j,0]==0 and img[i,j,1]==255 and img[i,j,2]==0:
                tiledai = i*100/img.shape[0]
                tilerong = j*100/img.shape[1]
                datagoc.append([tiledai,tilerong])
    # tinh dac trung sift
    #listkp = np.array(listkp)
    return datagoc
def docdactrungnhieu():
    listnew = []
    for i in range(10):
        url = "H:/Github/AINhandien/ex/hoa/hoa"+str(i+11)+".jpg"
        listkp = docdactrung1(url)
        listnew += listkp
    labellist=[]
    a = len(listnew)
    for i in range(a):
        labellist.append(1)
    for i in range(10):
        url = "H:/Github/AINhandien/ex/phat/phat"+str(i+1)+".jpg"
        listkp = docdactrung1(url)
        listnew += listkp
    b = len(listnew)
    for i in range(b-a):
        labellist.append(2)
    # for i in range(10):
    #     url = "H:/Github/AINhandien/ex/dung/dung"+str(i+11)+".jpg"
    #     listkp = docdactrung1(url)
    #     listnew += listkp
    # c = len(listnew)
    # for j in range(c-b):
    #     labellist.append(3)
    return listnew,labellist
print("Dang load data")
listnew,labellist = docdactrungnhieu()
print("Dang train")
listnew = np.array(listnew, dtype = np.float32)
labellist = np.array(labellist, dtype = np.int32)
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(0.1)
svm.setGamma(1)
svm.setCoef0(0)
svm.setDegree(2)
svm.setKernel(cv.ml.SVM_POLY)
svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, int(1e7), 1e-6))
svm.train(listnew, cv.ml.ROW_SAMPLE, labellist)
print("Da train xong")
#trainSVM()
#print("xong")
window = tk.Tk()
window.title("AI Nhan dien")

myLabel = tk.Label(window,text="Duong dan:   " )
myLabel2 = tk.Label(window,text="" )
window.geometry('350x200')
myLabel.grid(column = 0,row=1)
myLabel2.grid(column = 1,row=1)
myLabel3 = tk.Label(window,text="" )
myLabel3.grid(column = 1,row=5)
def clicked_open():
    window.filename = tk.filedialog.askopenfilename(title="Select A File", filetype=(("all files", "*"),("jpg files","*.jpg"),("png files","*.png"),("jpeg files","*.jpeg")))
    myLabel2.config(text=window.filename)
bt = tk.Button(window,text="Open File", command=clicked_open)
def clicked_sift():
    dactrung = docdactrung1(window.filename)
    hoa  =0
    phat =0
    dung =0
    respond = np.empty((len(dactrung), 1), dtype=np.int32)
    for i in range(dactrung.__len__()):
        sampleMat = np.matrix([dactrung[i]], dtype=np.float32)
        respond[i]=svm.predict(sampleMat)[1]
        if respond[i]==1 :
            hoa+=1
        elif respond[i]==2:
            phat+=1
        else:
            dung+=1
    if max(hoa,phat,dung) == hoa:
        myLabel3.config(text="Chữ ký của Hòa "+str(hoa*(100/len(dactrung)))+"%")
    elif max(hoa,phat,dung) == phat:
        myLabel3.config(text="Chữ ký của Phát "+str(phat*(100/len(dactrung)))+"%")
    else:
        myLabel3.config(text="Chữ ký của Dũng "+str(dung*(100/len(dactrung)))+"%")
    print(respond[0:len(dactrung)])
    

bt2 = tk.Button(window,text="Nhan dien", command=clicked_sift)
bt2.grid(column = 0,row = 3)
bt.grid(column =0,row = 0)
window.mainloop()    
