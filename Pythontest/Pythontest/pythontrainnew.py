import numpy as np
import cv2 as cv
from tkinter.ttk import Frame, Button, Style, Entry
import tkinter as tk
from tkinter import filedialog
import random as rng
import operator
import math
from tienxuly import xulyanh
from demo.thuatToan import detectOj
from tkinter import messagebox
def khoangcachchenhlech(x1,y1,x2,y2):
    kc = 0
    kc = abs(x2-x1)+abs(y2-y1)
    return kc
def docdactrung1(img):
    #img = cv.resize(img,(800,400))
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    dsg = cv.cornerHarris(gray,2,5,0.04)
    sift = cv.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    kp,des = sift.compute(gray,kp)
    img[dsg>0.04*dsg.max()] = [0,255,0]
    count = 0
    datagoc = []
    #Loai bo cac diem goc gan nhau
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
    count2 = 0
    num = 0
    listed = np.zeros((len(kp),1),dtype=np.float32)
    # moi diem goc lay 1 keypoint gan nhat
    for i in range(count):
        kc = []
        for j in range(len(kp)):
            kc.append((khoangcachchenhlech(datagoc[i,1],datagoc[i,0],datakp[j,1],datakp[j,2]),i,j))
        kc=sorted(kc,key=operator.itemgetter(0))
        can = True
        num = 0
        while can:
            if listed[kc[num][2]]!=1:
                listkp.append(datakp[kc[num][2]])
                listed[kc[num][2]]=1
                can = False
            num += 1

    for i in range(count):
        #cv.circle(img,(int(listkp[i][1]),int(listkp[i][2])),3,(255,0,0),2)
        listkp[i][1] = listkp[i][1]*200/img.shape[1]
        listkp[i][2]= listkp[i][2]*200/img.shape[0]
    # cv.imshow('ex2',img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    listkp=sorted(listkp,key=operator.itemgetter(1))
    return listkp
def docdactrungnhieu(listurl1,listurl2):
    listnew = []
    labellist = []
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
    listlb0 = []
    listlb1 = []
    listlb2 = []
    listlb3 = []
    listlb4 = []
    listlb5 = []
    listlb6 = []
    listlb7 = []
    listlb8 = []
    listlb9 = []
    for url in listurl1:
        #url = "C:/Github/AINhandien/ex/hoa/"+str(i)+".jpg"
        img = cv.imread(url)
        img = xulyanh(img)
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        img_list,img_detect = detectOj(img)
        for k in range(len(img_list)):
            listkp = docdactrung1(img_list[k])
            for j in range(len(listkp)):
                if listkp[j][1] <= 20:
                    listtrain0.append(listkp[j])
                    listlb0.append(1)
                elif listkp[j][1] > 20 and listkp[j][1]<=40:
                    listtrain1.append(listkp[j])
                    listlb1.append(1)
                elif listkp[j][1] > 40 and listkp[j][1]<=60:
                    listtrain2.append(listkp[j])
                    listlb2.append(1)
                elif listkp[j][1] > 60 and listkp[j][1]<=80:
                    listtrain3.append(listkp[j])
                    listlb3.append(1)
                elif listkp[j][1] > 80 and listkp[j][1]<=100:
                    listtrain4.append(listkp[j])
                    listlb4.append(1)
                elif listkp[j][1] > 100 and listkp[j][1]<=120:
                    listtrain5.append(listkp[j])
                    listlb5.append(1)
                elif listkp[j][1] > 120 and listkp[j][1]<=140:
                    listtrain6.append(listkp[j])
                    listlb6.append(1)
                elif listkp[j][1] > 140 and listkp[j][1]<=160:
                    listtrain7.append(listkp[j])
                    listlb7.append(1)
                elif listkp[j][1] > 160 and listkp[j][1]<=180:
                    listtrain8.append(listkp[j])
                    listlb8.append(1)
                elif listkp[j][1] > 180 and listkp[j][1]<=200:
                    listtrain9.append(listkp[j])
                    listlb9.append(1)
    for url in listurl2:
        #url = "C:/Github/AINhandien/ex/phat/"+str(i)+".jpg"
        img = cv.imread(url)
        img = xulyanh(img)
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        img_list,img_detect = detectOj(img)
        for k in range(len(img_list)):
            listkp = docdactrung1(img_list[k])
            for j in range(len(listkp)):
                if listkp[j][1] <= 20:
                    listtrain0.append(listkp[j])
                    listlb0.append(2)
                elif listkp[j][1] > 20 and listkp[j][1]<=40:
                    listtrain1.append(listkp[j])
                    listlb1.append(2)
                elif listkp[j][1] > 40 and listkp[j][1]<=60:
                    listtrain2.append(listkp[j])
                    listlb2.append(2)
                elif listkp[j][1] > 60 and listkp[j][1]<=80:
                    listtrain3.append(listkp[j])
                    listlb3.append(2)
                elif listkp[j][1] > 80 and listkp[j][1]<=100:
                    listtrain4.append(listkp[j])
                    listlb4.append(2)
                elif listkp[j][1] > 100 and listkp[j][1]<=120:
                    listtrain5.append(listkp[j])
                    listlb5.append(2)
                elif listkp[j][1] > 120 and listkp[j][1]<=140:
                    listtrain6.append(listkp[j])
                    listlb6.append(2)
                elif listkp[j][1] > 140 and listkp[j][1]<=160:
                    listtrain7.append(listkp[j])
                    listlb7.append(2)
                elif listkp[j][1] > 160 and listkp[j][1]<=180:
                    listtrain8.append(listkp[j])
                    listlb8.append(2)
                elif listkp[j][1] > 180 and listkp[j][1]<=200:
                    listtrain9.append(listkp[j])
                    listlb9.append(2)
    listnew.append(listtrain0)
    listnew.append(listtrain1)
    listnew.append(listtrain2)
    listnew.append(listtrain3)
    listnew.append(listtrain4)
    listnew.append(listtrain5)
    listnew.append(listtrain6)
    listnew.append(listtrain7)
    listnew.append(listtrain8)
    listnew.append(listtrain9)
    labellist.append(listlb0)
    labellist.append(listlb1)
    labellist.append(listlb2)
    labellist.append(listlb3)
    labellist.append(listlb4)
    labellist.append(listlb5)
    labellist.append(listlb6)
    labellist.append(listlb7)
    labellist.append(listlb8)
    labellist.append(listlb9)
    return listnew,labellist
def train_svm(listurl1,listurl2):
    print("Dang load data")
    listnew,labellist = docdactrungnhieu(listurl1,listurl2)
    print("Dang train")
    svm = np.empty((10),dtype = cv.ml_SVM)
    for i in range(10):
        listnew[i] = np.array(listnew[i], dtype = np.float32)
        labellist[i] = np.array(labellist[i], dtype = np.int32)
        if len(labellist[i]!=0):
            if (max(labellist[i])!=min(labellist[i])):
                svm[i] = cv.ml.SVM_create()
                svm[i].setType(cv.ml.SVM_C_SVC)
                svm[i].setC(100)
                svm[i].setGamma(1)
                svm[i].setCoef0(0)
                svm[i].setDegree(3)
                svm[i].setKernel(cv.ml.SVM_INTER)
                svm[i].setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, int(1e7), 1e-6))
                svm[i].train(listnew[i], cv.ml.ROW_SAMPLE, labellist[i])
                svm[i].save(str(i)+".dat")
            else:
                with open(str(i)+".txt",'w',encoding='utf-8') as f:
                    f.write(str(max(labellist[i])))
        else:
            with open(str(i)+".txt",'w',encoding='utf-8') as f:
                f.write(0)
    print("Da train xong")
listurl1 = []
listurl2 = []
window = tk.Tk()
window.title("AI Train Nhận diện chữ ký")
myLabel = tk.Label(window,text="")
window.geometry('280x170')
myLabel.place(x=0,y=25)
myLabel3 = tk.Label(window,text="" )
myLabel3.place(x=0,y=125)
myLabel4 = tk.Label(window,text="" )
myLabel4.place(x=0,y=75)
myLabel1 = tk.Label(window,text="Tên 1:")
myLabel1.place(x=110,y=3)
myLabel2 = tk.Label(window,text="Tên 2:")
myLabel2.place(x=110,y=50)
def clicked_open_1():
    myLabel.config(text="Đang chọn ảnh ... ")
    window.update()
    global listurl1
    listurl1 = tk.filedialog.askopenfilenames(title="Select List File", filetype=(("all files", "*"),("jpg files","*.jpg"),("png files","*.png"),("jpeg files","*.jpeg")))
    if(len(listurl1)!=0):
        myLabel.config(text="Đã chọn ảnh đối tượng 1")
        window.update()
    else:
        myLabel.config(text="Chưa chọn ảnh đối tượng 1")
        window.update()
bt = Button(window,text="Open Files 1", command=clicked_open_1)
def clicked_train():
    if(len(listurl1) == 0 or len(listurl2)==0):
        messagebox.showerror("Error","Bạn chưa chọn xong ảnh")
        #window.update()
        return
    if(len(name1.get())==0 or len(name2.get())==0):
        messagebox.showerror("Error","Bạn chưa nhập tên đối tượng")
        #window.update()
        return
    if(name1.get()==name2.get()):
        messagebox.showerror("Error","Tên bị trùng")
        #window.update()
        return
    with open("name.txt",'w',encoding='utf-8') as f:
        f.write(name1.get()+"\n")
        f.write(name2.get()+"\n")
    myLabel3.config(text="Đang train...")
    window.update()
    train_svm(listurl1,listurl2)
    myLabel3.config(text="Đã train xong!")
    window.update()
bt2 = Button(window,text="Train", command=clicked_train)
def clicked_open_2():
    myLabel4.config(text="Đang chọn ảnh ... ")
    window.update()
    global listurl2
    listurl2 = tk.filedialog.askopenfilenames(title="Select List File", filetype=(("all files", "*"),("jpg files","*.jpg"),("png files","*.png"),("jpeg files","*.jpeg")))
    if(len(listurl2)!=0):
        myLabel4.config(text="Đã chọn ảnh đối tượng 2")
        window.update()
    else:
        myLabel4.config(text="Chưa chọn ảnh đối tượng 2")
        window.update()
bt3 = Button(window,text="Open Files 2", command=clicked_open_2)
bt3.place(x=0,y=50)
bt2.place(x=0,y=100)
bt.grid(column =0,row = 0)
name1 = tk.StringVar()
name1_textbox = Entry(window,width = 15,textvariable = name1)
name1_textbox.place(x=150,y=3)
name2 = tk.StringVar()
name2_textbox = Entry(window,width = 15,textvariable = name2)
name2_textbox.place(x=150,y=50)
window.mainloop()    
