import numpy as np
import cv2 as cv
#import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import random as rng
import operator
import math
from tienxuly import xulyanh
from demo.thuatToan import detectOj
def khoangcachchenhlech(x1,y1,x2,y2):
    kc = 0
    kc = abs(x2-x1)+abs(y2-y1)
    #kc = math.sqrt(math.pow(x2-x1,2)+math.pow(y2-y1,2))
    return kc
def docdactrung1(img):
    #img = cv.resize(img,(800,400))
    #gray = img
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    dst = cv.cornerHarris(gray,2,5,0.04)
    sift = cv.xfeatures2d.SIFT_create()
    #sift = cv.xfeatures2d.SURF_create()
    kp = sift.detect(gray,None)
    kp,des = sift.compute(gray,kp)
    img[dst>0.04*dst.max()] = [0,255,0]
    count = 0
    datagoc = []
    #Loai bo cac diem goc gan nhau
    for i in range(img.shape[0]):
        for j in range(img.shape[1]): 
            if img[i,j,0]==0 and img[i,j,1]==255 and img[i,j,2]==0:
                #tiledai = i*100/img.shape[0]
                #tilerong = j*100/img.shape[1]
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
    
    # for i in range(count):
    #     cv.circle(img,(int(datagoc[i][1]),int(datagoc[i][0])),3,(0,0,255),2)
    # cv.imshow('ex2',img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
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
    # for i in range(count-1):
    #     for j in range(len(kp)):
    #         if(abs(datagoc[i,0]-datakp[j,1])<5 and abs(datagoc[i,1]-datakp[j,2])<5 and listed[j]!=1):
    #             listkp.append(datakp[j])
    #             kp2.append(kp[j])
    #             listed[j] = 1
    #             #print(datakp[j,1],datakp[j,2])
    #             count2+=1
    #             num += 1
    #             if(num>2): break
    #     num = 0
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
                kp2.append(kp[kc[num][2]])
                listed[kc[num][2]]=1
                can = False
            num += 1

    for i in range(count):
        # listkp[i][1] = listkp[i][1]*200/img.shape[0]
        # listkp[i][2]= listkp[i][2]*200/img.shape[1]
        #cv.circle(img,(int(listkp[i][1]),int(listkp[i][2])),3,(255,0,0),2)
        listkp[i][1] = listkp[i][1]*200/img.shape[1]
        listkp[i][2]= listkp[i][2]*200/img.shape[0]
    
    #listkp = np.array(listkp)
    #img=cv.drawKeypoints(img,kp2,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv.imshow('ex2',img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    listkp=sorted(listkp,key=operator.itemgetter(1))
    return listkp
def docdactrungnhieu():
    listnew = []
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
    
    #trainDataSign = np.zeros((20,2*10, 4), dtype=np.int32)
    #listnew = trainDataSign
    for i in range(20):
        url = "C:/Github/AINhandien/ex/hoa/"+str(i)+".jpg"
        img = cv.imread(url)
        img = xulyanh(img)
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        img_list = detectOj(img)
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
        #listnew[j].append(listkp[j])
        #listnew += listkp

    labellist=[]
    # a = len(listnew)
    # for i in range(a):
    #     labellist.append(1)
    for i in range(20):
        url = "C:/Github/AINhandien/ex/phat/"+str(i)+".jpg"
        img = cv.imread(url)
        img = xulyanh(img)
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        img_list = detectOj(img)
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
    # for i in range(10):
    #     url = "H:/Github/AINhandien/ex/dung/"+str(i)+".jpg"
    #     img = cv.imread(url)
    #     listkp = docdactrung1(img)
    #     for j in range(len(listkp)):
    #         if listkp[j][1] <= 20:
    #             listtrain0.append(listkp[j])
    #             listlb0.append(3)
    #         elif listkp[j][1] > 20 and listkp[j][1]<=40:
    #             listtrain1.append(listkp[j])
    #             listlb1.append(3)
    #         elif listkp[j][1] > 40 and listkp[j][1]<=60:
    #             listtrain2.append(listkp[j])
    #             listlb2.append(3)
    #         elif listkp[j][1] > 60 and listkp[j][1]<=80:
    #             listtrain3.append(listkp[j])
    #             listlb3.append(3)
    #         elif listkp[j][1] > 80 and listkp[j][1]<=100:
    #             listtrain4.append(listkp[j])
    #             listlb4.append(3)
    #         elif listkp[j][1] > 100 and listkp[j][1]<=120:
    #             listtrain5.append(listkp[j])
    #             listlb5.append(3)
    #         elif listkp[j][1] > 120 and listkp[j][1]<=140:
    #             listtrain6.append(listkp[j])
    #             listlb6.append(3)
    #         elif listkp[j][1] > 140 and listkp[j][1]<=160:
    #             listtrain7.append(listkp[j])
    #             listlb7.append(3)
    #         elif listkp[j][1] > 160 and listkp[j][1]<=180:
    #             listtrain8.append(listkp[j])
    #             listlb8.append(3)
    #         elif listkp[j][1] > 180 and listkp[j][1]<=200:
    #             listtrain9.append(listkp[j])
    #             listlb9.append(3)
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
    # b = len(listnew)
    # for i in range(b-a):
    #     labellist.append(2)
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
svm = np.empty((10),dtype = cv.ml_SVM)
for i in range(10):

    listnew[i] = np.array(listnew[i], dtype = np.float32)
    labellist[i] = np.array(labellist[i], dtype = np.int32)
    svm[i] = cv.ml.SVM_create()
    svm[i].setType(cv.ml.SVM_C_SVC)
    svm[i].setC(100)
    svm[i].setGamma(1)
    svm[i].setCoef0(0)
    svm[i].setDegree(3)
    svm[i].setKernel(cv.ml.SVM_INTER)
    svm[i].setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, int(1e7), 1e-6))
## [init]

#------------------------ 3. Train the svm ----------------------------------------------------
## [train]
    svm[i].train(listnew[i], cv.ml.ROW_SAMPLE, labellist[i])
    svm[i].save(str(i)+".dat")
## [train]
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
    img = cv.imread(window.filename)
    img = xulyanh(img)
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    listkp = docdactrung1(img)
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
    hoa  =0
    phat =0
    dung =0
    respond = 0
    res = []
    #respond = np.empty((len(listkp),1),dtype = np.int32)
    for i in range(10):
        for j in range(len(datalist[i])):
            datalist[i][j] = np.array(datalist[i][j],dtype = np.float32)
            sampleMat = np.matrix([datalist[i][j]], dtype=np.float32)
            respond=svm[i].predict(sampleMat)[1]
            res.append(respond)
            if respond==1 :
                hoa+=1
            elif respond==2:
                phat+=1
            else:
                dung+=1
    # respond = np.empty((len(dactrung), 1), dtype=np.int32)
    # for i in range(dactrung.__len__()):
    #     sampleMat = np.matrix([dactrung[i]], dtype=np.float32)
    #     respond[i]=svm.predict(sampleMat)[1]
    #     if respond[i]==1 :
    #         hoa+=1
    #     elif respond[i]==2:
    #         phat+=1
    #     else:
    #         dung+=1
    if max(hoa,phat,dung) == hoa:
        myLabel3.config(text="Chữ ký của Hòa "+str(hoa*(100/len(listkp)))+"%")
        #myLabel3.config(text="Chữ ký của Hòa ")
    elif max(hoa,phat,dung) == phat:
        myLabel3.config(text="Chữ ký của Phát "+str(phat*(100/len(listkp)))+"%")
        #myLabel3.config(text="Chữ ký của Phát ")
    else:
        myLabel3.config(text="Chữ ký của Dũng "+str(dung*(100/len(listkp)))+"%")
    #print(res[0:len(listkp)])
    

bt2 = tk.Button(window,text="Nhan dien", command=clicked_sift)
bt2.grid(column = 0,row = 3)
bt.grid(column =0,row = 0)
window.mainloop()    
