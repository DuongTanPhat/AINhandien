import numpy as np
import cv2 as cv
#import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import random as rng
import operator


NTRAINING_SAMPLES = 10 # Number of training samples per class
NUMBER_KEYPOINT = 10
NUMBER_OBJECT = 2
# Data for visual representation

trainDataSign = np.empty((NUMBER_KEYPOINT,NUMBER_OBJECT*NTRAINING_SAMPLES, 4), dtype=np.float32)
labels = np.empty((NUMBER_OBJECT*NTRAINING_SAMPLES, 1), dtype=np.int32)
labels[0:NTRAINING_SAMPLES,:] = 1                       # Class 1
labels[NTRAINING_SAMPLES:2*NTRAINING_SAMPLES,:] = 2     # Class 2
#labels[2*NTRAINING_SAMPLES:3*NTRAINING_SAMPLES,:] = 3    # Class 3

# --------------------- 1. Set up training data randomly ---------------------------------------
print('Starting loading data process')
for i in range(10):
    url = "H:/Github/AINhandien/ex/hoa/hoa"+str(i+1)+".jpg"
    img = cv.imread(url)
    img2 = cv.imread(url)
    img2 = cv.resize(img2,(800,400))
    gray= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    dst = cv.cornerHarris(gray,2,5,0.04)
    sift = cv.xfeatures2d.SIFT_create()
    kp= sift.detect(gray,None)
    kp,des = sift.compute(gray,kp)
    datalist = np.empty((len(kp),4),dtype = np.float32)
    for j in range(len(kp)):
        datalist[j] = [kp[j].size,kp[j].pt[0],kp[j].pt[1],kp[j].angle]
    datalist=sorted(datalist,key=operator.itemgetter(0))
    datalist.reverse();
    for j in range(NUMBER_KEYPOINT):
        trainDataSign[j,i,0:4]=datalist[j]
for i in range(10):
    url = "H:/Github/AINhandien/ex/phat/phat"+str(i+1)+".jpg"
    img = cv.imread(url)
    img2 = cv.imread(url)

    img2 = cv.resize(img,(800,400))
    gray= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp= sift.detect(gray,None)
    kp,des = sift.compute(gray,kp)
    datalist = np.empty((len(kp),4),dtype = np.float32)
    for j in range(len(kp)):
        datalist[j] = [kp[j].size,kp[j].pt[0],kp[j].pt[1],kp[j].angle]
    datalist=sorted(datalist,key=operator.itemgetter(0))
    datalist.reverse();
    for j in range(NUMBER_KEYPOINT):
        trainDataSign[j,i+10,0:4]=datalist[j]
# for i in range(10):
#     url = "H:/Github/AINhandien/ex/dung/dung"+str(i+1+10)+".jpg"
#     img = cv.imread(url)
#     img2 = cv.imread(url)

#     img2 = cv.resize(img,(800,400))
#     gray= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
#     sift = cv.xfeatures2d.SIFT_create()
#     kp= sift.detect(gray,None)
#     kp,des = sift.compute(gray,kp)
#     datalist = np.empty((len(kp),4),dtype = np.float32)
#     for j in range(len(kp)):
#         datalist[j] = [kp[j].size,kp[j].pt[0],kp[j].pt[1],kp[j].angle]
#     datalist=sorted(datalist,key=operator.itemgetter(0))
#     datalist.reverse();
#     for j in range(NUMBER_KEYPOINT):
#         trainDataSign[j,i+20,0:4]=datalist[j]
print('Finished loading data process')
## [setup1]
# Generate random points for the class 1
#------------------------ 2. Set up the support vector machines parameters --------------------
svm = np.empty((NUMBER_KEYPOINT),dtype = cv.ml_SVM)
print('Starting training process')
for i in range(NUMBER_KEYPOINT):

    
## [init]
    
    svm[i] = cv.ml.SVM_create()
    svm[i].setType(cv.ml.SVM_C_SVC)
    svm[i].setC(0.1)
    svm[i].setGamma(1)
    svm[i].setCoef0(0)
    svm[i].setDegree(2)
    svm[i].setKernel(cv.ml.SVM_POLY)
    svm[i].setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, int(1e7), 1e-6))
## [init]

#------------------------ 3. Train the svm ----------------------------------------------------
## [train]
    svm[i].train(trainDataSign[i], cv.ml.ROW_SAMPLE, labels)
## [train]
print('Finished training process')

window = tk.Tk()
window.title("AI Nhan dien")

myLabel = tk.Label(window,text="Duong dan:   " )
myLabel2 = tk.Label(window,text="" )
window.geometry('350x200')
myLabel.grid(column = 0,row=1)
myLabel2.grid(column = 1,row=1)
myLabel3 = tk.Label(window,text="" )
myLabel3.grid(column = 1,row=5)
respond = np.empty((NUMBER_KEYPOINT, 1), dtype=np.int32)
def clicked_open():
    window.filename = tk.filedialog.askopenfilename(title="Select A File", filetype=(("all files", "*"),("jpg files","*.jpg"),("png files","*.png"),("jpeg files","*.jpeg")))
    myLabel2.config(text=window.filename)
bt = tk.Button(window,text="Open File", command=clicked_open)
def clicked_sift():
    imgnew = cv.imread(window.filename)
    imgnew2 = cv.resize(imgnew,(800,400))
    gray= cv.cvtColor(imgnew2,cv.COLOR_BGR2GRAY)
    
    sift = cv.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    #cv.Laplacian()
#img=cv2.drawKeypoints(gray,kp)

#cv2.imwrite('sift_keypoints.jpg',img)
    kp,des = sift.compute(gray,kp)

    #img=cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    datalist = np.empty((len(kp),4),dtype = np.float32)
    for j in range(len(kp)):
        datalist[j] = [kp[j].size,kp[j].pt[0],kp[j].pt[1],kp[j].angle]
    datalist=sorted(datalist,key=operator.itemgetter(0))
    datalist.reverse();
    hoa  =0;
    phat =0;
    dung =0;
    for i in range(NUMBER_KEYPOINT):
        sampleMat = np.matrix([datalist[i]], dtype=np.float32)
        respond[i]=svm[i].predict(sampleMat)[1]
        if respond[i]==1 :
            hoa+=1;
        elif respond[i]==2:
            phat+=1;
        else:
            dung+=1;
    if max(hoa,phat,dung) == hoa:
        myLabel3.config(text="Chữ ký của Hòa "+str(hoa*(100/NUMBER_KEYPOINT))+"%")
    elif max(hoa,phat,dung) == phat:
        myLabel3.config(text="Chữ ký của Phát "+str(phat*(100/NUMBER_KEYPOINT))+"%")
    else:
        myLabel3.config(text="Chữ ký của Dũng "+str(dung*(100/NUMBER_KEYPOINT))+"%")
    print(respond[0:NUMBER_KEYPOINT])
    

bt2 = tk.Button(window,text="Nhan dien", command=clicked_sift)
bt2.grid(column = 0,row = 3)
bt.grid(column =0,row = 0)
window.mainloop()    
















##plt.scatter(4,5,50,'b','<')
##plt.scatter(50,5,50,'c','h')
#a = (np.random.randint(0,100,(100,2)).astype(np.float32))
#ketqua = (np.random.randint(0,2,(100,1)).astype(np.float32))
##print(ketqua);
##for i in range(10):
##    plt.scatter(a[i][0],a[i][1],a[i][2],'yellow','<')
##for i in range(10):
##    plt.scatter(a[i+10][0],a[i+10][1],a[i+10][2],'red','>')

##plt.show()

#retval	=	cv2.ml.SVM_create(	)
#retval.setType(100)
#retval.setKernel(0)
#retval.setC(1)
#retval.train(a	, 0, ketqua	)

#res = ketqua.getDefaultGrid(1)
#print(res)

# img = cv.imread('C:\\Users\\Tenshi\\Desktop\\chuky.jpg',1)

# img3 = np.uint8([[[137,135,157]]])
# hsv_img = cv.cvtColor(img3,cv.COLOR_BGR2HSV)
# print (hsv_img)
# hsv_img2 = cv.cvtColor(img,cv.COLOR_BGR2HSV)
# minValue = np.array([0,30,150])
# maxValue = np.array([255,255,255])
# mask = cv.inRange(hsv_img2,minValue,maxValue)

# final = cv.bitwise_and(img,img,mask=mask)
# #cv2.imshow('example', img2)
# #cv2.imshow('example2', img)
# cv.imshow('example3', final)
# cv.imshow('example1', img)
# #cv2.imshow('example4', subimg2)
# #cv2.imshow('example5', subimg3)
# #cv2.imshow('example5', hsv_img2)
# #cv2.imshow('example', img)
# #cv2.imwrite('new.jpg',img)
# cv.waitKey(0)
# cv.destroyAllWindows()
