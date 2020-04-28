import numpy as np
import cv2 as cv
#import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import random as rng
import operator


NTRAINING_SAMPLES = 10 # Number of training samples per class
FRAC_LINEAR_SEP = 0.9   # Fraction of samples which compose the linear separable part

# Data for visual representation
WIDTH = 512
HEIGHT = 512
I = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
trainDataSign = np.empty((10, 2*NTRAINING_SAMPLES, 4), dtype=np.float32)
labels = np.empty((2*NTRAINING_SAMPLES, 1), dtype=np.int32)
labels[0:NTRAINING_SAMPLES,:] = 1                   # Class 1
labels[NTRAINING_SAMPLES:2*NTRAINING_SAMPLES,:] = 2 # Class 2
# --------------------- 1. Set up training data randomly ---------------------------------------
for i in range(10):
    url = "H:/Github/AINhandien/ex/hoa/hoa"+str(i+1)+".jpg"
    img = cv.imread(url)
    img2 = cv.imread(url)

    img2 = cv.resize(img,(800,400))
    gray= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp= sift.detect(img2,None)
    kp,des = sift.compute(img2,kp)
    img2=cv.drawKeypoints(img2,kp,img2,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    datalist = np.empty((len(kp),4),dtype = np.float32)
    for j in range(len(kp)):
        datalist[j] = [kp[j].size,kp[j].pt[0],kp[j].pt[1],kp[j].angle]
    datalist=sorted(datalist,key=operator.itemgetter(0))
    datalist.reverse();
    for j in range(10):
        trainDataSign[j,i,0:4]=datalist[j]
for i in range(10):
    url = "H:/Github/AINhandien/ex/phat/phat"+str(i+1)+".jpg"
    img = cv.imread(url)
    img2 = cv.imread(url)

    img2 = cv.resize(img,(800,400))
    gray= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp= sift.detect(img2,None)
    kp,des = sift.compute(img2,kp)
    img2=cv.drawKeypoints(img2,kp,img2,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    datalist = np.empty((len(kp),4),dtype = np.float32)
    for j in range(len(kp)):
        datalist[j] = [kp[j].size,kp[j].pt[0],kp[j].pt[1],kp[j].angle]
    datalist=sorted(datalist,key=operator.itemgetter(0))
    datalist.reverse();
    for j in range(10):
        trainDataSign[j,i+10,0:4]=datalist[j]

rng.seed(100) # Random value generation class

# Set up the linearly separable part of the training data
nLinearSamples = int(FRAC_LINEAR_SEP * NTRAINING_SAMPLES)

## [setup1]
# Generate random points for the class 1
#------------------------ 2. Set up the support vector machines parameters --------------------
svm = np.empty((10),dtype = cv.ml_SVM)
for i in range(10):
    print('Starting training process')
## [init]
    
    svm[i] = cv.ml.SVM_create()
    svm[i].setType(cv.ml.SVM_C_SVC)
    svm[i].setC(0.1)
    svm[i].setKernel(cv.ml.SVM_LINEAR)
    svm[i].setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, int(1e7), 1e-6))
## [init]

#------------------------ 3. Train the svm ----------------------------------------------------
## [train]
    svm[i].train(trainDataSign[i], cv.ml.ROW_SAMPLE, labels)
## [train]
    print('Finished training process')

#------------------------ 4. Show the decision regions ----------------------------------------
## [show]
# green = (0,100,0)
# blue = (100,0,0)
# for i in range(I.shape[0]):
#     for j in range(I.shape[1]):
#         sampleMat = np.matrix([[j,i]], dtype=np.float32)
#         response = svm[0].predict(sampleMat)[1]

#         if response == 1:
#             I[i,j] = green
#         elif response == 2:
#             I[i,j] = blue
# ## [show]

# #----------------------- 5. Show the training data --------------------------------------------
# ## [show_data]
# thick = -1
# # Class 1
# for i in range(NTRAINING_SAMPLES):
#     px = trainDataSign[0,i,0]
#     py = trainDataSign[0,i,1]
#     cv.circle(I, (px, py), 3, (0, 255, 0), thick)

# # Class 2
# for i in range(NTRAINING_SAMPLES, 2*NTRAINING_SAMPLES):
#     px = trainDataSign[0,i,0]
#     py = trainDataSign[0,i,1]
#     cv.circle(I, (px, py), 3, (255, 0, 0), thick)
# ## [show_data]

# #------------------------- 6. Show support vectors --------------------------------------------
# ## [show_vectors]
# thick = 2
# sv = svm[0].getUncompressedSupportVectors()

# for i in range(sv.shape[0]):
#     cv.circle(I, (sv[i,0], sv[i,1]), 6, (128, 128, 128), thick)
# ## [show_vectors]

# cv.imwrite('result.png', I)                      # save the Image
# cv.imshow('SVM for Non-Linear Training Data', I) # show it to the user
# cv.waitKey()
































window = tk.Tk()
window.title("AI Nhan dien")

myLabel = tk.Label(window,text="Duong dan:   " )
myLabel2 = tk.Label(window,text="" )
window.geometry('350x200')
myLabel.grid(column = 0,row=1)
myLabel2.grid(column = 1,row=1)
url=""
def clicked_open():
    window.filename = tk.filedialog.askopenfilename(title="Select A File", filetype=(("jpg files","*.jpg"),("png files","*.png"),("all files", "*")))
    myLabel2.config(text=window.filename).pack()
bt = tk.Button(window,text="Open File", command=clicked_open)
def clicked_sift():
    img = cv.imread(window.filename)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    sift = cv.xfeatures2d.SIFT_create()
    kp = sift.detect(img,None)
    #cv.Laplacian()
#img=cv2.drawKeypoints(gray,kp)

#cv2.imwrite('sift_keypoints.jpg',img)
    kp,des = sift.compute(img,kp)

    img=cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow('ex',img)
    cv.imwrite('12.jpg',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

bt2 = tk.Button(window,text="SIFT", command=clicked_sift)
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
