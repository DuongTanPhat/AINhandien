import numpy as np
import cv2 as cv
#import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import random as rng
import operator
import math
import os
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
def khoangcachchenhlech(x1,y1,x2,y2):
    kc = 0
    kc = abs(x2-x1)+abs(y2-y1)
    #kc = math.sqrt(math.pow(x2-x1,2)+math.pow(y2-y1,2))
    return kc
def read_data():
    X = [] #chứa image
    Y = [] #chứa label
    image_descriptors = []
    for i in range(10):
        url = "H:/Github/AINhandien/ex/phat/phat"+str(i+1)+".jpg"
        img = cv.imread(url)
        img = cv.resize(img,(800,400))
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        image_descriptors.append(docdactrung1(url))
        X.append(img)
        Y.append(2)
    for i in range(10):
        url = "H:/Github/AINhandien/ex/hoa/hoa"+str(i+11)+".jpg"
        img = cv.imread(url)
        img = cv.resize(img,(800,400))
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        image_descriptors.append(docdactrung1(url))
        X.append(img)
        Y.append(1)
    # for label in os.listdir('path_to_image'):
    #     for img_file in os.listdir(os.path.join('path_to_image', label)):
    #         img = load_image(os.path.join('trainingset', label, img_file))
    #         X.append(img)
    #         Y.append(label2id[label])
    return X,Y,image_descriptors
def extract_sift_features(X):
  image_descriptors = []
  sift = cv.xfeatures2d.SIFT_create()
  for i in range(len(X)):
      _, des = sift.detectAndCompute(X[i], None)
      image_descriptors.append(des)
  return image_descriptors 
import time

def kmeans_bow(image_descriptors, num_clusters):
    start = time.time()
    bow_dict = []
    kmeans = []
    for i in range(20) :
        kmeans.append(KMeans(n_clusters=num_clusters, n_jobs = -1, verbose = 1).fit(image_descriptors[i]))
        bow_dict.append(kmeans[i].cluster_centers_)
    print('process time: ', time.time() - start)
    return bow_dict
def create_features_bow(image_descriptors, BoW, num_clusters):
    X_features = []
    for i in range(len(image_descriptors)):
       features = np.array([0] * num_clusters)
       if image_descriptors[i] is not None:
           distance = cdist(image_descriptors[i], BoW[i])
           argmin = np.argmin(distance, axis=1)
           for j in argmin:
               features[j] += 1
       X_features.append(features)
    return X_features

def docdactrung1(url):
    img = cv.imread(url)
    img = cv.resize(img,(800,400))
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    dst = cv.cornerHarris(gray,2,5,0.04)
    sift = cv.xfeatures2d.SIFT_create()
    #sift = cv.xfeatures2d.SURF_create()
    kp,des = sift.detectAndCompute(gray,None)
    #kp,des = sift.compute(gray,kp)
    img[dst>0.04*dst.max()] = [0,255,0]
    count = 0
    datagoc = []
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
    des2 = []
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
                des2.append(des[kc[num][2]])
                listed[kc[num][2]]=1
                can = False
            num += 1

    # for i in range(count):
    #     cv.circle(img,(int(listkp[i][1]),int(listkp[i][2])),3,(255,0,0),2)
    # cv.imshow('ex2',img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    #listkp = np.array(listkp)
    # img=cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # 
    
    return des2
def docdactrungnhieu():
    listnew = []
    trainDataSign = np.zeros((20,2*10, 4), dtype=np.int32)
    listnew = trainDataSign
    for i in range(10):
        url = "H:/Github/AINhandien/ex/hoa/hoa"+str(i+11)+".jpg"
        listkp = docdactrung1(url)

        #listnew[j].append(listkp[j])
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
num_clusters = 10
X,Y,image_descriptors = read_data()
#image_descriptors = extract_sift_features(X)
BoW =  kmeans_bow(image_descriptors,num_clusters)
X_features = create_features_bow(image_descriptors, BoW, num_clusters)
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(1)
svm.setGamma(1)
svm.setCoef0(0)
svm.setDegree(2)
svm.setKernel(cv.ml.SVM_POLY)
svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, int(1e7), 1e-6))
Y = np.array(Y,dtype = np.int32)
X_features = np.array(X_features, dtype = np.float32)
svm.train(X_features, cv.ml.ROW_SAMPLE, Y)
# print("Dang load data")
# listnew,labellist = docdactrungnhieu()
# print("Dang train")
# listnew = np.array(listnew, dtype = np.float32)
# labellist = np.array(labellist, dtype = np.int32)
# svm = cv.ml.SVM_create()
# svm.setType(cv.ml.SVM_C_SVC)
# svm.setC(1)
# svm.setGamma(0.1)
# svm.setCoef0(0)
# svm.setDegree(2)
# svm.setKernel(cv.ml.SVM_RBF)
# svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, int(1e7), 1e-6))
# svm.train(listnew, cv.ml.ROW_SAMPLE, labellist)
# print("Da train xong")
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
    #dactrung = docdactrung1(window.filename)
    url = window.filename
    img = cv.imread(url)
    img = cv.resize(img,(800,400))
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    hoa  =0
    phat =0
    dung =0
    sift = cv.xfeatures2d.SIFT_create()
    _, des = sift.detectAndCompute(img, None)
    kmeans=(KMeans(n_clusters=num_clusters, n_jobs = -1, verbose = 1).fit(des))
    bow_dict=kmeans.cluster_centers_
    features = np.array([0] * num_clusters)
    if des is not None:
           distance = cdist(des, bow_dict)
           argmin = np.argmin(distance, axis=1)
           for j in argmin:
               features[j] += 1
    dactrung = features
    sample = np.matrix([dactrung],dtype = np.float32)
    res = svm.predict(sample)[1]
    print(res)
    #respond = np.empty((len(dactrung), 1), dtype=np.int32)
    # for i in range(dactrung.__len__()):
    #     sampleMat = np.matrix([dactrung[i]], dtype=np.float32)
    #     respond[i]=svm.predict(sampleMat)[1]
    #     if respond[i]==1 :
    #         hoa+=1
    #     elif respond[i]==2:
    #         phat+=1
    #     else:
    #         dung+=1
    # if max(hoa,phat,dung) == hoa:
    #     myLabel3.config(text="Chữ ký của Hòa "+str(hoa*(100/len(dactrung)))+"%")
    # elif max(hoa,phat,dung) == phat:
    #     myLabel3.config(text="Chữ ký của Phát "+str(phat*(100/len(dactrung)))+"%")
    # else:
    #     myLabel3.config(text="Chữ ký của Dũng "+str(dung*(100/len(dactrung)))+"%")
    # print(respond[0:len(dactrung)])
    

bt2 = tk.Button(window,text="Nhan dien", command=clicked_sift)
bt2.grid(column = 0,row = 3)
bt.grid(column =0,row = 0)
window.mainloop()    
