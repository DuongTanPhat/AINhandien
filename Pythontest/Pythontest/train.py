import numpy as np
import cv2 as cv
#import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import random as rng
import operator
img = cv.imread("H:/Github/AINhandien/ex/hoa/2.jpg")
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d.SIFT_create()
kp = sift.detect(img,None)
green = np.empty((1,3),dtype=np.ndarray)
green = [0,255,0]
kp,des = sift.compute(img,kp)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,5,0.04)
img[dst>0.04*dst.max()] = [0,0,255]
k = dst.max()*0.04
count = 0
datagoc = []
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]): 
#         #if dst[i,j]>k:
#             #print(i,j)
#         if img[i,j,0]==0 and img[i,j,1]==0 and img[i,j,2]==255:

#             tiledai = i*100/img.shape[0]
#             tilerong = j*100/img.shape[1]
#             if(count==0):
#                 datagoc.append([tiledai,tilerong])
#                 count+=1
#             if(abs(tiledai-datagoc[count-1][0])>5 or abs(tilerong-datagoc[count-1][1])>5): #bo qua cac diem gan
#                 can = True
#                 for k in range(count):
#                     if(abs(tiledai-datagoc[k][0])<5 and abs(tilerong-datagoc[k][1])<5):
#                         can=False
#                         break;
#                 if can:
#                     datagoc.append([tiledai,tilerong])
#                     count+=1

# datagoc = np.array(datagoc)
img=cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# datakp = np.empty((len(kp),4),dtype = np.int32)

# for j in range(len(kp)):
#          datakp[j] = [kp[j].size,kp[j].pt[0]*100/img.shape[0],kp[j].pt[1]*100/img.shape[1],kp[j].angle]
# listkp = []
# count2 = 0
# num = 0
# listed = np.zeros((len(kp),1),dtype=np.int32)
# for i in range(count-1):
#     for j in range(len(kp)):
#         if(abs(datagoc[i,0]-datakp[j,1])<5 and abs(datagoc[i,1]-datakp[j,2])<5 and listed[j]!=1):
#             listkp.append(datakp[j])
#             listed[j] = 1
#             print(datakp[j,1],datakp[j,2])
#             count2+=1
#             num += 1
#             if(num>5): break
#     num = 0
cv.imshow('ex2',img)
cv.waitKey(0)
cv.destroyAllWindows()