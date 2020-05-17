import numpy as np
import cv2 as cv
#import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import random as rng
import operator

img = cv.imread("H:/Github/AINhandien/ex/phat/phat2.jpg")
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d.SIFT_create()
kp = sift.detect(img,None)

kp,des = sift.compute(img,kp)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,5,0.04)
img[dst>0.04*dst.max()] = [0,255,0]
cv.imshow('ex2',img)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i,j].all==[0,255,0]:
            print(i,j)

img=cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



for j in range(len(kp)):
        datalist[j] = [kp[j].size,kp[j].pt[0],kp[j].pt[1],kp[j].angle]

cv.imshow('ex2',img)
cv.waitKey(0)
cv.destroyAllWindows()