import numpy as np
import cv2 as cv
#import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import random as rng
import operator
import math
import os
import time
def clicked_open(window,myLabel2,):
    filelist = []
    window.filename = tk.filedialog.askopenfilenames(title="Select A File", filetype=(("all files", "*"),("jpg files","*.jpg"),("png files","*.png"),("jpeg files","*.jpeg")))
    myLabel2.config(text=window.filename)
    filelist.clear()
    for i in range(len(window.filename)):
        filelist.append(window.filename[i])
    return filelist
def nguongtongquat(img):
    T1 =0
    T2 =0
    muc = 5
    while muc >0:
        T1=T2
        g1 = 0
        g2 = 0
        gray1 =0
        gray2 = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] >= T2:
                    gray1 += img[i][j]
                    g1+=1
                else:
                    gray2 += img[i][j]
                    g2+=1
        if g1 != 0:
            gray1 = gray1 / g1
        if g2 != 0:
            gray2 = gray2 / g2
        T2 = (gray1+gray2)/2
        muc = T2-T1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j]>=T2:
                img[i][j] = 255
            else:
                img[i][j]=0
    return img
def catanh(img):
    x1=0
    y1=0
    x2=img.shape[0]
    y2 = img.shape[1]
    xong = False
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] == 0:
                x1 = i
                xong = True
                break
        if xong == True: break
    xong = False
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if img[j][i] == 0:
                y1 = i
                xong = True
                break
        if xong == True: break
    xong = False
    for i in range(img.shape[1]-1,-1,-1):
        for j in range(img.shape[0]-1,-1,-1):
            if img[j][i] == 0:
                y2 = i
                xong = True
                break
        if xong == True: break
    xong = False
    for i in range(img.shape[0]-1,-1,-1):
        for j in range(img.shape[1]-1,-1,-1):
            if img[i][j] == 0:
                x2 = i
                xong = True
                break
        if xong == True: break
    img = img[x1:x2,y1:y2]
    return img
def locanh(img):
    img = cv.medianBlur(img,3)
    return img
def donganh(img):
    a = np.array(([-1,-1,-1],[-1,-1,-1],[-1,-1,-1]))
    img = cv.erode(img,a)
    img = cv.dilate(img,a)
    img = cv.erode(img,a)
    img = cv.dilate(img,a)
    img = cv.erode(img,a)
    img = cv.dilate(img,a)
    img = cv.erode(img,a)
    img = cv.dilate(img,a)
    return img
def xulyanh(url):
    img = cv.imread(url)
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img = nguongtongquat(img)
    img = locanh(img)
    img = locanh(img)
    img = catanh(img)
    img = donganh(img)
    return img
    
def clicked_xuly(filelist):
    if len(filelist) != 0:
        for i in range(len(filelist)):
            img = xulyanh(filelist[i])
            cv.imwrite(str(i)+".jpg",img) 
