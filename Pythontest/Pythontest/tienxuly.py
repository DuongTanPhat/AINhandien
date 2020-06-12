import numpy as np
import cv2 as cv
#import matplotlib.pyplot as plt
from scipy import ndimage
import tkinter as tk
from tkinter import filedialog
import random as rng
import operator
import math
import os
import time
import demo.thuatToan
def nguongtongquat(img):
    #img = cv.adaptiveThreshold(img,255,cv.THRESH_BINARY,cv.THRESH_BINARY,101,1)
    T2 = cv.threshold(img,127,255,cv.THRESH_OTSU)
    return T2[1]
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
    return img
def xulyanh(img):
    # img = cv.resize(img,(800,400))
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img = cv.blur(img,ksize = (5,5))
    img = nguongtongquat(img)
    img = locanh(img)
    img = catanh(img)
    if img.shape[0] > img.shape[1]: img = ndimage.rotate(img,90)
    img = donganh(img)
    return img
