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
# window = tk.Tk()
# window.title("AI Nhan dien")

# myLabel = tk.Label(window,text="Duong dan:   " )
# myLabel2 = tk.Label(window,text="" )
# window.geometry('350x200')
# myLabel.grid(column = 0,row=1)
# myLabel2.grid(column = 1,row=1)
# myLabel3 = tk.Label(window,text="" )
# myLabel3.grid(column = 1,row=5)
# filelist = []
def clicked_open(window,filelist,myLabel2):
    window.filename = tk.filedialog.askopenfilenames(title="Select A File", filetype=(("all files", "*"),("jpg files","*.jpg"),("png files","*.png"),("jpeg files","*.jpeg")))
    myLabel2.config(text=window.filename)
    filelist.clear()
    for i in range(len(window.filename)):
        filelist.append(window.filename[i])

def nguongtongquat(img):
    #img = cv.adaptiveThreshold(img,255,cv.THRESH_BINARY,cv.THRESH_BINARY,101,1)
    T2 = cv.threshold(img,127,255,cv.THRESH_OTSU)
    # #
    # img[img == 1]=255 
    # T1 =0
    # T2 =0
    # muc = 5
    # while muc >0:
    #     T1=T2
    #     g1 = 0
    #     g2 = 0
    #     gray1 =0
    #     gray2 = 0
    #     for i in range(img.shape[0]):
    #         for j in range(img.shape[1]):
    #             if img[i][j] >= T2:
    #                 gray1 += img[i][j]
    #                 g1+=1
    #             else:
    #                 gray2 += img[i][j]
    #                 g2+=1
    #     if g1 != 0:
    #         gray1 = gray1 / g1
    #     if g2 != 0:
    #         gray2 = gray2 / g2
        
    #     if g2 == 0: T2 = gray1/2
    #     elif g1 == 0: T2 = gray2/2
    #     else: T2 = (gray1+gray2)/2
    #     muc = T2-T1
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if img[i][j]>=T2[0]:
    #             img[i][j] = 255
    #         else:
    #             img[i][j]=0
    # cv.imshow('ex2',img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    #return img
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
def rotate(image, angle, center = None, scale = 1.0): 
    (h, w) = image.shape[:2] 

    if center is None: 
     center = (w/2, h/2) 

    # Perform the rotation 
    M = cv.getRotationMatrix2D(center, angle, scale) 
    rotated = cv.warpAffine(image, M, (w, h)) 

    return rotated 
def rotateImage(image, angle): 
    row,col = image.shape 
    center=tuple(np.array([row,col])/2) 
    rot_mat = cv.getRotationMatrix2D(center,angle,1.0) 
    new_image = cv.warpAffine(image, rot_mat, (col,row)) 
    return new_image 
def donganh(img):
    a = np.array(([-1,-1,-1],[-1,-1,-1],[-1,-1,-1]))
    img = cv.erode(img,a)
    img = cv.dilate(img,a)
    return img
def xulyanh(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img = nguongtongquat(img)
    img = locanh(img)
    img = catanh(img)
    if img.shape[0] > img.shape[1]: img = ndimage.rotate(img,90)
    img = donganh(img)
    cv.imshow('ex2',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return img
    
def clicked_xuly(filelist):
    if len(filelist) != 0:
        for i in range(len(filelist)):
            img = cv.imread(filelist[i])
            img = xulyanh(img)
            cv.imwrite(str(i)+".jpg",img) 


# bt = tk.Button(window,text="Open File", command=clicked_open)

    
# bt2 = tk.Button(window,text="Xu ly", command=clicked_xuly)
# bt2.grid(column = 0,row = 3)
# bt.grid(column =0,row = 0)
# window.mainloop()   