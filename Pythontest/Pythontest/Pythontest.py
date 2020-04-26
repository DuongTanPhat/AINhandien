import numpy as np
import cv2 as cv
#import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog
window = tk.Tk()
window.title("AI Nhan dien")

myLabel = tk.Label(window,text="Nhap duong dan: " )
myLabel2 = tk.Label(window,text="" )
window.geometry('350x200')
myLabel.grid(column = 0,row=1)
myLabel2.grid(column = 0,row=4)
url=""
def clicked_open():
    window.filename = tk.filedialog.askopenfilename(title="Select A File", filetype=(("jpg files","*.jpg"),("png files","*.png"),("all files", "*")))
    myLabel2.config(text=window.filename).pack()
bt = tk.Button(window,text="Open File", command=clicked_open)
def clicked_sift():
    img = cv.imread(window.filename)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.Laplacian(gray,img)
    sift = cv.xfeatures2d.SIFT_create()
    kp = sift.detect(img,None)
#cv.Laplacian()
#img=cv2.drawKeypoints(gray,kp)

#cv2.imwrite('sift_keypoints.jpg',img)
    kp,des = sift.compute(img,kp)


    img=cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow('ex',img)
    cv.imwrite('11222.jpg',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
 
bt2 = tk.Button(window,text="SIFT", command=clicked_sift)
bt2.grid(column = 3,row = 3)
bt.grid(column =1,row = 0)
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

#img = cv2.imread('C:\\Users\\Tenshi\\Desktop\\backgroundj.jpg',1)
#img2 = cv2.imread('C:\\Users\\Tenshi\\Desktop\\ccc.jpg',1)
#cv2.line(img,(0,0),(400,300),(255,0,0),5)

#print (img.size)
#subimg = img[100:300,50:250]
#subimg2 = img2[100:300,130:330]
#subimg3 = cv2.add(subimg,subimg2)
#img3 = np.uint8([[[130,175,212]]])
#hsv_img = cv2.cvtColor(img3,cv2.COLOR_BGR2HSV)
#print (hsv_img)
#hsv_img2 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#minValue = np.array([10,40,20])
#maxValue = np.array([60,120,350])
#mask = cv2.inRange(hsv_img2,minValue,maxValue)

#final = cv2.bitwise_and(img,img,mask=mask)
##cv2.imshow('example', img2)
##cv2.imshow('example2', img)
#cv2.imshow('example3', final)
##cv2.imshow('example4', subimg2)
##cv2.imshow('example5', subimg3)
##cv2.imshow('example5', hsv_img2)
##cv2.imshow('example', img)
##cv2.imwrite('new.jpg',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
