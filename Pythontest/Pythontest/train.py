import numpy as np
import cv2 as cv
#import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import random as rng
import operator

img = cv.imread("H:/Github/AINhandien/ex/phat/phat8.jpg")
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#img = cv.resize(img2,(800,400))    
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