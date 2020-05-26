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
kp,des = sift.compute(img,kp)
count = 0
datagoc = []
img=cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('ex2',img)
cv.waitKey(0)
cv.destroyAllWindows()