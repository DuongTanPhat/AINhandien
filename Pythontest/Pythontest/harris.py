import numpy as np
import matplotlib.pyplot as plt
from PIL.Image import *
from scipy import ndimage
import cv2

# Hàm phát hiện góc bằng thuật toán Harris
def Harris(img):
    # Tính đạo hàm theo trục Ox và Oy
	ix = ndimage.sobel(img, 0)
	iy = ndimage.sobel(img, 1)
	
    # Tính các thành phần A, C, B
	ix2 = ix * ix
	iy2 = iy * iy
	ixy = ix * iy

    # Lọc nhiễu A, B, C bằng bộ lọc Gaussian
	ix2 = ndimage.gaussian_filter(ix2, sigma=2)
	iy2 = ndimage.gaussian_filter(iy2, sigma=2)
	ixy = ndimage.gaussian_filter(ixy, sigma=2)

	result = np.zeros((img.shape[0], img.shape[1]))
	r = np.zeros((img.shape[0], img.shape[1]))
	rmax = 0

    # Tìm ma trận r gồm điểm f của từng pixel 
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			m = np.array([[ix2[i, j], ixy[i, j]], [ixy[i, j], iy2[i, j]]], dtype=np.float64)
			r[i, j] = np.linalg.det(m) - 0.06 * (np.power(np.trace(m), 2))
			if r[i, j] > rmax:
				rmax = r[i, j]
	
    # Trực quan hóa việc phát hiện góc với ngưỡng bằng 0.01*max[r] với khung cửa sổ 3x3
	for i in range(img.shape[0] - 1):
		for j in range(img.shape[1] - 1):
			if r[i, j] > 0.06 * rmax \
            and r[i, j] > r[i-1, j-1] \
            and r[i, j] > r[i-1, j+1] \
            and r[i, j] > r[i+1, j-1] \
            and r[i, j] > r[i+1, j+1]:
				result[i, j] = 1

	pc, pr = np.where(result == 1)
	plt.plot(pr, pc, 'rp')
	plt.imshow(im, 'gray')
	plt.show()


im = cv2.imread("H:/Github/AINhandien/ex/hoa/2.jpg")
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64')
Harris(gray)