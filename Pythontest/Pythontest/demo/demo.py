import cv2
import numpy as np

img=cv2.imread('H:\\Datasign\\Duong\\33.jpg')
img2 = cv2.resize(img,(800,400))
cv2.imshow("imge origan", img2)
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

def filterColumns(array):
    res = list()
    for j in range(0, len(array[0])): 
        tmp = 0
        for i in range(0, len(array)): 
            tmp = tmp + array[i][j]
        res.append(tmp)
    return res

def filterRows(array):
    res = list()
    for j in range(0, len(array)): 
        tmp = 0
        for i in range(0, len(array[0])): 
            tmp = tmp + array[j][i]
        res.append(tmp)
    return res
def getLine(array):
    count=0
    lines=list()
    for i in range(0,len(array)):
        if(count==0 and array[i]!=0):
            lines.append(i)
            count=count+1
        if(count==1 and array[i]==0):
            lines.append(i)
            count=0
    return lines
def dropImage(line1,line2,image,direction):
    w,h=image.shape
    if(direction==1):
        img_crop = image[1:w,line1:line2]
    if(direction==0):
        img_crop = image[line1:line2,1:h]
    return img_crop

def demo():
    result=filterColumns(thresh)
    getLines=getLine(result)
    for i in range(0,len(getLines),2):
        image1=dropImage(getLines[i],getLines[i+1],thresh,1)
        # cv2.imshow('dsad', image1)
        image11=filterRows(image1)
        getImage1=getLine(image11)
        if(len(getImage1)!=0):
            if(len(getImage1)%2!=0):
                getImage1.append(len(image11)-1)
            print(getImage1)
            for j in range(0,len(getImage1),2):
                image2=dropImage(getImage1[j],getImage1[j+1],image1,0)
                crop_name = str(i)+str(j)+'_crop.jpg'
                print(crop_name)
                cv2.imshow(crop_name, image2)

demo()
# result=list()
# result=filterColumns(thresh)
# getLines=getLine(result)
# img_crop = dropImage(200,324,img,1)
# crop_name = i,j,'crop.jpg'
# cv2.imshow(crop_name, img_crop)






cv2.waitKey(0)
cv2.destroyAllWindows()
# print("width:",len(thresh[0]),"height:",len(thresh))
# cộng theo chiều dọc
# res=list()
# for j in range(0, len(thresh[0])): 
#     tmp = 0
#     for i in range(0, len(thresh)): 
#         tmp = tmp + thresh[i][j]
#     res.append(tmp)
#     print(j,i,tmp)
# cộng theo chiều ngang
# res = list()
# for j in range(0, len(array)): 
#     tmp = 0
#     for i in range(0, len(array[0])): 
#         tmp = tmp + array[j][i]
#     res.append(tmp)
# return res