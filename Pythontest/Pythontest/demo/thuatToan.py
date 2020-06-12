import cv2
import numpy as np
import operator
def canny(img):
    imgContour = img.copy()
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgGray,threshold1,threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    return imgContour,imgDil

def dropImage(img,x1,y1,x2,y2):
    img_crop = img[y1:y1+y2,x1:x1+x2]
    crop_name = "C:/Github/AINhandien/Pythontest/Pythontest/demo/crop/"+str(x1)+str(x2)+str(y1)+str(y2)+"_crop.jpg"
    cv2.imwrite(crop_name, img_crop)
    return img_crop

def getContours(img,imgDil,imgContour):
    im2, contours, hierarchy= cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    listimg = []
    listlocate = []
    listarea = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x , y , w, h = cv2.boundingRect(approx)
            zone = w*h
            listarea.append((zone,x,y,w,h))
    listarea=sorted(listarea,key=operator.itemgetter(0),reverse=True)
    for cnt in listarea:
        x , y , w, h = cnt[1:5]
        a = x+w
        b = y+h
        can = True
        for i in range(len(listlocate)):
            x1 = listlocate[i][0]
            y1 = listlocate[i][1]
            a1 = listlocate[i][0] + listlocate[i][2]
            b1 = listlocate[i][1] + listlocate[i][3]
            if (x1 <= x and y1 <= y and a1 >= a and b1 >= b):
                can = False
                break
            if (x>=x1 and y>=y1 and x <= a1 and y<=b1):
                listlocate[i][2] = max(a,a1) - x1
                listlocate[i][3] = max(b,b1) - y1
                can = False
                break
            if (x>=x1 and b>=y1 and (x<=a1 or b<=b1)):
                listlocate[i][0] = x1
                listlocate[i][1] = y
                listlocate[i][2] = max(a,a1) - x1
                listlocate[i][3] = max(b,b1) - y
                can = False
                break
            if (a>=x1 and y>=y1 and (a<=a1 or y<=b1)):
                listlocate[i][0] = x
                listlocate[i][1] = y1
                listlocate[i][2] = max(a,a1) - x
                listlocate[i][3] = max(b,b1) - y1
                can = False
                break
            if (a >= x1 and b>= y1 and (a<= a1 or b<= b1)):
                listlocate[i][0] = x
                listlocate[i][1] = y
                listlocate[i][2] = max(a,a1)  - x
                listlocate[i][3] = max(b,b1) - y
                can = False
                break
        if can:
            listlocate.append([x,y,w,h])
    for i in listlocate:
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]
        listimg.append(dropImage(img,x,y,w,h))
        cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)
    return listimg,imgContour
          
def detectOj(img):
    imgContour,imgDil = canny(img)
    list_img,imgContour = getContours(img,imgDil,imgContour)
    return list_img,imgContour
