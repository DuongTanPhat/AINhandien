import cv2
import numpy as np
import operator
#img = cv2.imread("H:/Github/AINhandien/ex/hoa/3.jpg")
# img = cv2.imread("H:/Github/AINhandien/Pythontest/Pythontest/demo/anhtest.png")
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
    crop_name = "H:/Github/AINhandien/Pythontest/Pythontest/demo/crop/"+str(x1)+str(x2)+str(y1)+str(y2)+"_crop.jpg"
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
        if area > 1500:
            # peri = cv2.arcLength(cnt, True)
            # approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # x , y , w, h = cv2.boundingRect(approx)
            # #listlocate.append((x,y,w,h))
            # listimg.append(dropImage(img,x,y,w,h))
            # cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)
            listarea.append((area,cnt))
    listarea=sorted(listarea,key=operator.itemgetter(0),reverse=True)
    for cnt in listarea:
        peri = cv2.arcLength(cnt[1], True)
        approx = cv2.approxPolyDP(cnt[1], 0.02 * peri, True)
        x , y , w, h = cv2.boundingRect(approx)
        can = True
        for i in range(len(listlocate)):
            if listlocate[i][0] <= x and listlocate[i][1] <= y and listlocate[i][0] + listlocate[i][2] >= w + x and listlocate[i][1] + listlocate[i][3] >= h + y:
                can = False
                break
        if can:
            listlocate.append((x,y,w,h))
            listimg.append(dropImage(img,x,y,w,h))
            cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)
    return listimg
          
def detectOj(img):
    imgContour,imgDil = canny(img)
    list_img = getContours(img,imgDil,imgContour)
    cv2.imshow("Result", imgContour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return list_img

#detectOj(img)
# while True:
#     img = cv2.imread('anhtest.png')
#     imgContour = img.copy()
#     imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
#     imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
#     threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
#     threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
#     imgCanny = cv2.Canny(imgGray,threshold1,threshold2)
#     kernel = np.ones((5, 5))
#     imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
#     getContours(imgDil,imgContour)
#     imgStack = stackImages(0.8,([img,imgContour]))
#     cv2.imshow("Result", imgStack)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break