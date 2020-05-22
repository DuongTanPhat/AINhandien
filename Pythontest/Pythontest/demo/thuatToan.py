import cv2
import numpy as np

img = cv2.imread('test.jpg')
imgContour = img.copy()
imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
imgCanny = cv2.Canny(imgGray,threshold1,threshold2)
kernel = np.ones((5, 5))
imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

def dropImage(x1,y1,x2,y2):
    img_crop = img[y1:y1+y2,x1:x1+x2]
    crop_name = 'crop/'+str(x1)+str(x2)+str(y1)+str(y2)+'_crop.jpg'
    cv2.imwrite(crop_name, img_crop)

def getContours(img,imgContour):
    im2, contours, hierarchy= cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x , y , w, h = cv2.boundingRect(approx)
            dropImage(x,y,w,h)
            cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)
          

getContours(imgDil,imgContour)
cv2.imshow("Result", imgContour)
cv2.waitKey(0)
cv2.destroyAllWindows()


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