
from matplotlib import pyplot as plt
import numpy as np
import cv2
import imutils




im = cv2.imread('vlcsnap-2022-06-30-20h31m33s220.png')

im = imutils.resize(im, height = 500)

imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#imgray = cv2.medianBlur(imgray, 11)

thresh = imgray
thresh = cv2.Canny(imgray,75, 200)

#cv2.imwrite('Canny.jpg', thresh)


contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
result = im.copy()
cv2.drawContours(result, cnts, -1,(0,255,0),2)



tmp=np.zeros(imgray.shape,np.uint8)
cv2.drawContours(tmp, cnts, -1,255,2)


#cv2.imshow('tmp', tmp)
#cv2.waitKey(0)

tmp = np.float32(tmp)
dst = cv2.cornerHarris(src=tmp,blockSize=11,ksize=11,k=0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
dst = np.uint8(dst)

ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(tmp,np.float32(centroids),(5,5),(-1,-1),criteria)
print (corners)
for i in range(1, len(corners)):
    result=cv2.putText(result, f"{int(corners[i][0]),int(corners[i][1])}", (int(corners[i][0]), int(corners[i][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.circle(result, (int(corners[i][0]), int(corners[i][1])), 3, (1, 227, 254), -1)





cv2.imshow('All contours', result)
cv2.waitKey(0)
cv2.imwrite('contour-test.jpg', result)
