import numpy as np
import cv2
cap1 = cv2.VideoCapture("output4.avi")
cap2 = cv2.VideoCapture("output8.avi")
width1 = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
height1 = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
width2 = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
height2 = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter("out.avi",fourcc, 20.0, (int(width1+width2),int(height1)))
out = cv2.VideoWriter("out2.avi",fourcc, 20.0, (1600,900))

while(cap1.isOpened() and cap2.isOpened()):
    ret1, img1 = cap1.read()
    ret2, img2 = cap2.read()
    if ret1 == False:
        print("Cannot read video")
        break
    img = np.hstack((img1,img2))
    img = cv2.resize(img,(1600,900))
    cv2.imshow("AAA", img)
    out.write(img)
    cv2.waitKey(15)
cap1.release()
cap2.release()
cv2.destroyAllWindows()
