import numpy as np
import cv2

capR = cv2.VideoCapture("output2.avi")
width = capR.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capR.get(cv2.CAP_PROP_FRAME_HEIGHT)
# capR.get(cv2.CAP_PROP_FPS)

CameraPosition = ""
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output3.avi",fourcc, 60.0, (int(width),int(height)))

while(capR.isOpened()):
    retR, imgR = capR.read()
    if retR == False:
        print("Cannot read video")
        break
    cv2.imshow("result",imgR)
    out.write(imgR)
    cv2.waitKey(1)
cv2.destroyAllWindows()
