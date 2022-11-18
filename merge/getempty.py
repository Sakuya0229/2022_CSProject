import numpy as np
import cv2

CameraPosition = "R"


capR = cv2.VideoCapture("video\\20221115\\b200\\empty\\CameraReader"+CameraPosition+".avi")
width = capR.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capR.get(cv2.CAP_PROP_FRAME_HEIGHT)
while(capR.isOpened()):
    retR, imgR = capR.read()
    if retR == False:
        print("Cannot read video")
        break
    cv2.imshow("result",imgR)
    cv2.waitKey(0)
    cv2.imwrite("empty"+CameraPosition+".png",imgR)
    break
cv2.destroyAllWindows()
