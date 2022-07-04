import cv2
from cv2 import UMat
import dlib
import numpy as np


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

f = cv2.FileStorage("Calibration.xml", cv2.FILE_STORAGE_READ)
intrinsic = f.getNode("intrinsic").mat()
distortion = f.getNode("distortion").mat()

objectPoints_b = np.zeros((3*3, 3), np.float32)
objectPoints_b[:, :2] =  np.mgrid[0:101:50, 0: 191:85].T.reshape(-1, 2)



def main():
    flag = 0
    while True:
        ret, img = cap.read()
        if ret != True :
            print("can't read!")
            exit(1)


        #detect body
        if flag%5 ==0:
            rects, weights = hog.detectMultiScale(img,winStride=(12, 8), scale=1.05,useMeanshiftGrouping = False)
            # rects, weights = hog.detectMultiScale(img,winStride=(40, 24), scale=1.05,useMeanshiftGrouping = False)
            for body_i in rects:
                (x, y, w, h) = body_i
                if h > 400 and w > 200:
                    x1 = x
                    y1 = y
                    x2 = x+w
                    y2 = x2 + h
                    imagePoints_b  = np.array([[y1, x1],
                                             [y1, (x1+x2)/2],
                                             [y1, x2],
                                             [(y1+y2)/2, x1],
                                             [(y1+y2)/2, (x1+x2)/2],
                                             [(y1+y2)/2, x2],
                                             [y2, x1],
                                             [y2, (x1+x2)/2],
                                             [y2, x2]]).round()
                    retval, rvec, tvec = cv2.solvePnP(objectPoints_b, imagePoints_b, intrinsic, distortion)
                    distance = tvec[2]
                    distance = int(distance)
            for body_i in rects:
                (x, y, w, h) = body_i
                if h > 400 and w > 200:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    img = cv2.putText(img, f"{distance}", (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
            flag +=1

        cv2.imshow("cap", img)
        key = cv2.waitKey(15)
        if key != -1:
            cv2.destroyAllWindows()
            break

main()

# objectPoints = np.array([[ [0], [0], [0]],
#                                      [ [0], [10], [0]],
#                                      [ [0], [20], [0]],
#                                      [[10],  [0], [0]],
#                                      [[10], [10], [0]],
#                                      [[10], [20], [0]],
#                                      [[20],  [0], [0]],
#                                      [[20], [10], [0]],
#                                      [[20], [20], [0]]], dtype=float)
#             print(objectPoints.shape)
#             objectPoints = cv2.cvtColor(objectPoints, cv2.COLOR_GRAY2BGR)
#             imagePoints  = np.array([[[y1], [x1]],
#                                      [[y1], [(x1+x2)/2]],
#                                      [[y1], [x2]],
#                                      [[(y1+y2)/2], [x1]],
#                                      [[(y1+y2)/2], [(x1+x2)/2]],
#                                      [[(y1+y2)/2], [x2]],
#                                      [[y2], [x1]],
#                                      [[y2], [(x1+x2)/2]],
#                                      [[y2], [x2]]]).round()
#             imagePoints = cv2.cvtColor(imagePoints, cv2.COLOR_GRAY2BGR)
