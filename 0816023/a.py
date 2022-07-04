import cv2
import numpy as np
cap = cv2.VideoCapture("CameraReaderR (2).avi")
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width,height)
#
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output2.avi',fourcc, 20.0, (int(width),int(height)))
f = cv2.FileStorage("Calibration46.xml", cv2.FILE_STORAGE_READ)
intrinsic = f.getNode("intrinsic").mat()
distortion = f.getNode("distortion").mat()

body_h = 180
body_w = 100
objectPoints_b = np.array([ [ 0,  0, 0],
                          [ 0,  body_w/2, 0],
                          [ 0,  body_w, 0],
                          [ body_h/2,  0, 0],
                          [ body_h/2,  body_w/2, 0],
                          [ body_h/2,  body_w, 0],
                          [   body_h,           0, 0],
                          [   body_h,  body_w/2, 0],
                          [   body_h,    body_w, 0]], dtype=float)
flag = 0
ans = 0
while(cap.isOpened()):
    ret, img = cap.read()
    if ret == False:
        print("Cannot read video")
        break
    # img = cv2.UMat(img)
    # img = hisEqulColor(img)
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
    # out.write(img)
    # img = cv2.UMat.get(img)
    cv2.imshow("result",img)
    cv2.waitKey(15)
cap.release()
# out.release()
print(ans)
cv2.destroyAllWindows()
