import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
capR = cv2.VideoCapture("CameraReaderR.avi")
capL = cv2.VideoCapture("CameraReaderL.avi")
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# fig, ax = plt.subplots()
# ax.set_xlim((-1000, 1000))
# ax.set_ylim((-1000, 1000))
distanceL = -1
distanceR = -1

width = capR.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capR.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(width,height)
#
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output2.avi',fourcc, 20.0, (int(2*width),int(height)))

fR = cv2.FileStorage("Calibration46.xml", cv2.FILE_STORAGE_READ)
intrinsicR = fR.getNode("intrinsic").mat()
distortionR = fR.getNode("distortion").mat()

fL = cv2.FileStorage("Calibration78.xml", cv2.FILE_STORAGE_READ)
intrinsicL = fL.getNode("intrinsic").mat()
distortionL = fL.getNode("distortion").mat()

cameraLposx = 92
cameraLposy = 372
cameraRposx = 92
cameraRposy = 570

Rbody_h = 180
Rbody_w = 80
Lbody_h = 180
Lbody_w = 50
RobjectPoints = np.array([ [ 0,  0, 0],
                          [ 0,  Rbody_w/2, 0],
                          [ 0,  Rbody_w, 0],
                          [ Rbody_h/2,  0, 0],
                          [ Rbody_h/2,  Rbody_w/2, 0],
                          [ Rbody_h/2,  Rbody_w, 0],
                          [   Rbody_h,           0, 0],
                          [   Rbody_h,  Rbody_w/2, 0],
                          [   Rbody_h,    Rbody_w, 0]], dtype=float)
LobjectPoints = np.array([ [ 0,  0, 0],
                          [ 0,  Lbody_w/2, 0],
                          [ 0,  Lbody_w, 0],
                          [ Lbody_h/2,  0, 0],
                          [ Lbody_h/2,  Lbody_w/2, 0],
                          [ Lbody_h/2,  Lbody_w, 0],
                          [   Lbody_h,           0, 0],
                          [   Lbody_h,  Lbody_w/2, 0],
                          [   Lbody_h,    Lbody_w, 0]], dtype=float)
flag = 0
ans = 0

def get_intersections(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d=math.sqrt((x1-x0)**2 + (y1-y0)**2)

    # non intersecting
    if d > r0 + r1 :
        print("non intersecting")
        return None
    # One circle within other
    if d < abs(r0-r1):
        print("One circle within other")
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        print("One circle within other")
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d
        y2=y0+a*(y1-y0)/d
        x3=x2+h*(y1-y0)/d
        y3=y2-h*(x1-x0)/d

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d

        return (x3, y3, x4, y4)


while(capR.isOpened() and capL.isOpened()):
    retR, imgR = capR.read()
    retL, imgL = capL.read()
    if retR == False or retL == False:
        print("Cannot read video")
        break
    # img = cv2.UMat(img)
    # img = hisEqulColor(img)
    if flag%5 ==0:
        rectsR, weightsR = hog.detectMultiScale(imgR,winStride=(12, 8), scale=1.05,useMeanshiftGrouping = False)
        rectsL, weightsL = hog.detectMultiScale(imgL,winStride=(12, 8), scale=1.05,useMeanshiftGrouping = False)
        # rects, weights = hog.detectMultiScale(img,winStride=(40, 24), scale=1.05,useMeanshiftGrouping = False)
        for body_i in rectsR:
            (Rx, Ry, Rw, Rh) = body_i
            if Rh > 400 and Rw > 200:
                Rx1 = Rx
                Ry1 = Ry
                Rx2 = Rx+Rw
                Ry2 = Rx2 + Rh
                RimagePoints  = np.array([[Ry1, Rx1],
                                         [Ry1, (Rx1+Rx2)/2],
                                         [Ry1, Rx2],
                                         [(Ry1+Ry2)/2, Rx1],
                                         [(Ry1+Ry2)/2, (Rx1+Rx2)/2],
                                         [(Ry1+Ry2)/2, Rx2],
                                         [Ry2, Rx1],
                                         [Ry2, (Rx1+Rx2)/2],
                                         [Ry2, Rx2]]).round()
                retval, rvec, tvec = cv2.solvePnP(RobjectPoints, RimagePoints, intrinsicR, distortionR)
                distanceR = tvec[2]
                distanceR = int(distanceR)
        for body_i in rectsL:
            (Lx, Ly, Lw, Lh) = body_i
            if Lh > 400 and Lw > 200:
                Lx1 = Lx
                Ly1 = Ly
                Lx2 = Lx+Lw
                Ly2 = Lx2 + Lh
                LimagePoints  = np.array([[Ly1, Lx1],
                                         [Ly1, (Lx1+Lx2)/2],
                                         [Ly1, Lx2],
                                         [(Ly1+Ly2)/2, Lx1],
                                         [(Ly1+Ly2)/2, (Lx1+Lx2)/2],
                                         [(Ly1+Ly2)/2, Lx2],
                                         [Ly2, Lx1],
                                         [Ly2, (Lx1+Lx2)/2],
                                         [Ly2, Lx2]]).round()
                retval, rvec, tvec = cv2.solvePnP(LobjectPoints, LimagePoints, intrinsicL, distortionL)
                print(rvec)
                distanceL = tvec[2]
                distanceL = int(distanceL)
    if distanceL != -1 and distanceR != -1:
        points= get_intersections(cameraLposx,cameraLposy,distanceL,cameraRposx,cameraRposy,distanceR)
        if points is not None:
            if points[0] > points[2]:
                px,py = points[0],points[1]
            else:
                px,py = points[2],points[3]

    # plt.ion()
    # circle1 = plt.Circle((cameraLposx, cameraLposx), distanceL, color='b', fill=False)
    # circle2 = plt.Circle((cameraRposx, cameraRposx), distanceR, color='b', fill=False)
    # ax.add_patch(circle1)
    # ax.add_patch(circle2)
    # plt.draw()
    # circle1.remove()
    # circle2.remove()


    for body_i in rectsR:
        (x, y, w, h) = body_i
        if h > 400 and w > 200:
            cv2.rectangle(imgR, (x, y), (x + w, y + h), (0, 0, 255), 2)
            imgR = cv2.putText(imgR, f"{distanceR}", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
            if distanceL != -1 and distanceR != -1 and points is not None:
                imgR = cv2.putText(imgR, f"{round(px,3),round(py,3)}", (x+w//2, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
    for body_i in rectsL:
        (x, y, w, h) = body_i
        if h > 400 and w > 200 :
            cv2.rectangle(imgL, (x, y), (x + w, y + h), (0, 0, 255), 2)
            imgL = cv2.putText(imgL, f"{distanceL}", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
            if distanceL != -1 and distanceR != -1 and points is not None:
                imgL = cv2.putText(imgL, f"{round(px,3),round(py,3)}", (x+w//2, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
    flag +=1
    # cv2.imshow("result",imgL)
    # img = cv2.UMat.get(img)

    cv2.namedWindow("AAA",0)
    cv2.resizeWindow("AAA", 1600, 900)
    img = np.hstack((imgL,imgR))
    cv2.imshow("AAA", img)
    out.write(img)
    cv2.waitKey(15)
    # plt.show()
capR.release()
capL.release()
# out.release()
print(ans)
cv2.destroyAllWindows()
