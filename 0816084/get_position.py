from matplotlib import pyplot as plt
import numpy as np
import cv2
import imutils


pts1 = np.float32([[94,775],[442,669],[859,1033],[1094,754]])
pts2 = np.float32([[307,80 ],[564,80],[307,468],[564,468]])
M = cv2.getPerspectiveTransform(pts1,pts2)

badminton_haar = cv2.CascadeClassifier(r"cascade.xml")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

capR = cv2.VideoCapture("CameraReaderR.avi")

width = capR.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capR.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(width,height)
#
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output2.avi',fourcc, 20.0, (int(width),int(height)))



flag = 0

while(capR.isOpened()):
    retR, imgR = capR.read()
    if retR == False:
        print("Cannot read video")
        break
    # img = cv2.UMat(img)
    # img = hisEqulColor(img)
    if flag%5 ==0:
        rectsR, weightsR = hog.detectMultiScale(imgR,winStride=(12, 8), scale=1.05,useMeanshiftGrouping = False)
    gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    badminton = badminton_haar.detectMultiScale(gray, 1.06, 1,minSize=(10, 10))



    for body_i in rectsR:
        (x, y, w, h) = body_i
        if h > 400 and w > 200:
            cv2.rectangle(imgR, (x, y), (x + w, y + h), (0, 0, 255), 2)
            point = np.float32(np.array([[[x+w/2,y+h-50]]]))
            #point=np.array([x+w/2,y+h-10])
            pos=cv2.perspectiveTransform(point,M)[0][0]
            im = cv2.putText(imgR, f"{round(pos[0],3),round(pos[1],3)}", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
    for (x, y, w, h) in badminton:
        cv2.rectangle(imgR, (x, y), (x + w, y + h), (0, 255, 0), 2)

    flag +=1
    # cv2.imshow("result",imgL)
    # img = cv2.UMat.get(img)

    cv2.namedWindow("AAA",0)
    cv2.resizeWindow("AAA", 1600, 900)
    #img = np.hstack((imgL,imgR))
    cv2.imshow("AAA", imgR)
    out.write(imgR)
    cv2.waitKey(15)
        # plt.show()
capR.release()
    # out.release()
cv2.destroyAllWindows()
