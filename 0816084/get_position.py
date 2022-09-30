from matplotlib import pyplot as plt
import numpy as np
import cv2
import imutils



pts1 = np.float32([[94,775],[442,669],[859,1033],[1094,754]])
pts2 = np.float32([[307,80 ],[564,80],[307,468],[564,468]])
M = cv2.getPerspectiveTransform(pts1,pts2)



hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


capR = cv2.VideoCapture("CameraReaderR (3).avi")
width = capR.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capR.get(cv2.CAP_PROP_FRAME_HEIGHT)


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output2.avi',fourcc, 20.0, (int(width),int(height)))

frame_count = 0
previous_frame = None
prepared_frame=None
#track=[]
humans=[]


while(capR.isOpened()):
    retR, imgR = capR.read()
    if retR == False:
        print("Cannot read video")
        break

    img_rgb = np.array(imgR)

    prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5,5), sigmaX=0)

    if frame_count%5 ==0:
        rectsR, weightsR = hog.detectMultiScale(imgR,winStride=(12, 8), scale=1.05,useMeanshiftGrouping = False)

    if rectsR is not ():
        humans=[]

    for body_i in rectsR:
        (x, y, w, h) = body_i
        if h > 400 and w > 200:
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
            point = np.float32(np.array([[[x+w/2,y+h-50]]]))
            pos=cv2.perspectiveTransform(point,M)[0][0]
            im = cv2.putText(img_rgb, f"{round(pos[0],3),round(pos[1],3)}", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
            #cv2.rectangle(prepared_frame, (x, y), (x + w, y + h), (0, 0, 0), -1)
        humans.append(body_i)
    for human in humans:
        (x, y, w, h) = human
        cv2.rectangle(prepared_frame, (x, y), (x + w, y + h), (0, 0, 0), -1)






    if (previous_frame is None):
        previous_frame = prepared_frame
        continue

    diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
    previous_frame = prepared_frame

    kernel = np.ones((5, 5))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)

    thresh_frame = cv2.threshold(src=diff_frame, thresh=30, maxval=255, type=cv2.THRESH_BINARY)[1]


    contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 100 or cv2.contourArea(contour) > 1600 :
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        flag=0
        ###############################################
        for human in humans:
            if (x >= human[0]+human[2] or y >= human[1]+human[3] or x+w <= human[0] or y+h <= human[1]):
                continue
            else:
                flag=1
                break
        if flag:
            continue
        ##################################################
        cv2.rectangle(img=img_rgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
        #track.append((x,y,w,h))

    #for tr in track:
    #    (x, y, w, h) = tr
    #    cv2.circle(img_rgb, (int(x+w/2),int(y+h/2)), 3, (0, 255, 0), -1)


    frame_count += 1
    cv2.namedWindow("AAA",0)
    cv2.resizeWindow("AAA", 1600, 900)
    cv2.imshow("AAA", img_rgb)
    out.write(img_rgb)
    cv2.waitKey(15)
capR.release()
cv2.destroyAllWindows()
