from matplotlib import pyplot as plt
import numpy as np
import cv2
import imutils

CameraPosition = ""

videopath = "video\\20221115\\SmartPhone\\01\\CameraReader"+CameraPosition+".mp4"
empty_frame_path = "empty"+CameraPosition+".png"


NeedOutput = 1


pts1 = np.float32([[94,775],[442,669],[859,1033],[1094,754]])
pts2 = np.float32([[307,80 ],[564,80],[307,468],[564,468]])
M = cv2.getPerspectiveTransform(pts1,pts2)



hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


capR = cv2.VideoCapture(videopath)
width = capR.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capR.get(cv2.CAP_PROP_FRAME_HEIGHT)

empty_frame = cv2.imread(empty_frame_path)
empty_frame = cv2.resize(empty_frame,(int(width),int(height)))
empty_frame = cv2.cvtColor(empty_frame, cv2.COLOR_BGR2GRAY)
empty_frame = cv2.GaussianBlur(src=empty_frame, ksize=(5,5), sigmaX=0)
########################### Output video  #####################################################
if NeedOutput:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("output2.avi",fourcc, cv2.CAP_PROP_FPS, (int(width),int(height)))

humanpos = (0,0)
badminton = (0,0)
frame_count = 0
previous_frame = None
prepared_frame=None
humans=[]
while(capR.isOpened()):
    retR, imgR = capR.read()
    if retR == False:
        print("Cannot read video")
        break

    img_rgb = np.array(imgR)

###########################find human position #####################################################
    if frame_count % 5 == 0:
        rectsR, weightsR = hog.detectMultiScale(imgR,winStride=(12, 8), scale=1.025,useMeanshiftGrouping = False)

    for body_i in rectsR:
        (x, y, w, h) = body_i
        if h > 400 and w > 200:
            point = np.float32(np.array([[[x+w/2,y+h-50]]]))
            pos=cv2.perspectiveTransform(point,M)[0][0]
            cv2.putText(img_rgb, f"{round(pos[0],3),round(pos[1],3)}", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
            humanpos = (round(pos[0],3),round(pos[1],3))
            humans=[]
            humans.append(body_i)
    if len(humans) > 0:
        (x, y, w, h) = humans[0]
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
###########################find human position #####################################################

###########################find badminton position #####################################################


    prepared_frame = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5,5), sigmaX=0)


    diff_frame = cv2.absdiff(src1 = empty_frame, src2 = prepared_frame)

    kernel = np.ones((5, 5))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)
    thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]

    cimage,contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 100 or cv2.contourArea(contour) > 1600 :
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        NotBadminton = 0
        for human in humans:
            if (x >= human[0]+human[2]+100 or y >= human[1]+human[3]+50 or x+w <= human[0]-100 or y+h <= human[1]-50):
                continue
            else:
                NotBadminton = 1
                break
        if NotBadminton:
            continue
        cv2.rectangle(img=img_rgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
        badminton = (x, y)
###########################find badminton position #####################################################


    frame_count += 1
    cv2.namedWindow("OutputWindow",0)
    cv2.resizeWindow("OutputWindow", 1600, 900)
    cv2.imshow("OutputWindow", img_rgb)
    if NeedOutput:
        out.write(img_rgb)
    print(frame_count,humanpos,badminton)
    cv2.waitKey(1)
capR.release()
cv2.destroyAllWindows()
