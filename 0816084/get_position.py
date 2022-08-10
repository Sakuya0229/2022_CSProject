from matplotlib import pyplot as plt
import numpy as np
import cv2
import imutils



pts1 = np.float32([[94,775],[442,669],[859,1033],[1094,754]])
pts2 = np.float32([[307,80 ],[564,80],[307,468],[564,468]])
M = cv2.getPerspectiveTransform(pts1,pts2)


#badminton_haar = cv2.CascadeClassifier(r"cascade.xml")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


capR = cv2.VideoCapture("CameraReaderR.avi")
width = capR.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capR.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(width,height)
#
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output2.avi',fourcc, 20.0, (int(width),int(height)))

frame_count = 0
previous_frame = None
prepared_frame=None

while(capR.isOpened()):
    retR, imgR = capR.read()
    if retR == False:
        print("Cannot read video")
        break

        # 1. Load image; convert to RGB
    img_rgb = np.array(imgR)
    #img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)
    #gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)



    if frame_count%5 ==0:
        rectsR, weightsR = hog.detectMultiScale(imgR,winStride=(12, 8), scale=1.05,useMeanshiftGrouping = False)
    for body_i in rectsR:
        (x, y, w, h) = body_i
        if h > 400 and w > 200:
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
            point = np.float32(np.array([[[x+w/2,y+h-50]]]))
            #point=np.array([x+w/2,y+h-10])
            pos=cv2.perspectiveTransform(point,M)[0][0]
            im = cv2.putText(img_rgb, f"{round(pos[0],3),round(pos[1],3)}", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(prepared_frame, (x, y), (x + w, y + h), (0, 0, 0), -1)
    #if ((frame_count % 2) == 0):

        # 2. Prepare image; grayscale and blur
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5,5), sigmaX=0)


        # 3. Set previous frame and continue if there is None
    if (previous_frame is None):
        # First frame; there is no previous one yet
        previous_frame = prepared_frame
        continue

        # calculate difference and update previous frame
    diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
    previous_frame = prepared_frame

    # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
    kernel = np.ones((5, 5))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)

    # 5. Only take different areas that are different enough (>20 / 255)
    thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]


    contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 1:
            # too small: skip!
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(img=img_rgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)



    frame_count += 1
    cv2.namedWindow("AAA",0)
    cv2.resizeWindow("AAA", 1600, 900)
    cv2.imshow("AAA", img_rgb)
    out.write(img_rgb)
    cv2.waitKey(15)
        # plt.show()
capR.release()
    # out.release()
cv2.destroyAllWindows()
