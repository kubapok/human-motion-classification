import numpy as np
import cv2
import sys
from ROI import ROI

face_cascade = cv2.CascadeClassifier('/home/kuba/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/kuba/opencv/data/haarcascades/haarcascade_eye.xml')


cap = cv2.VideoCapture(sys.argv[1])

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

i = 0
ROIframe = np.zeros((480, 640), dtype=np.uint8)
mask = np.zeros((480, 640), dtype=np.uint8)
ret, frame = cap.read()
prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
lindex, rindex, uindex, dindex, = 0, 10, 0, 10
i = 0
while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)  # w skali szarości 0-255 ale wiekszosc skrajne
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = frame_gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey),
                          (ex + ew, ey + eh), (0, 255, 0), 2)
    if np.count_nonzero(fgmask) > 100:
        lindex, rindex, dindex, uindex = ROI.get_POI_corners(fgmask, 0.02)
        if lindex and rindex and dindex and uindex:
            mask = np.zeros((480, 640), dtype=np.uint8)
            mask[dindex: uindex, lindex:rindex] = 255
            ROIframe_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)
            flow = cv2.calcOpticalFlowFarneback(
                prvs, ROIframe_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            ROIonly_flow_gray = flow[dindex: uindex, lindex:rindex]
            vertical_movement, horizontal_movement = cv2.split(
                ROIonly_flow_gray)
            print(vertical_movement.sum(), '\t' * 5, horizontal_movement.sum())
            prvs = ROIframe_gray
            i += 1
            cv2.line(frame, (400, 100), (400 + int(vertical_movement.sum() / 1000),
                                         100 + int(horizontal_movement.sum() / 1000)), (255, 0, 0), 5)
            cv2.line(frame, (400, 100), (400, 100), (0, 0, 255), 5)
            if i == 60:
                pass
    cv2.rectangle(frame, (lindex, dindex),
                  (rindex, uindex), (0, 255, 0), 3)
    # cv2.rectangle(frame, (240, 320),
    #               (242, 322), (0, 0, 255), 3)
    # cv2.line(frame, (400, 100), (400+vertical_movement.sum()/1000, 100 + horizontal_movement.sum() / 1000), (255, 0, 0), 5)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
