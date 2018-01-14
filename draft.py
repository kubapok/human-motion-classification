import numpy as np
import cv2
import sys
from ROI import ROI


cap = cv2.VideoCapture(sys.argv[1])

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

i = 0
ROIframe = np.zeros((480, 640), dtype=np.uint8)
mask = np.zeros((480, 640), dtype=np.uint8)
ret, frame = cap.read()
prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)  # w skali szarości 0-255 ale wiekszosc skrajne
    if np.count_nonzero(fgmask) > 100:
        lindex, rindex, dindex, uindex = ROI.get_POI_corners(fgmask, 0.02)
        if lindex and rindex and dindex and uindex:
            mask = np.zeros((480, 640), dtype=np.uint8)
            mask[dindex: uindex, lindex:rindex] = 255
            ROIframe = cv2.bitwise_and(frame, frame, mask=mask)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs,frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            from IPython import embed; embed()
            prvs = frame_gray
            cv2.rectangle(frame, (lindex, dindex),
                          (rindex, uindex), (0, 255, 0), 3)

    # i += 1
    # if i == 50:
    #     from IPython import embed; embed()
    cv2.imshow('frame', mask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
