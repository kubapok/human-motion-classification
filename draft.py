import numpy as np
import cv2
import sys
import numy as np

cap = cv2.VideoCapture(sys.argv[1])

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

i = 0 
while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame',fgmask)
    i+=1
    if i == 30*4:
        from IPython import embed; embed()
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
