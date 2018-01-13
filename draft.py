import numpy as np
import cv2
import sys



def get_index_corner(hist, stopit):
    # cumulated = [sum(hist[:i]) for i in range(len(hist))]
    cumulated = np.cumsum(hist)
    for i in range(len(cumulated)):
        if stopit < cumulated[i]:
            return i


def get_POI(hist, quantile):
    hhist = get_hhist(hist)
    vhist = get_vhist(hist)
    hqd = quantile * sum(hhist)
    hqu = (1 - quantile) * sum(hhist)
    vqd = quantile * sum(vhist)
    vqu = (1 - quantile) * sum(vhist)

    lindex = get_index_corner(hhist, vqd)
    rindex = get_index_corner(hhist, vqu)
    dindex = get_index_corner(vhist, hqd)
    uindex = get_index_corner(vhist, hqu)
    # return (100, 300, 100, 300)
    return (lindex, rindex, dindex, uindex)


def get_hhist(img):
    return np.count_nonzero(fgmask, axis=0)


def get_vhist(img):
    return np.count_nonzero(fgmask, axis=1)

cap = cv2.VideoCapture(sys.argv[1])

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)


    if np.count_nonzero(fgmask) > 100:
        fgmask_grey = fgmask
        lindex, rindex, dindex, uindex = get_POI(fgmask_grey, 0.02)
        if lindex and rindex and dindex and uindex:
            cv2.rectangle(frame, (lindex, dindex), (rindex, uindex), (0, 255, 0), 3)

    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
