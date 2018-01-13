import numpy as np
import cv2


def get_index_corner(hist, stopit):
    cumulated = [sum(hist[:i]) for i in range(len(hist))]
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

    return (lindex, rindex, dindex, uindex)


def get_hhist(img):
    return np.count_nonzero(fgmask, axis=0)


def get_vhist(img):
    return np.count_nonzero(fgmask, axis=1)


# fgmask = cv2.imread('fgmask.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('fgmask.jpg')
fgmask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
lindex, rindex, dindex, uindex = get_POI(fgmask, 0.02)
cv2.rectangle(img, (lindex, dindex), (rindex, uindex), (0, 255, 0), 3)
cv2.imwrite('a.jpg', img)
