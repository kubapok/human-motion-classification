import numpy as np
import cv2


def get_index_corner(hist, stopit):
    cumulated = [sum(hist[:i]) for i in range(len(hist))]
    for i in range(len(cumulated)):
        if stopit < cumulated[i]:
            return i


fgmask = cv2.imread('fgmask.jpg', cv2.IMREAD_GRAYSCALE)
hhist = np.count_nonzero(fgmask, axis=0)
vhist = np.count_nonzero(fgmask, axis=1)


hq5 = 0.03 * sum(hhist)
hq95 = 0.97 * sum(hhist)
vq5 = 0.03 * sum(vhist)
vq95 = 0.93 * sum(vhist)


lindex = get_index_corner(hhist, vq5)
rindex = get_index_corner(hhist, vq95)
dindex = get_index_corner(vhist, hq5)
uindex = get_index_corner(vhist, hq95)


img = cv2.imread('fgmask.jpg')
# :cv2.rectangle(fgmask, (lindex, dindex ), (rindex, uindex ), (0, 255, 0), 3)
cv2.rectangle(img, (lindex, dindex ), (rindex, uindex ), (0, 255, 0), 3)
cv2.imwrite('a.jpg',img)
