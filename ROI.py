import numpy as np


class ROI():
    def get_index_corner(hist, stopit):
        cumulated = np.cumsum(hist)
        for i in range(len(cumulated)):
            if stopit < cumulated[i]:
                return i

    def get_POI_corners(hist, quantile):
        hhist = ROI.get_hhist(hist)
        vhist = ROI.get_vhist(hist)
        hqd = quantile * sum(hhist)
        hqu = (1 - quantile) * sum(hhist)
        vqd = quantile * sum(vhist)
        vqu = (1 - quantile) * sum(vhist)

        lindex = ROI.get_index_corner(hhist, vqd)
        rindex = ROI.get_index_corner(hhist, vqu)
        dindex = ROI.get_index_corner(vhist, hqd)
        uindex = ROI.get_index_corner(vhist, hqu)
        return (lindex, rindex, dindex, uindex)

    def get_hhist(img):
        return np.count_nonzero(img, axis=0)

    def get_vhist(img):
        return np.count_nonzero(img, axis=1)
