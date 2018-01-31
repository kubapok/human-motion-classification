import numpy as np
import cv2
import sys
from ROI import ROI
from fuzzy_classifier import FuzzyClassifier
from statistics import mean

cap = cv2.VideoCapture(sys.argv[1])

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

classifier = FuzzyClassifier()
# classifier.plot_variables()
from tree_classifier import tree_classifier
tree_classifier.train()

i = 0
ROIframe = np.zeros((480, 640), dtype=np.uint8)
mask = np.zeros((480, 640), dtype=np.uint8)
ret, frame = cap.read()
prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
lindex, rindex, uindex, dindex, = 0, 10, 0, 10
i = 0

ratio_history = [0 for _ in range(10)]
v_motion_history = [0 for _ in range(10)]
h_motion_history = [0 for _ in range(10)]

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)  # w skali szarości 0-255 ale wiekszosc skrajne
    if np.count_nonzero(fgmask) > 100:
        lindex, rindex, dindex, uindex = ROI.get_POI_corners(fgmask, 0.02)
        if lindex and rindex and dindex and uindex:
            mask = np.zeros((480, 640), dtype=np.uint8)
            mask[dindex: uindex, lindex:rindex] = 255
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ROIframe_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)
            flow = cv2.calcOpticalFlowFarneback(
                prvs, ROIframe_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            ROIonly_flow_gray = flow[dindex: uindex, lindex:rindex]
            vertical_movement, horizontal_movement = cv2.split(
                ROIonly_flow_gray)

            #***
            height = uindex - dindex
            width = rindex - lindex
            v = height * width


            ratio = height/width
            v_motion = 10*vertical_movement.sum()/v
            h_motion = 10*horizontal_movement.sum()/v

            ratio_history.append(ratio)
            ratio_history = ratio_history[1:]
            v_motion_history.append(v_motion)
            v_motion_history = v_motion_history[1:]
            h_motion_history.append(h_motion)
            h_motion_history = h_motion_history[1:]

            data = (vertical_movement.sum(), horizontal_movement.sum(), height, width)
            motion = classifier.classify(data)
            predicted = tree_classifier.predict([mean(v_motion_history),mean(h_motion_history), mean(ratio_history)])
            print(tree_classifier.classes_dict[predicted],'\t',float("{0:.2f}".format(mean(v_motion_history))), '\t', float("{0:.2f}".format(mean(h_motion_history))) , '\t', float("{0:.2f}".format(mean(ratio_history))), '\t')
            #***

            prvs = ROIframe_gray
            i += 1
            cv2.line(frame, (400, 100), (400+int(vertical_movement.sum()/1000), 100 + int(horizontal_movement.sum() / 1000)), (255, 0, 0), 5)
            cv2.line(frame, (400, 100), (400,100), (0, 0, 255), 5)
            if i == 60:
                pass
    cv2.rectangle(frame, (lindex, dindex),
                  (rindex, uindex), (0, 255, 0), 3)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
