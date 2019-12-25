from util.util import getData
from process.parmWriter import img_slicer
from process.newParmWriter import data_producer
import cv2 as cv



img = cv.imread('../pic/original.jpg')

clip_0_ll, clip_0_lw, clip_0_rl, clip_0_rw = 1200, 1000, 2300, 1700
clip_img = img_slicer(img, clip_0_ll, clip_0_lw, clip_0_rl, clip_0_rw)

data_producer('original.jpg', clip_0_ll, clip_0_lw, clip_0_rl, clip_0_rw)

data = getData('../data/original.jpg1.json')

midline = data['midlinePoints']

expand = data['expandContours']

contours = data['contours']

cv.line(clip_img, midline[0], midline[1], (0, 0, 255), 2)


cv.drawContours(clip_img, expand, -1, (0, 0, 255), 1)

cv.fillPoly(clip_img, pts=[expand], color=(255, 255, 255))

cv.imshow('expand', clip_img)

cv.waitKey()
cv.destroyAllWindows()