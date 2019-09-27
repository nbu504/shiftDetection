import cv2 as cv
import numpy as np

ori_img = cv.imread('pic/original.jpg')
left_top_l = 1200
left_top_w = 1000
right_bottle_l = 2300
right_bottle_w = 1700

clip_img = ori_img[left_top_w:right_bottle_w, left_top_l:right_bottle_l, :]

#cv.imwrite('pic/clip.jpg', clip_img)

#cv.imshow('clip_img', clip_img)

gray_img = cv.cvtColor(clip_img, cv.COLOR_BGR2GRAY)
edges_img = cv.Canny(gray_img, 50, 150, apertureSize=3)
#cv.imshow('gray_img', gray_img)
cv.imshow('edges', edges_img)

lines = cv.HoughLines(edges_img, 1, np.pi/180, 150)

print(lines.shape)
midline_angle = 0
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(clip_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv.imshow("image line", clip_img)



cv.waitKey()
cv.destroyAllWindows()