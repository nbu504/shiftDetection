# coding:utf8
import cv2 as cv
import numpy as np
import json

# 获取原图数据
with open('data.json', 'r') as f:
    data = json.load(f)
    oriExpandContours = np.array(data['expandContours'])
    oriContours = np.array(data['contours'])
    oriMidlinePoints = [tuple(x) for x in data['midlinePoint']]

new_img = cv.imread('pic/DSC03749.jpg')
#print(new_img.shape)
pix = 0.094  #毫米
left_top_l = 1200  #切割的左上角长
left_top_w = 1000  #切割的左上角宽
right_bottle_l = 2300  #右下角长
right_bottle_w = 1700  #右下角宽

clip_img = new_img[left_top_w:right_bottle_w, left_top_l:right_bottle_l, :]  #图片切割的结果

#cv.imwrite('pic/clip.jpg', clip_img)

#cv.imshow('clip_img', clip_img)

gray_img = cv.cvtColor(clip_img, cv.COLOR_BGR2GRAY)  #灰度化
edges_img = cv.Canny(gray_img, 50, 150, apertureSize=3)  #边缘检测
ret, binary = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
#cv.imshow('gray_img', gray_img)
#cv.imshow('edges', edges_img)
#cv.imshow('thresh', binary)

lines = cv.HoughLines(edges_img, 1, np.pi/180, 150)  #houghline直线检测

#print(lines.shape)

ls = []  #保存直线上的点

for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    #cv.line(clip_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    ls.append((x1, y1))
    ls.append((x2, y2))

#计算中轴线的两个点，确定中轴线
mid_x1 = int((ls[0][0] + ls[2][0]) / 2)
mid_y1 = int((ls[0][1] + ls[2][1]) / 2)
mid_x2 = int((ls[1][0] + ls[3][0]) / 2)
mid_y2 = int((ls[1][1] + ls[3][1]) / 2)
ls.append((mid_x1, mid_y1))
ls.append((mid_x2, mid_y2))

cv.line(clip_img, (mid_x1, mid_y1), (mid_x2, mid_y2), (255, 0, 0), 2)
#中轴直线参数
A = mid_y2 - mid_y1
B = - (mid_x2 - mid_x1)
C = (mid_y1 - mid_x1) * (mid_x2 - mid_x1)

contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# _, contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for contour in contours:
    if cv.contourArea(contour) > 1000.0:
        cv.drawContours(clip_img, contour, -1, (255, 0, 0), 1)
        contour_new = np.copy(contour)

#cv.imwrite('pic/maocao.jpg', clip_img)

cv.line(clip_img, oriMidlinePoints[0], oriMidlinePoints[1], (0, 0, 255), 2)

cv.drawContours(clip_img, oriContours, -1, (0, 0, 255), 1)

cv.drawContours(clip_img, oriExpandContours, -1, (0, 0, 255), 1)

cv.imshow("image line", clip_img)

cv.waitKey()
cv.destroyAllWindows()