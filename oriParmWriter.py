# coding:utf8
import cv2 as cv
import numpy as np
import json

ori_img = cv.imread('pic/original.jpg')
#print(ori_img.shape)
pix = 0.094  #毫米
left_top_l = 1200  #切割的左上角长
left_top_w = 1000  #切割的左上角宽
right_bottle_l = 2300  #右下角长
right_bottle_w = 1700  #右下角宽

clip_img = ori_img[left_top_w:right_bottle_w, left_top_l:right_bottle_l, :]  #图片切割的结果

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

cv.line(clip_img, (mid_x1, mid_y1), (mid_x2, mid_y2), (0, 0, 255), 2)
#中轴直线参数
A = mid_y2 - mid_y1
B = - (mid_x2 - mid_x1)
C = (mid_y1 - mid_x1) * (mid_x2 - mid_x1)

contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# _, contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for contour in contours:
    if cv.contourArea(contour) > 1000.0:
        cv.drawContours(clip_img, contour, -1, (0, 0, 255), 1)
        storeOriContour = contour
        contour_new = np.copy(contour)
        #contour_new = []
        k = 0  #时间戳
        for index in range(contour.shape[0]):
            x_hat = contour[index][0][0]
            y_hat = contour[index][0][1]
            # 延伸后 与中轴內积
            a11 = mid_x1 - mid_x2
            a12 = mid_y1 - mid_y2
            b1 = a11 * x_hat + a12 * y_hat
            #延伸后点与原点同侧 2.5 / 0.094 = 26.5
            b2 = 0
            dis = np.abs(A * x_hat + B * y_hat + C) / np.sqrt(A * A + B * B)
            if A * x_hat + B * y_hat + C > 0:
                b2 = (26.5 + dis) * np.sqrt(A * A + B * B) - C
            else:
                b2 = - (26.5 + dis) * np.sqrt(A * A + B * B) - C
            a21 = A
            a22 = B
            #克拉默法则求解
            div = a11 * a22 - a12 * a21
            x_new = (b1 * a22 - a12 * b2) / div
            y_new = (a11 * b2 - b1 * a21) / div
            # contour_new[index][0][0] = x_new
            # contour_new[index][0][1] = y_new
            x_new = int(x_new)
            y_new = int(y_new)
            #contour_new.append([[x_new, y_new]])
            contour_new[index][0][0] = x_new
            contour_new[index][0][1] = y_new
            #角落插值
            # if k > 0:
            #     max_dis = np.sqrt(np.power((x_new - contour_new[k - 1][0][0]), 2)
            #                       + np.power((y_new - contour_new[k - 1][0][1]), 2))
            #     if max_dis > 18:
            #         left = np.min(x_new, contour_new[k][0][0])
            #         right = np.max(x_new, contour_new[k][0][0])
            #         x_chazhi = np.linspace(left, right, 36)
            # k += 1
        contour_new = np.array(contour_new)
        print(type(contour_new), contour_new.shape)
        print(type(contour), contour.shape)
        cv.drawContours(clip_img, contour_new, -1, (0, 0, 255), 1)
        # contour_new = np.concatenate((contour_new, [[[1, 1]]]), axis=0)
        # print(contour_new.shape)
#cv.imwrite('pic/maocao.jpg', clip_img)

jsonResult = {'contours': storeOriContour.tolist(),
              'midlinePoint': [(mid_x1, mid_y1), (mid_x2, mid_y2)],
              'expandContours': contour_new.tolist()}

with open('data.json', 'w') as fw:

    json.dump(jsonResult, fw)
# cv.imshow("image line", clip_img)
#
# cv.waitKey()
# cv.destroyAllWindows()