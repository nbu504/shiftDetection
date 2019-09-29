import cv2 as cv
import numpy as np
from scipy import interpolate

def imgOp(ori_img, left_top_l=1200, left_top_w=1000, right_bottle_l=2300, right_bottle_w=1700):
    pix = 0.094  #毫米
    # 图片切割的结果
    clip_img = np.copy(ori_img[left_top_w:right_bottle_w, left_top_l:right_bottle_l, :])

    gray_img = cv.cvtColor(clip_img, cv.COLOR_BGR2GRAY)  #灰度化
    ret, binary = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    # 定义结构元素
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    opened = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    #cv.imshow('edge', opened)

    edges_img = cv.Canny(opened, 50, 150, apertureSize=3)  #边缘检测
    cv.imwrite('pic/edge.jpg', edges_img)
    lines = cv.HoughLines(edges_img, 1, np.pi/180, 100)  #houghline直线检测

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
        ls.append((x1, y1))
        ls.append((x2, y2))

    #计算中轴线的两个点，确定中轴线
    mid_x1 = int((ls[0][0] + ls[2][0]) / 2)
    mid_y1 = int((ls[0][1] + ls[2][1]) / 2)
    mid_x2 = int((ls[1][0] + ls[3][0]) / 2)
    mid_y2 = int((ls[1][1] + ls[3][1]) / 2)
    ls.append((mid_x1, mid_y1))
    ls.append((mid_x2, mid_y2))

    cv.line(ori_img, (mid_x1 + left_top_l, mid_y1 + left_top_w), (mid_x2 + left_top_l, mid_y2 + left_top_w), (0, 0, 255), 2)
    #中轴直线参数
    A = mid_y2 - mid_y1
    B = - (mid_x2 - mid_x1)
    C = mid_y1 * (mid_x2 - mid_x1) - mid_x1 * (mid_y2 - mid_y1)
#####################################
    bin_array = np.array(edges_img)
    cv.imshow('bin', bin_array)
    new_X, new_Y = np.where(bin_array == 255)
    ls_xy = []
    for index in range(new_X.shape[0]):
        x_hat = new_X[index]
        y_hat = new_Y[index]
        a11 = mid_x1 - mid_x2
        a12 = mid_y1 - mid_y2
        b1 = a11 * x_hat + a12 * y_hat
        # 延伸后点与原来点同侧 2.5 / 0.094 = 26.5
        dis = np.fabs(A * x_hat + B * y_hat + C) / np.sqrt(A * A + B * B)
        if A * x_hat + B * y_hat + C > 0:
            b2 = (26.5 + dis) * np.sqrt(A * A + B * B) - C
        else:
            b2 = - (26.5 + dis) * np.sqrt(A * A + B * B) - C
        a21 = A
        a22 = B
        # 克拉默法则求解
        div = a11 * a22 - a12 * a21
        x_new = int((b1 * a22 - a12 * b2) / div)
        y_new = int((a11 * b2 - b1 * a21) / div)

        ls_xy.append([x_new, y_new])
        ori_img[x_new + left_top_w][y_new + left_top_l] = (0, 0, 255)

    cv.imwrite('pic/test.jpg', ori_img)
    return ls_xy

image = cv.imread('pic/original.jpg')
clip_0_ll, clip_0_lw, clip_0_rl, clip_0_rw = 1200, 1000, 2300, 1700
clip_1_ll, clip_1_lw, clip_1_rl, clip_1_rw = 2300, 1000, 3000, 1700

l1 = imgOp(image, left_top_l=clip_0_ll, left_top_w=clip_0_lw, right_bottle_l=clip_0_rl, right_bottle_w=clip_0_rw)
for i in range(len(l1)):
    cv.drawContours(image, l1[i], -1, (0, 0, 255), 7, offset=(clip_0_ll, clip_0_lw))

l2 = imgOp(image, left_top_l=clip_1_ll, left_top_w=clip_1_lw, right_bottle_l=clip_1_rl, right_bottle_w=clip_1_rw)
for i in range(len(l2)):
    cv.drawContours(image, l2[i], -1, (0, 0, 255), 7, offset=(clip_1_ll, clip_1_lw))

#cv.imwrite('pic/test.jpg', image)
cv.waitKey()
cv.destroyAllWindows()
