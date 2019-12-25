# coding: utf8
import cv2 as cv
import numpy as np
from util.util import writeData


def img_slicer(img, left_top_l, left_top_w, right_bottle_l, right_bottle_w):
    # 图片切割
    return np.copy(img[left_top_w:right_bottle_w, left_top_l:right_bottle_l, :])


def data_producer(img_name, left_top_l, left_top_w, right_bottle_l, right_bottle_w):
    img = cv.imread('../pic/' + img_name)
    clip_img = img_slicer(img, left_top_l, left_top_w, right_bottle_l, right_bottle_w)

    gray_img = cv.cvtColor(clip_img, cv.COLOR_BGR2GRAY)  # 灰度化
    ret, binary = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)

    # 定义结构元素
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    opened = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)

    edges_img = cv.Canny(opened, 50, 150, apertureSize=3)  # 边缘检测

    lines = cv.HoughLines(edges_img, 1, np.pi / 180, 100)  # houghline直线检测

    # print(lines.shape)

    ls = []  # 保存直线上的点

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
        # cv.line(clip_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        ls.append((x1, y1))
        ls.append((x2, y2))

    # 计算中轴线的两个点，确定中轴线
    mid_x1 = int((ls[0][0] + ls[2][0]) / 2)
    mid_y1 = int((ls[0][1] + ls[2][1]) / 2)
    mid_x2 = int((ls[1][0] + ls[3][0]) / 2)
    mid_y2 = int((ls[1][1] + ls[3][1]) / 2)
    # ls.append((mid_x1, mid_y1))
    # ls.append((mid_x2, mid_y2))

    cv.line(clip_img, (mid_x1, mid_y1), (mid_x2, mid_y2), (0, 0, 255), 2)
    # 中轴直线参数
    A = mid_y2 - mid_y1
    B = - (mid_x2 - mid_x1)
    C = (mid_y1 - mid_x1) * (mid_x2 - mid_x1)

    # contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    bin_array = np.array(edges_img)
    cv.imshow('bin', bin_array)
    new_X, new_Y = np.where(bin_array == 255)

    contour_new = np.zeros((new_X.shape[0], 1, 2), dtype='int64')
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

        contour_new[index][0][0] = y_new
        contour_new[index][0][1] = x_new

    contour_new = np.array(contour_new)


    # cv.drawContours(clip_img, contour_new, -1, (0, 0, 255), 1)


    jsonResult = {'shape': clip_img.shape,
                  'contours': [ls],
                  'midlinePoints': [(mid_x1, mid_y1), (mid_x2, mid_y2)],
                  'expandContours': contour_new.tolist()}

    writeData('../data/' + img_name + '1.json', jsonResult)

    return clip_img
