# coding:utf8
import numpy as np
import json
import os
import cv2 as cv

def areaCal(img):
    # 白像素面积计算
    area = 0
    height, width, _ = img.shape
    for i in range(height):
        for j in range(width):
            if img[i, j] == 255:
                area += 1
    return area


def rotate_bound(image, angle):
    # 获取图像的尺寸
    # 旋转中心
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)

    # 设置旋转矩阵
    M = cv.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像旋转后的新边界
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    return M, cv.warpAffine(image, M, (nW, nH))

def writeData(filename, data):
    if not os.path.exists('data/'):
        os.mkdir('data/')

    with open(filename, 'w') as fw:
        json.dump(data, fw)

def getData(filename):
    # 获取图片数据: 扩展外边缘， 边缘， 中轴线点
    with open(filename, 'r') as f:
        data = json.load(f)
        shape = data['shape']
        expandContours = np.array(data['expandContours'])
        contours = np.array(data['contours'])
        midlinePoints = [tuple(x) for x in data['midlinePoints']]

    return {"shape": shape,
            "expandContours": expandContours,
            "contours": contours,
            "midlinePoints": midlinePoints,
            }


