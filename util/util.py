# coding:utf8
import numpy as np
import json
import os

def areaCal(img):
    # 白像素面积计算
    area = 0
    height, width, _ = img.shape
    for i in range(height):
        for j in range(width):
            if img[i, j] == 255:
                area += 1
    return area

def writeData(filename, data):
    if not os.path.exists('data/'):
        os.mkdir('data/')

    with open(filename, 'w') as fw:
        json.dump(data, fw)

def getData(filename):
    # 获取图片数据: 扩展外边缘， 边缘， 中轴线点
    with open('data/' + filename +'.json', 'r') as f:
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


