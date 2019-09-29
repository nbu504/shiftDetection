# coding: utf8
from __future__ import print_function
import cv2 as cv
import numpy as np
from util.util import areaCal, getData

def compare(standardImgName, toCompareImgName, threshHold):
    standardData = getData(standardImgName)
    toCompareData = getData(toCompareImgName)

    # 原图扩大边缘后mask
    mask_img = np.zeros((toCompareData['shape'][0], toCompareData['shape'][1], 1), np.uint8)
    cv.fillPoly(mask_img, pts=[standardData['expandContours']], color=(255, 255, 255))
    oriArea = areaCal(mask_img)

    cv.fillPoly(mask_img, pts=[toCompareData['contours']], color=(255, 255, 255))
    newArea = areaCal(mask_img)

    # 计算新图增加白色面积
    bulgeArea = newArea - oriArea
    print("oriArea: %d, newArea: %d, bulgeArea: %d, percent: %.5f%%" % (
        oriArea, newArea, bulgeArea, float(bulgeArea) / oriArea * 100))

    if bulgeArea >= threshHold:
        print("不合格")
    else:
        print("合格")



