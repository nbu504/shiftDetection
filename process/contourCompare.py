# coding: utf8
from __future__ import print_function
import cv2 as cv
import numpy as np
from util.util import areaCal, getData

def drawer(img, standardData, toCompareData):
    cv.line(img, standardData['midlinePoints'][0], standardData['midlinePoints'][1], (0, 0, 255), 2)

    cv.line(img, toCompareData['midlinePoints'][0], toCompareData['midlinePoints'][1], (255, 0, 0), 2)

    cv.drawContours(img, standardData['expandContours'], -1, (0, 0, 255), 1)

    cv.drawContours(img, toCompareData['contours'], -1, (255, 0, 0), 1)

    cv.imshow('contour', img)

    mask_img = np.zeros((toCompareData['shape'][0], toCompareData['shape'][1], 1), np.uint8)

    cv.fillPoly(mask_img, pts=[standardData['expandContours']], color=(255, 255, 255))
    cv.imshow('origin_expand', mask_img)

    cv.fillPoly(mask_img, pts=[toCompareData['contours']], color=(255, 255, 255))
    cv.imshow('new_contour', mask_img)

    cv.waitKey(0)
    cv.destroyAllWindows()

def compare(standardImgName, toCompareImgName, threshHold):
    standardData = getData('data/' + standardImgName + '.json')
    toCompareData = getData('data/' + toCompareImgName + '.json')

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

    return {'standardData': standardData,
            'toCompareData': toCompareData,
            }



