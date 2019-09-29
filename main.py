# coding:utf8
from process.parmWriter import data_producer
from process.contourCompare import compare

IMG1_NAME = 'original.jpg' # 标准图片
IMG2_NAME = 'DSC03749.jpg' # 待检测图片

THRESH_HOLD = 400 # 超过多少个跨界像素判定为不合格
clip_0_ll, clip_0_lw, clip_0_rl, clip_0_rw = 1200, 1000, 2300, 1700

# 分别生成数据
clip_img1 = data_producer(IMG1_NAME, clip_0_ll, clip_0_lw, clip_0_rl, clip_0_rw)
clip_img2 = data_producer(IMG2_NAME, clip_0_ll, clip_0_lw, clip_0_rl, clip_0_rw)

# 判断是否越界
compare(IMG1_NAME, IMG2_NAME, THRESH_HOLD)

