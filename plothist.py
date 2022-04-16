import cv2
import numpy as np
import matplotlib.pyplot as plt
subtractor = cv2.createBackgroundSubtractorMOG2()
def greyandblur(img):
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(grey, (5,5), 0)
    return gauss
# 讀取圖檔
# frame1 = cv2.imread('basketball1_360x270.bmp')
# frame2 = cv2.imread('basketball2_360x270.bmp')
frame1 = cv2.imread('dumptruck1_360x270.bmp')
frame2 = cv2.imread('dumptruck2_360x270.bmp')
#img = cv2.imread('image.jpg')
# fr1 = greyandblur(frame1)
# fr2 = greyandblur(frame2)
# 轉為灰階圖片
gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
_ = subtractor.apply(gray)
sb = subtractor.apply(gray1)
#sb = 255-sb
# 計算直方圖每個 bin 的數值
hist = cv2.calcHist([gray], [0], None, [255], [0, 255])
hist1 = cv2.calcHist([gray1], [0], None, [255], [0, 255])
hist2 = cv2.calcHist([sb], [0], None, [255], [0, 255])

# 畫出直方圖
# plt.hist(gray.ravel(), 256, [0, 256])
# plt.hist(gray1.ravel(), 256, [0, 256])
plt.plot(hist)
plt.plot(hist1)
plt.legend()
#plt.plot(hist2)
#plt.imshow(sb)
plt.show()