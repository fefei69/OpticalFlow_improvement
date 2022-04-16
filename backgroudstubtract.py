import cv2
import matplotlib.pyplot as plt 
import numpy as np
subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
subtractor2 = cv2.createBackgroundSubtractorKNN(detectShadows=False)
# frame1 = cv2.imread('basketball1_360x270.bmp')
# frame2 = cv2.imread('basketball2_360x270.bmp')
frame1 = cv2.imread('dumptruck1_360x270.bmp')
frame2 = cv2.imread('dumptruck2_360x270.bmp')
#for erosion and dilation
kernel = np.ones((3,3), np.uint8)
step = 9
#edge detection first??
def TVL1(previous,current):
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(previous, current, None)
    return flow
def binary(image):
    ret, th1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    return th1
def erosion(image):
    erosion = cv2.erode(image, kernel, iterations = 2)
    return erosion
def dilation(image,iterations):
    dilation = cv2.dilate(image, kernel, iterations)
    return dilation
def laplacian_edgedetector(img):
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    return laplacian
def canny(gauss):
    canny = cv2.Canny(gauss, 30, 150)
    return canny
def greyandblur(img):
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(grey, (5,5), 0)
    return gauss
frame1_gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
frame2_gray = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
fr1 = greyandblur(frame1)
fr2 = greyandblur(frame2)
frame1_canny = canny(fr1)
frame2_canny = canny(fr2)
_ = subtractor.apply(fr1)
sb = subtractor.apply(fr2)
_ = subtractor2.apply(fr1)
knn = subtractor2.apply(fr2)
sb_binary = binary(sb)
# sb_erode = erosion(sb)
sb_dil = dilation(sb,5)

#subt = fr2 - fr1
# _ = subtractor2(fr1)
# next = subtractor2(fr2)
output1 = cv2.bitwise_and(sb_dil, fr1)
output2 = cv2.bitwise_and(sb_dil, fr2)
output1_bitwise_canny = cv2.bitwise_and(sb_dil, frame1_canny)
output2_bitwise_canny = cv2.bitwise_and(sb_dil, frame2_canny)
output1_canny = canny(output1)
output2_canny = canny(output2)
# output1_canny = erosion(output1_canny)
# output2_canny = erosion(output2_canny)
output1_laplacian = laplacian_edgedetector(output1)
output2_laplacian = laplacian_edgedetector(output2)
flow = cv2.calcOpticalFlowFarneback(output1_canny, output2_canny, None, 0.5, 3, 15, 3, 5, 1.2, 0)
flow_TVL1_ori = TVL1(frame1_gray,frame2_gray)
flow_ori = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
flow_farne = cv2.calcOpticalFlowFarneback(fr1, fr2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
flow_bitwise_canny = cv2.calcOpticalFlowFarneback(output1_bitwise_canny, output2_bitwise_canny, None, 0.5, 3, 20, 3, 5, 1.2, 0)
res1 = cv2.addWeighted(fr1,0.5,sb,0.5,20)
res2 = cv2.addWeighted(fr2,0.5,sb,0.5,20)
why_dilation = cv2.addWeighted(frame1_gray,0.5,sb,0.5,20)
why_dilation1 = cv2.addWeighted(frame1_gray,0.5,sb_dil,0.5,20)
plt.imshow(frame1)
# plt.quiver(np.arange(0, flow_TVL1_ori.shape[1], step), np.arange( 0,flow_TVL1_ori.shape[0], step), 
#                flow_TVL1_ori[::step, ::step, 0], flow_TVL1_ori[::step, ::step, 1],color = "red")
# plt.quiver(np.arange(0, flow_ori.shape[1], step), np.arange( 0,flow_ori.shape[0], step), 
#                flow_ori[::step, ::step, 0], flow_ori[::step, ::step, 1],color = "red")
# plt.quiver(np.arange(0, flow.shape[1], step), np.arange( 0,flow.shape[0], step), 
#                flow[::step, ::step, 0], flow[::step, ::step, 1],color = "red")
plt.quiver(np.arange(0, flow_farne.shape[1], step), np.arange( 0,flow_farne.shape[0], step), 
               flow_farne[::step, ::step, 0], flow_farne[::step, ::step, 1],color = "red")
# plt.quiver(np.arange(0, flow_bitwise_canny.shape[1], step), np.arange( 0,flow_bitwise_canny.shape[0], step), 
#                 flow_bitwise_canny[::step, ::step, 0], flow_bitwise_canny[::step, ::step, 1],color = "red")
               
plt.show()
# cv2.imshow("knn",knn)
# cv2.imshow("fr1",res1)
# cv2.imshow("canny",frame1_canny)
cv2.imshow("why dialtion",why_dilation)
cv2.imshow("why dialtion1",why_dilation1) 
cv2.imshow("background subtraction ",sb)
cv2.imshow("background subtraction and dilation",sb_dil)
# cv2.imshow("fr2",output1)
# cv2.imshow("fr3",output2)
cv2.imshow("fr4",frame1_canny)
# cv2.imshow("fr5",output2_canny)
cv2.imshow("bitwise canny1",output1_bitwise_canny)
# cv2.imshow("bitwise canny2",output2_bitwise_canny)
cv2.waitKey(0)
