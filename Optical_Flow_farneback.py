from typing import final
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import scipy
import os 
x_pos = []
y_pos = []
subtracao1 = cv2.createBackgroundSubtractorMOG2() #背景分離 
subtracao2 = cv2.createBackgroundSubtractorKNN(detectShadows=False)
filepath = "C:/vscode/machinevision_project/basketball"
step = 9
def background_subtractor(filepath):
    for file in os.listdir(filepath):
        stillFrame = cv2.imread(os.path.join(filepath,file))
        grey = cv2.cvtColor(stillFrame,cv2.COLOR_BGR2GRAY)
        back = subtracao2.apply(grey)
        cv2.imshow("Cut Image", back)
        cv2.waitKey(0)
        #cv2.destroyAllWindows()

def TVL1(previous,current):
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(previous, current, None)
    #cv2.DualTVL1OpticalFlow_create()
    disp_x = flow[..., 0]
    disp_y = flow[..., 1]
    return flow


def draw(img,disp_x,disp_y):
    #seperate into 40x30 region each region : 9x9
    #製作畫布
    mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    #畫上箭頭
    x=0
    y=0
    for i in range(30): #270/9
        for j in range(40):  #360/9
            cv2.circle(mask,(9*j+5,9*i+5),1,(255,0,0),-1)
            new_x = round(9*j+5+disp_x[9*i+5,9*j+5])
            new_y = round(9*i+5+disp_y[9*i+5,9*j+5])
            cv2.arrowedLine(mask,(9*j+5,9*i+5),(new_x,new_y),(255,255,0),1,tipLength=0.5)
            x_pos.append(new_x)
            y_pos.append(new_y)
            #cv2.line(mask,(9*j+5,9*i+5),(new_x,new_y),(255,255,0),1)
    return mask ,x_pos , y_pos
frame1 = cv2.imread('basketball1_360x270.bmp')
frame2 = cv2.imread('basketball2_360x270.bmp')
# frame1 = cv2.imread('dumptruck1_360x270.bmp')
# frame2 = cv2.imread('dumptruck2_360x270.bmp')
#background_subtractor(filepath)
median_blur1 = cv2.medianBlur(frame1, 5)
median_blur2 = cv2.medianBlur(frame2, 5)
gauss_1 = cv2.GaussianBlur(frame1, (5,5), 0)
gauss_2 = cv2.GaussianBlur(frame2,(5,5),0)
grey1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
grey2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
gauss_4subt1 = cv2.GaussianBlur(grey1, (3,3), 5)
gauss_4subt2 = cv2.GaussianBlur(grey2, (3,3), 5)
#blur = cv2.GaussianBlur(grey,(3,3),5)
#print(gauss_4subt.shape)
_ = subtracao1.apply(gauss_4subt1)
img_su1 = subtracao1.apply(gauss_4subt2)
output1 = cv2.bitwise_and(gauss_4subt1, img_su1)
output2 = cv2.bitwise_and(gauss_4subt2, img_su1)
flow_subt = cv2.calcOpticalFlowFarneback(output1, output2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
disp_x_subt = flow_subt[..., 0]
disp_y_subt = flow_subt[..., 1]
finalRes , a , b = draw(frame1,disp_x_subt,disp_y_subt)
#img_su2 = subtracao.apply(gauss_4subt2)
#print(median_blur1.shape)
#frame1 = median_blur1
# frame2 = median_blur2
frame1 = gauss_1
frame2 = gauss_2

#fl = TVL1(frame1,frame2)
hsv = np.zeros_like(frame1)
hsv1 = np.zeros_like(frame1)
hsv1[..., 1] = 255 #飽和度
#print(hsv.shape)
hsv[..., 1] = 255 #飽和度
prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
fl_tvl1 = TVL1(prev,next)
fl_x = fl_tvl1[..., 0]
fl_y = fl_tvl1[..., 1]
mag, ang = cv2.cartToPolar(fl_x, fl_y) #(x,y)
hsv1[..., 0] = ang*180/np.pi/2 #色相 運動方向
hsv1[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) #亮度值 運動快慢
flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
disp_x = flow[..., 0]
disp_y = flow[..., 1]

# disp_x = cv2.normalize(disp_x, None, 0, 10, cv2.NORM_MINMAX)
# disp_y = cv2.normalize(disp_y, None, 0, 10, cv2.NORM_MINMAX)
#print(np.min(disp_x))
#print(disp_x)
#print(flow.shape)
#print(flow[...,0].shape) 
#print(mask.shape)
#p0 = cv.goodFeaturesToTrack(prev, mask = None, **feature_params)
#p1, st, err = cv2.calcOpticalFlowPyrLK(prev, next, p0, None, **lk_params)
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1]) #(x,y)
#print(ang)
hsv[..., 0] = ang*180/np.pi/2 #色相 運動方向
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) #亮度值 運動快慢
#print(hsv[..., 2].shape)
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
bgr1 = cv2.cvtColor(hsv1, cv2.COLOR_HSV2BGR)
ts ,x,y= draw(frame1,disp_x,disp_y)
result = cv2.addWeighted(frame1,0.5,ts,0.5,20)

tvl1 , x_tv , y_tv = draw(frame1,fl_x,fl_y)
tvl1_result = cv2.addWeighted(frame1,0.5,tvl1,0.5,20)
Farneback_subt = cv2.addWeighted(frame1,0.5,finalRes,0.5,20)
#plt.imshow(ts)
#plt.show()
#cv2.imshow('optical flow arrow version', ts)
# cv2.imshow('subtract forground1', img_su1)
# cv2.imshow('subtract forground2', img_su2)
# cv2.imshow("tvl1 optical flow ",tvl1_result)
# cv2.imshow("optical flow hsv version",bgr)
# cv2.imshow("tvl1 optical flow hsv version",bgr1)
# cv2.imshow("optical flow arrow vetor version",result)
#f, axarr = plt.subplots(2,2)
# plt.quiver(np.arange(0, flow.shape[1], step), np.arange( 0,flow.shape[0], step), 
#                flow[::step, ::step, 0], flow[::step, ::step, 1],color = "blue")

# plt.quiver(np.arange(0, fl_tvl1.shape[1], step), np.arange( 0,fl_tvl1.shape[0], step), 
#                flow[::step, ::step, 0], flow[::step, ::step, 1],color = "blue")

plt.quiver(np.arange(0, flow_subt.shape[1], step), np.arange( 0,flow_subt.shape[0], step), 
               flow[::step, ::step, 0], flow[::step, ::step, 1],color = "blue")
               
plt.show()
cv2.imshow("optical flow arrow vetor version",finalRes)
cv2.waitKey(0)
#hsv[..., 0] = ang*180/np.pi/2
#hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#print(flow)
#cv2.imshow('optical flow', flow)
# print(type(x))
# print(type(disp_y))
# plt.quiver(x, y ,disp_x,disp_y)
# plt.show()