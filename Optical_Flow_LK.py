import numpy as np
import cv2
import matplotlib.pyplot as plt 

# frame1 = cv2.imread('dumptruck1_360x270.bmp')
# frame2 = cv2.imread('dumptruck2_360x270.bmp')
frame1 = cv2.imread('basketball1_360x270.bmp')
frame2 = cv2.imread('basketball2_360x270.bmp')
prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
img3 = cv2.resize(frame1, (180, 270//2), interpolation=cv2.INTER_AREA)
img4 = cv2.resize(frame1, (90, 270//4), interpolation=cv2.INTER_AREA)
step = 9
# params for corner detection
feature_params = dict( maxCorners = 100,
					qualityLevel = 0.3,
					minDistance = 7,
					blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize = (15, 15),
				maxLevel = 2,
				criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
							10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it

p0 = cv2.goodFeaturesToTrack(prev, mask = None,
							**feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(frame1)




# calculate optical flow
p1, st, err = cv2.calcOpticalFlowPyrLK(prev,
									next,
									p0, None,
									**lk_params)

# Select good points
good_new = p1[st == 1]
good_old = p0[st == 1]
#good_new = p1
#good_old = p0

# print(p1)
# print(st)
# print(err)
# draw the tracks
# plt.quiver(np.arange(0, p1.shape[1], step), np.arange( 0,p1.shape[0], step), 
#                p1[::step, ::step, 0], p1[::step, ::step, 1],color = "blue")
# plt.show()
for i, (new, old) in enumerate(zip(good_new,
								good_old)):
	a, b = new.ravel()
	c, d = old.ravel()
	a = int(a)*2
	b = int(b)*2
	c = int(c)*2
	d = int(d)*2
	# print(a,b)
	# mask = cv2.line(mask, (a, b), (c, d),
	# 				color[i].tolist(), 2)

	mask = cv2.arrowedLine(mask,(c,d),(a,b),(255,255,0),1,tipLength=0.5)
	
	frame2 = cv2.circle(frame2, (a, b), 5,
					color[i].tolist(), -1)
#print(frame2.shape,mask.shape)	
#img1 = cv2.add(frame2, mask)
img2 = cv2.add(frame1, mask)

# cv2.imshow('select good points (corners)', frame2)
# cv2.imshow('downscale image 3', img3)
# cv2.imshow('downscale image 4', img4)
# cv2.imshow('frame1 + mask', img2)
# cv2.imshow('mask', mask)
cv2.waitKey(0)


# Updating Previous frame and points
# old_gray = frame_gray.copy()
# p0 = good_new.reshape(-1, 1, 2)


