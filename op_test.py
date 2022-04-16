import cv2
import numpy as np

mask = np.zeros((100, 100, 3), np.uint8)

p0 = [np.array([50, 50])]
p1 = [np.array([60, 60])]

dt = 100

for i, (old,new) in enumerate(zip(p0,p1)):
    a, b = np.ravel(old)
    c, d = np.ravel(new)
    #displacement
    v_x = int((c - a)/dt)
    v_y = int((b - d)/dt)
    scale = 100
    v_x = int((c-a)/dt*scale)
    v_y = int((b-d)/dt*scale)

    mask = cv2.arrowedLine(mask, (int(a),int(b)), (int(a)+v_x, int(b)+v_y), (255, 255, 255), 1)
    image = cv2.rotate(mask, cv2.ROTATE_180)

cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()