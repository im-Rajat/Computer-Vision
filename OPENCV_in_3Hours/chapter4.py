"""
Shapes and Texts
"""

from cv2 import cv2
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)
# print(img.shape)
# img[0:40, 40:70] = 255, 0, 0

cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 255), 3)
cv2.rectangle(img, (0, 0), (250, 350), (0, 255, 255), 2)
cv2.circle(img, (256, 256), 100, (255, 255, 0), 5)
cv2.putText(img, "OPEN CV", (200, 256), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 2)

cv2.imshow("Image", img)

cv2.waitKey(0)