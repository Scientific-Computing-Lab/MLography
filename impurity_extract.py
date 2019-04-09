import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./tags_png/scan1tag1.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV)

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)
opening = cv.morphologyEx(opening, cv.MORPH_ERODE, kernel, iterations=2)
# sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)
# Finding sure foreground area

# dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
# ret, sure_fg = cv.threshold(dist_transform, 0.2*dist_transform.max(), 255, 0)

sure_fg = opening


# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)
cv.imshow('sure_fg', sure_fg)
cv.waitKey(0)
cv.destroyAllWindows()


# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv.watershed(img, markers)
img[markers == -1] = [0, 0, 255]

cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
