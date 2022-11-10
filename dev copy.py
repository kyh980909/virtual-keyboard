import cv2
import numpy as np

img = np.full((960, 1280, 3), 255, np.uint8)

thickness = 1

cv2.rectangle(img, (100, 200), (700, 600), (0, 255, 0), thickness=thickness)
cv2.line(img, (100, 200), (700, 600), (255, 0, 0), thickness=thickness)
cv2.line(img, (700, 200), (100, 600), (255, 0, 0), thickness=thickness)

rows, cols = img.shape[:2]
mask = np.zeros((rows+2, cols+2), np.uint8)
new_val = (0, 0, 255)
loDiff, upDiff = (10, 10, 10), (10, 10, 10)

def onMouse(event, x, y, flags, param):
    global mask, img
    if event == cv2.EVENT_FLAG_LBUTTON:
        seed = (x,y)

        retval = cv2.floodFill(img, mask, seed, new_val, loDiff, upDiff)
        cv2.imshow('img', img)

cv2.imshow("img", img)
cv2.setMouseCallback('img', onMouse)
cv2.waitKey()
cv2.destroyAllWindows()