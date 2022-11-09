import cv2
import numpy as np

img = np.full((800, 800, 3), 255, np.uint8)
# img = cv2.imread("data/webcam_1.jpg")
# img = cv2.resize(img, dsize=(800, 800), interpolation=cv2.INTER_AREA)

thickness = 2

cv2.rectangle(img, (100, 200), (700, 600), (255, 0, 0), thickness=thickness)
cv2.line(img, (100, 200), (700, 600), (255, 0, 0), thickness=thickness)
cv2.line(img, (700, 200), (100, 600), (255, 0, 0), thickness=thickness)

img2 = np.zeros_like(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(th)
print(cnt)
print(stats)
print(centroids)

for i in range(cnt):
    img2[labels==i] = [int(j) for j in np.random.randint(0, 255, 3)]


cv2.imshow("img", img)
cv2.imshow("img2", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
cv2.waitKey()
cv2.destroyAllWindows()