import shapeRecognition
import numpy as np
import cv2


img = cv2.imread("prueba_2.jpg")

p1 = np.array([293, 759])
p2 = np.array([1306,  799])
p3 = np.array([209, 299])
p4 = np.array([1398, 300])


mask = np.zeros(img.shape[0:2], dtype=np.uint8)

points = np.array([p1, p2, p3, p4])

cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

res = cv2.bitwise_and(img, img, mask=mask)
rect = cv2.boundingRect(points)  # returns (x,y,w,h) of the rect
cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

cv2.imshow("Cropped", cropped)
