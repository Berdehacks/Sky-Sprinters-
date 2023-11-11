import shapeRecognition
import numpy as np
import cv2


img = cv2.imread("prueba_2.jpg")

p1 = np.array([293, 759])
p2 = np.array([1306,  799])
p3 = np.array([209, 299])
p4 = np.array([1398, 300])

Q1 = p2 - p1
Q2 = p4 - p3

normalized_q1 = Q1 / np.linalg.norm(Q1)
normalized_q2 = Q2 / np.linalg.norm(Q2)
print(normalized_q1)
print(normalized_q2)

p = 0.3

shorten_Q1 = normalized_q1 * p  # adjust percentage of horizontal roi
shorten_Q2 = normalized_q2 * p  # adjust percentage of horizontal roi

PQ1 = p1 + shorten_Q1
PQ2 = p3 + shorten_Q2

PQ1Y = PQ1[1]
PQ2Y = PQ2[1]

p1y = p1[1]
p3y = p3[1]

print(p1, p2, PQ1)

minY = min(PQ2Y, p3y)
maxY = max(PQ1Y, p1y)

print(minY, maxY, p1[0], int(PQ2[0]))


img = img[p3[1]:p1[1], p1[0]:p1[0] + int(PQ1[0])]
# img = img[int(maxY):minY, int(PQ2[0]):p1[0]]

shape = shapeRecognition.shapeRecognition(img)
print(shape)

cv2.imshow("roi", img)
cv2.waitKey(0)
